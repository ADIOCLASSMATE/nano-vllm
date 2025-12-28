import math
import pickle
import os
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from jetengine.config import Config
from jetengine.engine.sequence import Sequence, RunType, SequenceStatus
from jetengine.models.sdar import SDARForCausalLM
from jetengine.models.sdar_moe import SDARMoeForCausalLM
from jetengine.models.llada import LladaForCausalLM
from jetengine.utils.context import set_context, get_context, reset_context
from jetengine.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.gpu_offset = config.gpu_offset
        self.rank = rank
        self.device_id = rank + self.gpu_offset  # Actual GPU device ID
        self.event = event

        # Manually initialize NCCL process group (like nanovllm)
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(self.device_id)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.torch_dtype)
        torch.set_default_device("cuda")

        # Create process group for tensor parallelism
        if self.world_size > 1:
            ranks = list(range(self.world_size))
            self.tp_group = dist.new_group(ranks)
        else:
            self.tp_group = None

        model_kwargs = {"config": hf_config,
                        "process_group": self.tp_group}
        if "sdar" in hf_config.model_type and "moe" in hf_config.model_type:
            raise ValueError(f"MoE not supported for dp tp hybrid yet")
            self.ModelClass = SDARMoeForCausalLM
        elif "sdar" in hf_config.model_type:
            self.ModelClass = SDARForCausalLM
        elif "llada" in hf_config.model_type.lower():
            self.ModelClass = LladaForCausalLM
        else:
            raise ValueError(f"Unsupported model type: {hf_config.model_type}")
        self.model = self.ModelClass(**model_kwargs)
        load_model(self.model, config.model)
        # Sampler is removed from here
        self.warmup_model()
        self.allocate_kv_cache()
        # CUDA graph capture for block diffusion is complex and omitted for this example
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # Setup shared memory for multi-process communication (like nanovllm)
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="jetengine", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="jetengine")
                self.loop()
        
    def reinit_model(self):
        self.model = self.ModelClass(
            self.config.hf_config, self.tp_group)

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """Loop for non-rank-0 processes to receive commands via shared memory."""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """Read command from shared memory (for non-rank-0 processes)."""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """Write command to shared memory (for rank-0 process)."""
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """Call method on all processes via shared memory."""
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens //
                       max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len, self.config.mask_token_id)
                for _ in range(num_seqs)]
        self.run(seqs, RunType.PREFILL)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = config.num_key_value_heads // self.world_size
        block_bytes = 2 * config.num_hidden_layers * self.block_size * \
            num_kv_heads * config.head_dim * config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(
            total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        print(f"[rank {dist.get_rank()}][KVCache] Allocated {config.num_kvcache_blocks:,} blocks "
        f"({config.num_kvcache_blocks * block_bytes / (1024**3):.2f} GiB) "
        f"based on peak memory usage.")
        self.kv_cache = torch.zeros(2, config.num_hidden_layers, config.num_kvcache_blocks,
                                    self.block_size, num_kv_heads, config.head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        if max_len == 0:
            return None
        block_tables = [seq.block_table + [-1] *
                        (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables
    def prepare_prefill_loop(self, seqs: list[Sequence]):
        input_ids, positions, cu_seqlens_q, slot_mapping, is_last_step = [], [], [0], [], []
        max_seqlen_q = 0
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq.token_ids)
            positions.extend(range(seqlen))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen)
            max_seqlen_q = max(max_seqlen_q, seqlen)
            is_last_step.append(False)
            # Slot mapping for prefill
            if not seq.block_table:
                continue
            # Slot mapping for prefill
            if not seq.block_table:
                continue
            for i in range(seqlen):
                block_idx = i // self.block_size
                block_offset = i % self.block_size
                physical_block_id = seq.block_table[block_idx]
                slot = physical_block_id * self.block_size + block_offset
                slot_mapping.append(slot)

        device = torch.device("cuda")
        input_ids_cpu = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True)
        positions_cpu = torch.tensor(positions, dtype=torch.int64, pin_memory=True)
        cu_seqlens_q_cpu = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True)
        slot_mapping_cpu = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True)

        input_ids = input_ids_cpu.to(device=device, non_blocking=True)
        positions = positions_cpu.to(device=device, non_blocking=True)
        cu_seqlens_q = cu_seqlens_q_cpu.to(device=device, non_blocking=True)
        slot_mapping = slot_mapping_cpu.to(device=device, non_blocking=True)
        set_context(
            run_type=RunType.PREFILL,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            slot_mapping=slot_mapping,
            is_last_denoise_step=is_last_step,  # <-- Pass the new flag
            block_length=self.config.block_length
        )
        return input_ids, positions

    def prepare_prefill(self, seqs: list[Sequence]):
        device = torch.device("cuda") 
        all_token_ids_flat = []
        all_block_tables_flat = []
        seqlens_list = []
        block_table_lens_list = []
        is_last_step_list = []

        for seq in seqs:
            seq_len = len(seq.token_ids)
            seqlens_list.append(seq_len)
            all_token_ids_flat.extend(seq.token_ids)
            is_last_step_list.append(False)

            if seq.block_table:
                block_table_lens_list.append(len(seq.block_table))
                all_block_tables_flat.extend(seq.block_table)
            else:
                block_table_lens_list.append(0)
        
        # (total_tokens,)
        input_ids_cpu = torch.tensor(all_token_ids_flat, dtype=torch.int64, pin_memory=True)
        input_ids = input_ids_cpu.to(device=device, non_blocking=True)
        
        # (total_physical_blocks,)
        flat_block_tables_cpu = torch.tensor(
            all_block_tables_flat, dtype=torch.int32, pin_memory=True
        )
        flat_block_tables = flat_block_tables_cpu.to(device=device, non_blocking=True)
        
        # (B,)
        seqlens_q_cpu = torch.tensor(seqlens_list, dtype=torch.int32, pin_memory=True)
        block_table_lens_cpu = torch.tensor(
            block_table_lens_list, dtype=torch.int32, pin_memory=True
        )
        is_last_step_cpu = torch.tensor(is_last_step_list, dtype=torch.bool, pin_memory=True)

        seqlens_q = seqlens_q_cpu.to(device=device, non_blocking=True)
        block_table_lens = block_table_lens_cpu.to(device=device, non_blocking=True)
        is_last_step = is_last_step_cpu.to(device=device, non_blocking=True)
        
        batch_size = len(seqs)
        if batch_size == 0:
            # Handle empty batch case
            cu_seqlens_q_cpu = torch.tensor([0], dtype=torch.int32, pin_memory=True)
            cu_seqlens_q = cu_seqlens_q_cpu.to(device=device, non_blocking=True)
            max_seqlen_q = 0
            positions_cpu = torch.empty(0, dtype=torch.int64, pin_memory=True)
            slot_mapping_cpu = torch.empty(0, dtype=torch.int32, pin_memory=True)
            positions = positions_cpu.to(device=device, non_blocking=True)
            slot_mapping = slot_mapping_cpu.to(device=device, non_blocking=True)
        else:
            # (B+1,)
            cu_seqlens_q = torch.nn.functional.pad(
                seqlens_q.cumsum(dim=0, dtype=torch.int32), (1, 0)
            )
            cu_block_table_lens = torch.nn.functional.pad(
                block_table_lens.cumsum(dim=0, dtype=torch.int32), (1, 0)
            )
            max_seqlen_q = seqlens_q.max().item()
            total_tokens = input_ids.shape[0]
            token_indices_global = torch.arange(total_tokens, dtype=torch.int64, device=device)
            seq_start_offsets = torch.repeat_interleave(
                cu_seqlens_q[:-1], repeats=seqlens_q
            )
            positions = token_indices_global - seq_start_offsets # This is `i`
            
            # [True, False, True, ...] (B,)
            has_block_table_mask_per_seq = (block_table_lens > 0)
            # [T, T, T, F, F, T, T, T, T, ...] (total_tokens,)
            has_block_table_mask_per_token = torch.repeat_interleave(
                has_block_table_mask_per_seq, repeats=seqlens_q
            )
            # Filter `positions` to only those that need a slot
            # (total_slots,)
            i = positions[has_block_table_mask_per_token]
            # (total_tokens,)
            seq_idx_for_each_token = torch.repeat_interleave(
                torch.arange(batch_size, device=device), repeats=seqlens_q
            )
            seq_idx_with_slot = seq_idx_for_each_token[has_block_table_mask_per_token]
            block_idx = i // self.block_size
            block_offset = i % self.block_size
            block_table_start_offsets = cu_block_table_lens[seq_idx_with_slot]
            block_table_global_idx = block_table_start_offsets + block_idx
            physical_block_id = flat_block_tables[block_table_global_idx]
            slot_mapping = physical_block_id * self.block_size + block_offset
        set_context(
            run_type=RunType.PREFILL,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            slot_mapping=slot_mapping.to(torch.int32), # Ensure int32
            is_last_denoise_step=is_last_step,
            block_length=self.config.block_length
        )
        return input_ids, positions

    def prepare_denoise_loop(self, seqs: list[Sequence]):
        input_ids, positions = [], []
        cached_lens = []

        for seq in seqs:
            # The query is the current intermediate block
            q_tokens = seq.intermediate_block_tokens
            q_len = len(q_tokens)

            # The context (key/value) is the confirmed part of the sequence
            k_len = len(seq)

            input_ids.extend(q_tokens)
            # Positions are global
            positions.extend(range(k_len, k_len + q_len))
            cached_lens.append(k_len)

        device = torch.device("cuda")
        input_ids_cpu = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True)
        positions_cpu = torch.tensor(positions, dtype=torch.int64, pin_memory=True)
        cached_lens_cpu = torch.tensor(cached_lens, dtype=torch.int32, pin_memory=True)

        input_ids = input_ids_cpu.to(device=device, non_blocking=True)
        positions = positions_cpu.to(device=device, non_blocking=True)
        cached_lens = cached_lens_cpu.to(device=device, non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        set_context(
            run_type=RunType.DENOISE,
            context_lens=cached_lens,
            block_tables=block_tables,
            block_length=self.config.block_length
        )

        return input_ids, positions
    
    def prepare_denoise(self, seqs: list[Sequence]):
        device = torch.device("cuda")
        
        # Optimization: Stack tensors directly from sequences (assumed to be on device or easily movable)
        # This avoids creating a large list of integers and then converting to tensor on CPU
        block_tokens_list = []
        for seq in seqs:
            t = seq.intermediate_block_tokens
            if t.device != device:
                t = t.to(device, non_blocking=True)
                seq.intermediate_block_tokens = t # Update sequence state for future steps
            block_tokens_list.append(t)
            
        input_ids = torch.stack(block_tokens_list).view(-1)

        # Create cached_lens directly on device if possible, or move efficiently
        # len(seq) is fast (list len), so creating tensor on CPU then moving is standard for small batches
        # But we can try to avoid pin_memory overhead for small tensors if we want, 
        # though pin_memory is generally good. 
        # Let's stick to the pattern but ensure it's efficient.
        cached_lens = torch.tensor(
            [len(seq) for seq in seqs], 
            dtype=torch.int32, 
            device=device
        )
        
        block_len = seqs[0].block_length
        start_positions = cached_lens.unsqueeze(1)
        offsets = torch.arange(
            block_len, 
            dtype=torch.int64, 
            device=device
        ).unsqueeze(0)

        positions = (start_positions + offsets).view(-1)
        block_tables = self.prepare_block_tables(seqs)

        set_context(
            run_type=RunType.DENOISE,
            context_lens=cached_lens,
            block_tables=block_tables,
            block_length=self.config.block_length
        )

        return input_ids, positions


    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor):
        return self.model.compute_logits(self.model(input_ids, positions))

    @torch.inference_mode()
    def _run_denoise_with_cudagraph(
        self,
        seqs: list[Sequence],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor | None, bool]:
        if (
            self.enforce_eager
            or not hasattr(self, "graphs")
            or not getattr(self, "graphs", None)
        ):
            return None, False

        context = get_context()
        if context.run_type != RunType.DENOISE:
            return None, False

        batch_size = len(seqs)
        if batch_size == 0:
            return None, False

        graph = self.graphs.get(batch_size)
        if graph is None:
            return None, False

        graph_vars = self.graph_vars
        block_len = self.config.block_length
        global_bs = batch_size * block_len

        if (
            global_bs > graph_vars["input_ids"].shape[0]
            or context.context_lens is None
        ):
            return None, False

        graph_vars["input_ids"][:global_bs].copy_(input_ids)
        graph_vars["positions"][:global_bs].copy_(positions)
        graph_context_lens = graph_vars["context_lens"][:batch_size]
        graph_context_lens.copy_(context.context_lens)

        graph_block_tables = graph_vars["block_tables"][:batch_size]
        graph_block_tables.fill_(-1)
        if context.block_tables is not None:
            required_blocks = context.block_tables.shape[1]
            if required_blocks > graph_block_tables.shape[1]:
                if self.rank == 0:
                    # print(
                    #     "[CUDAGraph] Block table requirement exceeds captured capacity; "
                    #     "falling back to eager execution."
                    # )
                    pass
                return None, False
            graph_block_tables[:, :required_blocks].copy_(context.block_tables)

        set_context(
            run_type=RunType.DENOISE,
            context_lens=graph_context_lens,
            block_tables=graph_block_tables,
            block_length=self.config.block_length,
            is_last_denoise_step=context.is_last_denoise_step,
        )

        graph.replay()
        hidden_states = graph_vars["outputs"][:global_bs]
        logits = self.model.compute_logits(hidden_states)
        return logits, True

    def run(self, seqs: list[Sequence], run_type: RunType) -> torch.Tensor:
        if run_type == RunType.PREFILL:
            input_ids, positions = self.prepare_prefill(seqs)
        elif run_type == RunType.DENOISE:
            input_ids, positions = self.prepare_denoise(seqs)
        else:
            return None

        if run_type == RunType.DENOISE and not self.enforce_eager:
            logits, used_graph = self._run_denoise_with_cudagraph(
                seqs, input_ids, positions
            )
            if not used_graph:
                logits = self.run_model(input_ids, positions)
        else:
            logits = self.run_model(input_ids, positions)
        reset_context()
        return logits if self.rank == 0 else None

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        max_bs = min(self.config.max_num_seqs, 256)
        max_global_bs = max_bs * self.config.block_length
        max_num_blocks = math.ceil(
            (config.max_model_len + self.config.block_length) / self.block_size
        ) + 1
        input_ids = torch.zeros(max_global_bs, dtype=torch.int64)
        positions = torch.zeros(max_global_bs, dtype=torch.int64)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(
            max_global_bs, config.hidden_size, dtype=config.torch_dtype
        )
        self.graph_bs = [bs for bs in [1, 2, 4, 8] if bs <= max_bs] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(run_type=RunType.DENOISE,
                        context_lens=context_lens[:bs], block_tables=block_tables[:bs], block_length=self.config.block_length)
            global_bs = bs * self.config.block_length
            outputs[:global_bs] = self.model(
                input_ids[:global_bs], positions[:global_bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:global_bs] = self.model(
                    input_ids[:global_bs], positions[:global_bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    def update_model_param_with_broadcast(self, param_dict: dict[str, torch.Tensor]):
        """Update model parameters by broadcasting full weights via NCCL.
        
        Rank 0 sends full (unsharded) weights to all ranks via NCCL broadcast.
        Each rank then applies its own weight_loader to handle tensor parallel sharding.
        
        Args:
            param_dict: A dictionary mapping parameter names to new tensor values.
                       These should be FULL (unsharded) weights from the source model.
        """
        if self.world_size == 1:
            # Single GPU: use weight_loader directly
            self._apply_weights_with_loader(param_dict)
            return
        
        # Multi-GPU: store param_dict for rank 0 to use during broadcast
        self._pending_param_dict = param_dict
        
        # Signal other ranks to start receiving (send param info via shared memory)
        param_names = list(param_dict.keys())
        shapes = [list(t.shape) for t in param_dict.values()]
        dtypes = [str(t.dtype) for t in param_dict.values()]
        
        # call() will trigger _receive_and_apply_weights on all ranks including rank 0
        self.call("_receive_and_apply_weights", param_names, shapes, dtypes)

    def _receive_and_apply_weights(self, param_names: list[str], shapes: list[list[int]], dtypes: list[str]):
        """Receive weights via NCCL broadcast and apply with weight_loader.
        
        Called on all ranks. Rank 0 broadcasts, other ranks receive.
        """
        # Get param_dict from pending storage (only rank 0 has it)
        param_dict = getattr(self, '_pending_param_dict', None)
        
        packed_modules_mapping = getattr(self.model, "packed_modules_mapping", {})
        
        for i, weight_name in enumerate(param_names):
            shape = torch.Size(shapes[i])
            dtype = getattr(torch, dtypes[i].replace('torch.', ''))
            
            # Create buffer for receiving
            if self.rank == 0 and param_dict is not None:
                # Rank 0: use the actual tensor
                full_weight = param_dict[weight_name].to(device='cuda', dtype=dtype)
            else:
                # Other ranks: create empty buffer to receive
                full_weight = torch.empty(shape, dtype=dtype, device='cuda')
            
            # Broadcast from rank 0 (use default process group, like nanovllm)
            dist.broadcast(full_weight, src=0)
            
            # Now apply with weight_loader (each rank shards differently based on tp_rank)
            self._apply_single_weight(weight_name, full_weight, packed_modules_mapping)
            
            # Free memory
            del full_weight
        
        dist.barrier()
        
        # Clear pending param_dict
        if hasattr(self, '_pending_param_dict'):
            del self._pending_param_dict

    def _apply_weights_with_loader(self, param_dict: dict[str, torch.Tensor]):
        """Apply weights using weight_loader (single GPU path)."""
        packed_modules_mapping = getattr(self.model, "packed_modules_mapping", {})
        for weight_name, tensor in param_dict.items():
            self._apply_single_weight(weight_name, tensor, packed_modules_mapping)

    def _apply_single_weight(self, weight_name: str, tensor: torch.Tensor, packed_modules_mapping: dict):
        """Apply a single weight using the appropriate weight_loader."""
        from jetengine.utils.loader import _is_moe_expert_weight, _load_expert_weight_to_fused
        
        # Check if this weight is part of a packed module (e.g., q_proj -> qkv_proj)
        for k, (v, shard_id) in packed_modules_mapping.items():
            if k in weight_name:
                param_name = weight_name.replace(k, v)
                # Handle MoE expert weights
                if _is_moe_expert_weight(param_name):
                    try:
                        _load_expert_weight_to_fused(self.model, param_name, tensor, shard_id)
                    except (AttributeError, KeyError):
                        pass
                else:
                    try:
                        param = self.model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader", None)
                        if weight_loader is not None:
                            with torch.no_grad():
                                weight_loader(param, tensor, shard_id)
                    except AttributeError:
                        pass
                break
        else:
            # Not a packed module
            # Handle MoE expert weights
            if _is_moe_expert_weight(weight_name):
                try:
                    _load_expert_weight_to_fused(self.model, weight_name, tensor)
                except (AttributeError, KeyError):
                    pass
            else:
                # Load directly with weight_loader
                try:
                    param = self.model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", None)
                    if weight_loader is not None:
                        with torch.no_grad():
                            weight_loader(param, tensor)
                    else:
                        # Fallback: direct copy (for non-sharded params like layernorm)
                        with torch.no_grad():
                            param.data.copy_(tensor.to(param.device, param.dtype))
                except AttributeError:
                    pass