import math
import pickle
import os
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from typing import Union, List, Optional, Any

# 假设之前的 unified_sequence 中定义了 RunMode, SequenceStatus
# 如果实际路径不同，请调整。这里统一从 nanovllm 导入
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, RunType, SequenceStatus, RunMode
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

# Models
# 假设这些模型都在 nanovllm.models 下，或者你需要根据实际路径调整
from nanovllm.models.qwen3 import Qwen3ForCausalLM
try:
    from nanovllm.models.sdar import SDARForCausalLM
    from nanovllm.models.llada import LladaForCausalLM
except ImportError:
    # 占位，如果用户只跑 AR 模式可能不需要这些
    SDARForCausalLM = None
    LladaForCausalLM = None
    DreamForCausalLM = None

from nanovllm.layers.sampler import Sampler

class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.gpu_offset = getattr(config, 'gpu_offset', 0)
        self.rank = rank
        self.device_id = rank + self.gpu_offset
        self.event = event

        # Initialize Distributed Environment
        # 注意: 这里沿用了 AR 代码的 init 方式，如果是单卡或由外部启动 dist，需调整
        if not dist.is_initialized():
             dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        
        torch.cuda.set_device(self.device_id)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")

        # --- Model Selection Logic ---
        model_kwargs = {"config": hf_config}
        model_type = hf_config.model_type.lower()
        
        if "qwen" in model_type:
            self.ModelClass = Qwen3ForCausalLM
        elif "sdar" in model_type:
            if "moe" in model_type:
                raise ValueError("MoE not supported for dp tp hybrid yet")
            self.ModelClass = SDARForCausalLM
        elif "llada" in model_type:
            self.ModelClass = LladaForCausalLM
        elif "dream" in model_type:
            self.ModelClass = DreamForCausalLM
        else:
            # Fallback or raise
            raise ValueError(f"Unsupported model type: {hf_config.model_type}")

        self.model = self.ModelClass(**model_kwargs)
        load_model(self.model, config.model)
        
        # Sampler for AR mode
        self.sampler = Sampler()

        # Warmup & Alloc
        self.warmup_model()
        self.allocate_kv_cache()
        
        # CUDA Graphs
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # Shared Memory / Worker Loop (AR Style)
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    # -------------------------------------------------------------------------
    # System / IPC Methods
    # -------------------------------------------------------------------------

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool, self.graph_vars
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    # -------------------------------------------------------------------------
    # Initialization & Memory
    # -------------------------------------------------------------------------

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens = getattr(self.config, 'max_num_batched_tokens', 8192)
        max_model_len = self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        
        # Determine how to create dummy sequences based on capability
        # AR warmup usually sufficient to warm up allocator
        seqs = [Sequence([0] * max_model_len, sampling_params=None, mode=RunMode.AR) for _ in range(num_seqs)]
        self.run(seqs, RunType.PREFILL) # Generic warmup
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        
        # Reserve some buffer
        available_mem = total * config.gpu_memory_utilization - used - peak + current
        config.num_kvcache_blocks = int(available_mem) // block_bytes
        
        if config.num_kvcache_blocks <= 0:
            raise RuntimeError(f"OOM: Not enough memory for KV cache. Block bytes: {block_bytes}, Avail: {available_mem}")

        if self.rank == 0:
            print(f"[KV Cache] Allocated {config.num_kvcache_blocks:,} blocks. Size: {config.num_kvcache_blocks * block_bytes / 1024**3:.2f} GB")

        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    # -------------------------------------------------------------------------
    # Core Helpers
    # -------------------------------------------------------------------------

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        if max_len == 0: return None
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        top_ps = []
        for seq in seqs:
            temperatures.append(seq.temperature)
            top_ps.append(seq.top_p)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        top_ps = torch.tensor(top_ps, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures, top_ps

    # -------------------------------------------------------------------------
    # AR Pipeline (Nanovllm Style)
    # -------------------------------------------------------------------------

    def prepare_prefill_ar(self, seqs: list[Sequence]):
        input_ids, positions, slot_mapping = [], [], []
        cu_seqlens_q, cu_seqlens_k = [0], [0]
        max_seqlen_q, max_seqlen_k = 0, 0
        block_tables = None
        
        for seq in seqs:
            seqlen = len(seq)
            # Only process new tokens
            new_part = seq[seq.num_cached_tokens:]
            input_ids.extend(new_part)
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            if not seq.block_table: continue
            
            # Map slots for the new part
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                is_last_block = (i == seq.num_blocks - 1)
                end = start + (seq.last_block_num_tokens if is_last_block else self.block_size)
                slot_mapping.extend(list(range(start, end)))
                
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        # To GPU
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        set_context(
            run_type=RunType.PREFILL, # AR prefill is compatible with this enum
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping, block_tables=block_tables
        )
        return input_ids, positions

    def prepare_decode_ar(self, seqs: list[Sequence]):
        input_ids, positions, slot_mapping, context_lens = [], [], [], []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
            
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        
        set_context(
            run_type=RunType.DENOISE, # AR Decode maps to DENOISE/DECODE context
            slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables
        )
        return input_ids, positions

    # -------------------------------------------------------------------------
    # Diffusion Pipeline (JetEngine Style)
    # -------------------------------------------------------------------------

    def prepare_prefill_diffusion(self, seqs: list[Sequence]):
        # Diffusion prefill differs: it often processes chunk-aligned segments
        # and has complex slot mapping due to potential block alignment issues
        input_ids, positions, is_last_step = [], [], []
        cu_seqlens_q = [0]
        max_seqlen_q = 0
        slot_mapping = []
        
        for seq in seqs:
            seqlen = len(seq.token_ids) # Use full token_ids (prefill part)
            input_ids.extend(seq.token_ids)
            positions.extend(range(seqlen))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen)
            max_seqlen_q = max(max_seqlen_q, seqlen)
            is_last_step.append(False)
            
            if not seq.block_table: continue
            
            # Map every token to a physical slot
            for i in range(seqlen):
                block_idx = i // self.block_size
                block_offset = i % self.block_size
                physical_block_id = seq.block_table[block_idx]
                slot = physical_block_id * self.block_size + block_offset
                slot_mapping.append(slot)
                
        device = torch.device("cuda")
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        is_last_step_t = torch.tensor(is_last_step, dtype=torch.bool).to(device, non_blocking=True)

        set_context(
            run_type=RunType.PREFILL,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_q,
            slot_mapping=slot_mapping, 
            is_last_denoise_step=is_last_step_t,
            block_length=getattr(self.config, 'block_length', 128)
        )
        return input_ids, positions

    def prepare_denoise_diffusion(self, seqs: list[Sequence]):
        block_len = len(seqs[0].intermediate_block_tokens)
        device = torch.device("cuda")
        
        cached_lens = torch.tensor([len(seq) for seq in seqs], dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        
        input_ids_list = [seq.intermediate_block_tokens for seq in seqs]
        input_ids = torch.tensor(input_ids_list, dtype=torch.int64, pin_memory=True).view(-1).to(device, non_blocking=True)
        
        # Calculate global positions: start of cache + offset in block
        start_positions = cached_lens.unsqueeze(1)
        offsets = torch.arange(block_len, dtype=torch.int64, device=device).unsqueeze(0)
        positions = (start_positions + offsets).view(-1)
        
        block_tables = self.prepare_block_tables(seqs)

        set_context(
            run_type=RunType.DENOISE,
            context_lens=cached_lens,
            block_tables=block_tables,
            block_length=getattr(self.config, 'block_length', 128)
        )
        return input_ids, positions

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, 
                  run_type: RunType, mode: RunMode, batch_size: int = 0):
        
        # Case 1: Prefill (Generic) or Eager Mode -> Direct Forward
        if run_type == RunType.PREFILL or self.enforce_eager:
            return self.model.compute_logits(self.model(input_ids, positions))
        
        # Case 2: CUDA Graph Execution
        # We need to distinct between AR Graph (BS) and Diffusion Graph (BS * L)
        
        context = get_context()
        graph = None
        
        if mode == RunMode.AR:
            # Simple AR Decode Graph
            if input_ids.size(0) > 512: # Threshold from original code
                 return self.model.compute_logits(self.model(input_ids, positions))
            
            bs = input_ids.size(0)
            target_graph_bs = next((x for x in self.graph_bs_ar if x >= bs), None)
            if target_graph_bs:
                graph = self.graphs_ar[target_graph_bs]
                graph_vars = self.graph_vars_ar
                
                # Copy inputs
                graph_vars["input_ids"][:bs] = input_ids
                graph_vars["positions"][:bs] = positions
                graph_vars["slot_mapping"].fill_(-1)
                graph_vars["slot_mapping"][:bs] = context.slot_mapping
                graph_vars["context_lens"].zero_()
                graph_vars["context_lens"][:bs] = context.context_lens
                graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
                
                graph.replay()
                return self.model.compute_logits(graph_vars["outputs"][:bs])

        elif mode == RunMode.DIFFUSION:
            # Diffusion Denoise Graph
            # Batch size passed in is number of sequences, but input_ids is (B * L)
            if batch_size == 0: batch_size = len(input_ids) // getattr(self.config, 'block_length', 1)
            
            target_graph_bs = next((x for x in self.graph_bs_diff if x >= batch_size), None)
            
            # Check conditions for graph validity
            if target_graph_bs:
                graph = self.graphs_diff.get(target_graph_bs)
                graph_vars = self.graph_vars_diff
                block_len = self.config.block_length
                global_bs = batch_size * block_len
                
                # Check limits
                if global_bs <= graph_vars["input_ids"].shape[0] and context.context_lens is not None:
                     graph_vars["input_ids"][:global_bs].copy_(input_ids)
                     graph_vars["positions"][:global_bs].copy_(positions)
                     graph_vars["context_lens"][:batch_size].copy_(context.context_lens)
                     
                     graph_vars["block_tables"][:batch_size].fill_(-1)
                     required_blocks = context.block_tables.shape[1]
                     if required_blocks <= graph_vars["block_tables"].shape[1]:
                         graph_vars["block_tables"][:batch_size, :required_blocks].copy_(context.block_tables)
                         
                         graph.replay()
                         return self.model.compute_logits(graph_vars["outputs"][:global_bs])

        # Fallback to eager if graph conditions not met
        return self.model.compute_logits(self.model(input_ids, positions))

    def run(self, seqs: list[Sequence], run_type: RunType) -> Union[list[int], torch.Tensor, None]:
        if not seqs: return None
        
        # Assume homogeneous batch
        mode = seqs[0].mode
        
        # 1. Prepare Inputs
        if mode == RunMode.AR:
            if run_type == RunType.PREFILL:
                input_ids, positions = self.prepare_prefill_ar(seqs)
            else:
                input_ids, positions = self.prepare_decode_ar(seqs)
        else: # Diffusion
            if run_type == RunType.PREFILL:
                input_ids, positions = self.prepare_prefill_diffusion(seqs)
            else:
                input_ids, positions = self.prepare_denoise_diffusion(seqs)

        # 2. Run Model (Compute Logits)
        logits = self.run_model(input_ids, positions, run_type, mode, batch_size=len(seqs))

        # 3. Post-Process / Sampling
        # AR: Sample token IDs here to avoid sending logits
        if mode == RunMode.AR:
            if self.rank == 0:
                temperatures, top_ps = self.prepare_sample(seqs)
                token_ids = self.sampler(logits, temperatures, top_ps).tolist()
                reset_context()
                return token_ids
            else:
                reset_context()
                return None
        
        # Diffusion: Return logits to scheduler for complex remasking
        else:
            reset_context()
            return logits if self.rank == 0 else None

    # -------------------------------------------------------------------------
    # Parameter Sync (AR Speculative Feature)
    # -------------------------------------------------------------------------

    def update_model_param_with_broadcast(self, param_dict: dict[str, torch.Tensor]):
        if self.world_size == 1:
            self._apply_weights_with_loader(param_dict)
            return
        self._pending_param_dict = param_dict
        param_names = list(param_dict.keys())
        shapes = [list(t.shape) for t in param_dict.values()]
        dtypes = [str(t.dtype) for t in param_dict.values()]
        self.call("_receive_and_apply_weights", param_names, shapes, dtypes)

    def _receive_and_apply_weights(self, param_names, shapes, dtypes):
        param_dict = getattr(self, '_pending_param_dict', None)
        packed_modules_mapping = getattr(self.model, "packed_modules_mapping", {})
        
        for i, weight_name in enumerate(param_names):
            shape = torch.Size(shapes[i])
            dtype = getattr(torch, dtypes[i].replace('torch.', ''))
            
            if self.rank == 0 and param_dict:
                full_weight = param_dict[weight_name].to(device='cuda', dtype=dtype)
            else:
                full_weight = torch.empty(shape, dtype=dtype, device='cuda')
            
            dist.broadcast(full_weight, src=0)
            self._apply_single_weight(weight_name, full_weight, packed_modules_mapping)
            del full_weight
            
        dist.barrier()
        if hasattr(self, '_pending_param_dict'): del self._pending_param_dict

    def _apply_weights_with_loader(self, param_dict):
        packed_modules_mapping = getattr(self.model, "packed_modules_mapping", {})
        for name, tensor in param_dict.items():
            self._apply_single_weight(name, tensor, packed_modules_mapping)

    def _apply_single_weight(self, weight_name, tensor, packed_modules_mapping):
        # Implementation of weight loader application
        for k, (v, shard_id) in packed_modules_mapping.items():
            if k in weight_name:
                param_name = weight_name.replace(k, v)
                try:
                    param = self.model.get_parameter(param_name)
                    if hasattr(param, "weight_loader"):
                        with torch.no_grad(): param.weight_loader(param, tensor, shard_id)
                except AttributeError: pass
                break
        else:
            try:
                param = self.model.get_parameter(weight_name)
                if hasattr(param, "weight_loader"):
                    with torch.no_grad(): param.weight_loader(param, tensor)
                else:
                    with torch.no_grad(): param.data.copy_(tensor.to(param.device, param.dtype))
            except AttributeError: pass

    # -------------------------------------------------------------------------
    # CUDA Graph Capture
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def capture_cudagraph(self):
        # We need two separate capture routines or memory pools
        # For simplicity, we capture both sets if capability allows, 
        # or we could lazy-init. Here we capture both at startup.
        
        self.capture_cudagraph_ar()
        
        # Only capture Diffusion graphs if model supports it or config implies it
        # Check if block_length is defined, implying diffusion potential
        if hasattr(self.config, 'block_length') and self.config.block_length > 1:
            self.capture_cudagraph_diffusion()

    def capture_cudagraph_ar(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size, dtype=hf_config.dtype)
        
        self.graph_bs_ar = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs_ar = {}
        self.graph_pool_ar = None

        for bs in reversed(self.graph_bs_ar):
            graph = torch.cuda.CUDAGraph()
            # Context setup for capture
            set_context(
                run_type=RunType.DENOISE, # Maps to decode context logic in kernels
                slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs]
            )
            
            # Warmup & Capture
            self.model(input_ids[:bs], positions[:bs]) 
            with torch.cuda.graph(graph, self.graph_pool_ar):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
                
            if self.graph_pool_ar is None: self.graph_pool_ar = graph.pool()
            self.graphs_ar[bs] = graph
            reset_context()

        self.graph_vars_ar = dict(
            input_ids=input_ids, positions=positions, slot_mapping=slot_mapping,
            context_lens=context_lens, block_tables=block_tables, outputs=outputs
        )

    def capture_cudagraph_diffusion(self):
        config = self.config
        block_len = config.block_length
        max_bs = min(config.max_num_seqs, 256)
        max_global_bs = max_bs * block_len
        max_num_blocks = math.ceil((config.max_model_len + block_len) / self.block_size) + 1
        
        input_ids = torch.zeros(max_global_bs, dtype=torch.int64)
        positions = torch.zeros(max_global_bs, dtype=torch.int64)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_global_bs, config.hidden_size, dtype=config.hf_config.dtype)
        
        self.graph_bs_diff = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs_diff = {}
        self.graph_pool_diff = None

        for bs in reversed(self.graph_bs_diff):
            graph = torch.cuda.CUDAGraph()
            # Diffusion Context
            set_context(
                run_type=RunType.DENOISE,
                context_lens=context_lens[:bs], block_tables=block_tables[:bs], block_length=block_len
            )
            global_bs = bs * block_len
            
            self.model(input_ids[:global_bs], positions[:global_bs])
            with torch.cuda.graph(graph, self.graph_pool_diff):
                outputs[:global_bs] = self.model(input_ids[:global_bs], positions[:global_bs])
            
            if self.graph_pool_diff is None: self.graph_pool_diff = graph.pool()
            self.graphs_diff[bs] = graph
            reset_context()
            
        self.graph_vars_diff = dict(
            input_ids=input_ids, positions=positions, context_lens=context_lens,
            block_tables=block_tables, outputs=outputs
        )