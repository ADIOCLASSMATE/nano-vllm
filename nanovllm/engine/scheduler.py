from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.utils.context import RunType


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.mask_token_id = config.mask_token_id
        self.config = config
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.denoise_ready: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running and not self.denoise_ready

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool, RunType]:
        """Schedule sequences for execution.
        
        Returns:
            (seqs, is_prefill, run_type)
            is_prefill: backward compatibility flag (True for PREFILL, False for DENOISE/DECODE)
            run_type: RunType.PREFILL, RunType.DENOISE, or standard DECODE
        """
        from nanovllm.utils.context import RunType
        
        # Prefill phase: process waiting sequences with prompt prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True, RunType.PREFILL

        # Denoise phase: process denoise blocks for LLADA/SDAR
        while self.denoise_ready and num_seqs < self.max_num_seqs:
            seq = self.denoise_ready.popleft()
            if not self.block_manager.can_append(seq):
                self.denoise_ready.appendleft(seq)
                break
            num_seqs += 1
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, False, RunType.DENOISE

        # Decode phase: standard token-by-token decoding for Qwen3 and other autoregressive models
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        
        if not scheduled_seqs:
            # No sequences available
            return [], False, RunType.PREFILL  # Return PREFILL as safe default
        
        self.running.extendleft(reversed(scheduled_seqs))
        # Determine run_type based on sequence properties
        # If sequence has intermediate_block_tokens (LLADA/SDAR), it's DENOISE
        # Otherwise it's standard DECODE (Qwen3)
        run_type = RunType.DENOISE if scheduled_seqs and scheduled_seqs[0].intermediate_block_tokens is not None else RunType.PREFILL
        return scheduled_seqs, False, run_type

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], data, run_type=None):
        """Process model outputs and update sequence states.
        
        Adapted from jetengine/engine/scheduler.py:462
        
        For PREFILL: data is token_ids list, update cached tokens
        For DENOISE: data is logits tensor (B*L, vocab_size), sample and update blocks
        For DECODE: data is token_ids list, append one token per sequence (standard autoregressive)
        """
        from nanovllm.utils.context import RunType
        
        if run_type is None:
            run_type = RunType.PREFILL
        
        if run_type == RunType.DENOISE:
            # data is logits tensor (B*L, vocab_size)
            self._postprocess_denoise(seqs, data)
        else:
            # data is token_ids list (PREFILL or standard DECODE)
            token_ids = data
            # Standard DECODE phase: one token per sequence (for Qwen3, etc.)
            for seq, token_id in zip(seqs, token_ids or []):
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    if seq in self.running:
                        self.running.remove(seq)
    
    def _postprocess_denoise(self, seqs: list[Sequence], logits: torch.Tensor):
        """Handle denoise phase sampling and token updates with advanced remasking strategies.
        
        Adapted from jetengine/engine/scheduler.py:462
        
        Supports multiple remasking strategies:
        - sequential: Transfer tokens sequentially from first mask position
        - low_confidence_static: Transfer top-k by confidence (static threshold)
        - low_confidence_dynamic: Dynamic threshold-based transfer
        - entropy_bounded: Transfer based on entropy cumsum
        - random: Random selection of mask tokens
        """
        import torch.nn.functional as F
        
        batch_size = len(seqs)
        block_len = seqs[0].block_length if seqs else 4
        device = logits.device
        EPS = 1e-12
        
        # Reshape logits: (B*L, V) -> (B, L, V)
        logits_reshaped = logits.view(batch_size, block_len, -1)
        
        # Apply temperature and softmax to get probabilities
        temperatures = torch.tensor(
            [seq.temperature for seq in seqs], 
            device=device, 
            dtype=torch.float32
        ).view(batch_size, 1, 1)
        
        logits_scaled = logits_reshaped / temperatures
        probs = F.softmax(logits_scaled, dim=-1)  # (B, L, V)
        
        # Calculate entropies: (B, L)
        entropies_all = -(probs.clamp_min(EPS) * (probs.clamp_min(EPS)).log()).sum(dim=-1)
        
        # Sample tokens from probabilities: (B, L)
        probs_flat = probs.view(-1, probs.shape[-1])  # (B*L, V)
        sampled_flat = torch.multinomial(probs_flat, num_samples=1).view(batch_size, block_len)
        batch_seq_x0 = sampled_flat.to(torch.int64)
        
        # Get probabilities and log-probabilities of sampled tokens: (B, L)
        batch_seq_x0_p = torch.gather(probs, -1, batch_seq_x0.unsqueeze(-1)).squeeze(-1)
        batch_seq_x0_logp = torch.log(batch_seq_x0_p.clamp_min(EPS))
        
        # Current tokens in intermediate blocks: (B, L)
        batch_current_tokens = torch.tensor(
            [seq.intermediate_block_tokens for seq in seqs],
            device=device,
            dtype=torch.int64
        )
        
        # Get existing trajectories, logprobs, entropies: (B, L)
        batch_logprobs = torch.tensor(
            [seq.block_logprobs or [0.0] * block_len for seq in seqs],
            device=device,
            dtype=torch.float32
        )
        
        batch_entropies = torch.tensor(
            [seq.block_entropies or [0.0] * block_len for seq in seqs],
            device=device,
            dtype=torch.float32
        )
        
        batch_trajectory = torch.tensor(
            [seq.block_trajectory or [0] * block_len for seq in seqs],
            device=device,
            dtype=torch.long
        )
        
        # Get num_to_transfer for each sequence
        num_to_transfer_list = [
            seq.num_transfer_tokens_per_step[seq.current_denoising_step] 
            if seq.current_denoising_step < len(seq.num_transfer_tokens_per_step) else 0
            for seq in seqs
        ]
        batch_num_to_transfer = torch.tensor(
            num_to_transfer_list,
            device=device,
            dtype=torch.long
        )
        
        # Global step for trajectory tracking
        batch_global_step_plus_1 = torch.tensor(
            [seq.global_denoising_step + 1 for seq in seqs],
            device=device,
            dtype=torch.long
        ).unsqueeze(1)  # (B, 1)
        
        # Status masks
        all_statuses = [seq.status for seq in seqs]
        denoising_mask_bool = torch.tensor(
            [s == SequenceStatus.DENOISING for s in all_statuses],
            device=device
        )
        saving_mask_bool = torch.tensor(
            [s == SequenceStatus.SAVING for s in all_statuses],
            device=device
        )
        
        denoising_mask = denoising_mask_bool.unsqueeze(1)  # (B, 1)
        
        # Mask of mask tokens in denoising sequences: (B, L)
        mask_token_mask = (batch_current_tokens == self.mask_token_id) & denoising_mask
        
        # Get remasking strategies for each sequence
        strategies = [
            seq.remasking_strategy if status == SequenceStatus.DENOISING else ''
            for seq, status in zip(seqs, all_statuses)
        ]
        
        # Create strategy masks
        seq_mask = torch.tensor(['sequential' in s for s in strategies], device=device).unsqueeze(1)
        low_conf_static_mask = torch.tensor(['low_confidence_static' in s for s in strategies], device=device).unsqueeze(1)
        low_conf_dynamic_mask = torch.tensor(['low_confidence_dynamic' in s for s in strategies], device=device).unsqueeze(1)
        entropy_bounded_mask = torch.tensor(['entropy_bounded' in s for s in strategies], device=device).unsqueeze(1)
        random_mask = torch.tensor([s == 'random' for s in strategies], device=device).unsqueeze(1)
        
        # Initialize transfer index: (B, L)
        transfer_index = torch.zeros((batch_size, block_len), dtype=torch.bool, device=device)
        
        # --- Strategy: 'sequential' ---
        if seq_mask.any():
            # Find first mask position for each sequence
            first_mask_pos = torch.argmax(mask_token_mask.int(), dim=1, keepdim=True)  # (B, 1)
            
            range_tensor = torch.arange(block_len, device=device).unsqueeze(0)  # (1, L)
            
            start_pos_b = first_mask_pos
            end_pos_b = (start_pos_b + batch_num_to_transfer.unsqueeze(1)).clamp_max(block_len)
            
            seq_transfer_index = (range_tensor >= start_pos_b) & (range_tensor < end_pos_b) & mask_token_mask
            transfer_index = torch.where(seq_mask, seq_transfer_index, transfer_index)
        
        # --- Strategy: 'low_confidence_static' ---
        if low_conf_static_mask.any():
            confidence = torch.where(mask_token_mask, batch_seq_x0_p, -torch.inf)
            
            max_k = batch_num_to_transfer.max().item()
            if max_k > 0:
                _, top_indices = torch.topk(confidence, k=max_k, dim=1)
                k_mask = torch.arange(max_k, device=device).unsqueeze(0) < batch_num_to_transfer.unsqueeze(1)
                static_transfer_index = torch.zeros_like(confidence, dtype=torch.bool).scatter_(1, top_indices, k_mask)
                transfer_index = torch.where(low_conf_static_mask, static_transfer_index, transfer_index)
        
        # --- Strategy: 'low_confidence_dynamic' ---
        if low_conf_dynamic_mask.any():
            dyn_thresholds = torch.tensor(
                [seq.dynamic_threshold if hasattr(seq, 'dynamic_threshold') else 0.9 for seq in seqs],
                device=device,
                dtype=torch.float32
            ).unsqueeze(1)  # (B, 1)
            
            confidence = torch.where(mask_token_mask, batch_seq_x0_p, -torch.inf)
            
            # Threshold-based transfer
            dyn_transfer_index = (confidence > dyn_thresholds)
            
            # Fallback to sequential if not enough tokens
            num_transferred_dyn = dyn_transfer_index.sum(dim=1)
            needs_fallback = (num_transferred_dyn < batch_num_to_transfer) & low_conf_dynamic_mask.squeeze(1)
            
            if needs_fallback.any():
                fallback_mask_token_mask = mask_token_mask[needs_fallback]
                fallback_num_to_transfer = batch_num_to_transfer[needs_fallback].unsqueeze(1)
                
                first_mask_pos = torch.argmax(fallback_mask_token_mask.int(), dim=1, keepdim=True)
                range_tensor = torch.arange(block_len, device=device).unsqueeze(0)
                
                start_pos_b = first_mask_pos
                end_pos_b = (start_pos_b + fallback_num_to_transfer).clamp_max(block_len)
                
                fallback_indices = (range_tensor >= start_pos_b) & (range_tensor < end_pos_b) & fallback_mask_token_mask
                dyn_transfer_index[needs_fallback] = fallback_indices
            
            transfer_index = torch.where(low_conf_dynamic_mask, dyn_transfer_index, transfer_index)
        
        # --- Strategy: 'entropy_bounded' ---
        if entropy_bounded_mask.any():
            masked_entropies = torch.where(mask_token_mask, entropies_all, torch.inf)
            ent_sorted, order = torch.sort(masked_entropies, dim=1, descending=False)
            ent_sorted_masked = torch.where(ent_sorted == torch.inf, 0.0, ent_sorted)
            
            cumsum = torch.cumsum(ent_sorted_masked, dim=1)
            
            eb_thresholds = torch.tensor(
                [seq.eb_threshold if hasattr(seq, 'eb_threshold') else 2.0 for seq in seqs],
                device=device,
                dtype=torch.float32
            ).unsqueeze(1)
            
            k_tensor = torch.searchsorted(cumsum, eb_thresholds, right=False)
            k_tensor.clamp_min_(1)
            
            k_mask_eb = torch.arange(block_len, device=device).unsqueeze(0) < k_tensor
            eb_transfer_index = torch.zeros_like(mask_token_mask, dtype=torch.bool).scatter_(1, order, k_mask_eb)
            
            transfer_index = torch.where(entropy_bounded_mask, eb_transfer_index, transfer_index)
        
        # --- Strategy: 'random' ---
        if random_mask.any():
            scores = torch.rand((batch_size, block_len), device=device)
            scores = scores.masked_fill(~mask_token_mask, -1.0)
            
            max_k = batch_num_to_transfer.max().item()
            if max_k > 0:
                _, top_indices = scores.topk(max_k, dim=-1)
                k_mask = torch.arange(max_k, device=device).unsqueeze(0) < batch_num_to_transfer.unsqueeze(1)
                random_transfer_index = torch.zeros_like(mask_token_mask, dtype=torch.bool).scatter_(1, top_indices, k_mask)
                transfer_index = torch.where(random_mask, random_transfer_index, transfer_index)
        
        # Final transfer index: only transfer mask tokens that passed strategy filter
        final_transfer_index = transfer_index & mask_token_mask
        
        # Update tokens: replace with sampled tokens where transfer_index is True
        batch_new_tokens = torch.where(final_transfer_index, batch_seq_x0, batch_current_tokens)
        
        # Update trajectory: record first step when token was accepted
        batch_new_trajectory = torch.where(
            final_transfer_index & (batch_trajectory == 0),
            batch_global_step_plus_1.squeeze(1),
            batch_trajectory
        )
        
        # Update logprobs and entropies
        batch_new_logprobs = torch.where(final_transfer_index, batch_seq_x0_logp, batch_logprobs)
        batch_new_entropies = torch.where(final_transfer_index, entropies_all, batch_entropies)
        
        # Calculate step transitions
        new_denoising_steps = torch.tensor(
            [seq.current_denoising_step for seq in seqs],
            device=device,
            dtype=torch.long
        ) + denoising_mask_bool.long()
        
        new_global_steps = torch.tensor(
            [seq.global_denoising_step for seq in seqs],
            device=device,
            dtype=torch.long
        ) + denoising_mask_bool.long()
        
        # Check if block is fully denoised
        is_fully_denoised = (~(batch_new_tokens == self.mask_token_id).any(dim=1)) | \
                            (new_denoising_steps >= torch.tensor(
                                [seq.denoising_steps for seq in seqs],
                                device=device,
                                dtype=torch.long
                            ))
        
        # Update sequence states
        for i, seq in enumerate(seqs):
            if denoising_mask_bool[i]:
                # Update tokens and metadata
                seq.intermediate_block_tokens = batch_new_tokens[i].cpu().tolist()
                seq.block_trajectory = batch_new_trajectory[i].cpu().tolist()
                seq.block_logprobs = batch_new_logprobs[i].cpu().tolist()
                seq.block_entropies = batch_new_entropies[i].cpu().tolist()
                
                # Update step counters
                seq.current_denoising_step = new_denoising_steps[i].item()
                seq.global_denoising_step = new_global_steps[i].item()
                
                # Record number of tokens transferred
                seq.num_to_transfer = final_transfer_index[i].sum().item()
                
                # Check if block is done
                if is_fully_denoised[i]:
                    seq.status = SequenceStatus.SAVING
            
            elif saving_mask_bool[i]:
                # Commit block to confirmed tokens
                seq.commit_block(seq.intermediate_block_tokens)
                seq.num_to_transfer = 0
                
                if seq.num_completion_tokens >= seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    if seq in self.running:
                        self.running.remove(seq)
                else:
                    # Prepare next denoise block
                    seq.start_new_block()
                    self.denoise_ready.append(seq)
        
        # Clean up finished sequences
        finished_seqs = [seq for seq in self.running if seq.is_finished]
        if finished_seqs:
            for seq in finished_seqs:
                self.block_manager.deallocate(seq)
            self.running = [seq for seq in self.running if not seq.is_finished]
