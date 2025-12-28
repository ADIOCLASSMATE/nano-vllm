from collections import deque

import torch

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
        # DECODE phase: use RunType.PREFILL for standard autoregressive decoding
        # (All decode-phase sequences go through standard token-by-token logic)
        return scheduled_seqs, False, RunType.PREFILL

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
        
        Adapted from jetengine/engine/scheduler.py:134 (postprocess_loop)
        
        Uses per-sequence processing (not fully batched) to handle varying sequence parameters.
        """
        import torch.nn.functional as F
        import numpy as np
        
        device = logits.device
        EPS = 1e-12
        
        # Process each sequence individually
        start_idx = 0
        for seq in seqs:
            if seq.status != SequenceStatus.DENOISING:
                continue
            
            block_len = seq.block_length
            
            # Extract logits for this sequence
            seq_logits = logits[start_idx:start_idx + block_len]  # (block_len, vocab_size)
            start_idx += block_len
            
            # Apply temperature and softmax
            temperatures = seq.temperature
            logits_scaled = seq_logits / temperatures
            probs = F.softmax(logits_scaled, dim=-1)  # (block_len, vocab_size)
            
            # Calculate entropies
            seq_entropies = -(probs.clamp_min(EPS) * (probs.clamp_min(EPS)).log()).sum(dim=-1)
            
            # Sample tokens
            seq_x0 = torch.multinomial(probs, num_samples=1).view(-1).to(torch.int64)  # (block_len,)
            seq_x0_p = torch.gather(probs, -1, seq_x0.unsqueeze(-1)).squeeze(-1)  # (block_len,)
            seq_x0_logp = torch.log(seq_x0_p.clamp_min(EPS))
            
            # Current block tokens and mask indices
            current_block_tensor = torch.tensor(seq.intermediate_block_tokens, device=device, dtype=torch.int64)
            mask_index = (current_block_tensor == self.mask_token_id)
            
            # Number of tokens to transfer in this step
            num_to_transfer = seq.num_transfer_tokens_per_step[seq.current_denoising_step] \
                if seq.current_denoising_step < len(seq.num_transfer_tokens_per_step) else block_len
            
            # Initialize transfer index
            transfer_index = torch.zeros(block_len, dtype=torch.bool, device=device)
            
            # --- Apply remasking strategy ---
            strategy = seq.remasking_strategy
            
            if 'sequential' in strategy:
                # Transfer tokens sequentially from first mask position
                if mask_index.any():
                    first_mask_pos = mask_index.nonzero(as_tuple=True)[0].min().item()
                    end_pos = min(first_mask_pos + num_to_transfer, block_len)
                    transfer_index[first_mask_pos:end_pos] = True
            
            elif 'low_confidence_static' in strategy:
                # Transfer top-k by confidence
                confidence = torch.where(mask_index, seq_x0_p, torch.tensor(-np.inf, device=device))
                if num_to_transfer > 0 and mask_index.sum() > 0:
                    num_to_transfer = min(num_to_transfer, mask_index.sum().item())
                    _, top_indices = torch.topk(confidence, k=num_to_transfer)
                    transfer_index[top_indices] = True
            
            elif 'low_confidence_dynamic' in strategy:
                # Dynamic threshold-based transfer
                confidence = torch.where(mask_index, seq_x0_p, torch.tensor(-np.inf, device=device))
                dyn_threshold = seq.dynamic_threshold if hasattr(seq, 'dynamic_threshold') else 0.9
                
                # Threshold-based transfer
                transfer_index = (confidence > dyn_threshold)
                
                # Fallback to sequential if not enough tokens
                if transfer_index.sum() < num_to_transfer and mask_index.sum() > 0:
                    # Fall back to sequential strategy
                    transfer_index = torch.zeros(block_len, dtype=torch.bool, device=device)
                    if mask_index.any():
                        first_mask_pos = mask_index.nonzero(as_tuple=True)[0].min().item()
                        end_pos = min(first_mask_pos + num_to_transfer, block_len)
                        transfer_index[first_mask_pos:end_pos] = True
                
                num_to_transfer = transfer_index.sum().item()
            
            elif 'entropy_bounded' in strategy:
                # Transfer based on entropy cumsum
                eb_threshold = seq.eb_threshold if hasattr(seq, 'eb_threshold') else 2.0
                
                masked_entropies = torch.where(mask_index, seq_entropies, torch.tensor(np.inf, device=device))
                ent_sorted, order = torch.sort(masked_entropies, descending=False)
                ent_sorted_masked = torch.where(ent_sorted == np.inf, 0.0, ent_sorted)
                
                cumsum = torch.cumsum(ent_sorted_masked, dim=0)
                k = torch.searchsorted(cumsum, torch.tensor(eb_threshold, dtype=torch.float32, device=device), right=False).item()
                k = max(1, k)  # At least 1 token
                
                # Get indices of mask tokens that should be transferred
                mask_positions = mask_index.nonzero(as_tuple=True)[0]
                if len(mask_positions) > 0:
                    selected_indices = mask_positions[order[:k]]
                    transfer_index[selected_indices] = True
                
                num_to_transfer = k
            
            elif strategy == 'random':
                # Random selection
                if mask_index.sum() > 0:
                    mask_positions = mask_index.nonzero(as_tuple=True)[0]
                    num_to_select = min(num_to_transfer, len(mask_positions))
                    selected_indices = mask_positions[torch.randperm(len(mask_positions), device=device)[:num_to_select]]
                    transfer_index[selected_indices] = True
                    num_to_transfer = transfer_index.sum().item()
            
            # --- Update sequence state ---
            # Only transfer mask tokens that passed strategy filter
            final_transfer_index = transfer_index & mask_index
            
            # Update tokens
            new_block_list = current_block_tensor.tolist()
            accepted_tokens = seq_x0[final_transfer_index].tolist()
            accepted_tokens_entropy = seq_entropies[final_transfer_index].tolist()
            accepted_tokens_logprobs = seq_x0_logp[final_transfer_index].tolist()
            original_indices = final_transfer_index.nonzero(as_tuple=True)[0].tolist()
            
            # Ensure trajectory, logprobs, entropies are initialized
            if seq.block_trajectory is None or len(seq.block_trajectory) != block_len:
                seq.block_trajectory = [0] * block_len
            if seq.block_logprobs is None or len(seq.block_logprobs) != block_len:
                seq.block_logprobs = [0.0] * block_len
            if seq.block_entropies is None or len(seq.block_entropies) != block_len:
                seq.block_entropies = [0.0] * block_len
            
            # Update tokens and trajectory
            first_time_global = seq.global_denoising_step + 1
            for idx, token, entropy, logprob in zip(original_indices, accepted_tokens, accepted_tokens_entropy, accepted_tokens_logprobs):
                new_block_list[idx] = token
                # Track trajectory (only mark once)
                if seq.block_trajectory[idx] == 0:
                    seq.block_trajectory[idx] = first_time_global
                # Store logprobs and entropies (update every time)
                seq.block_logprobs[idx] = logprob
                seq.block_entropies[idx] = entropy
            
            # Update sequence
            seq.intermediate_block_tokens = new_block_list
            seq.current_denoising_step += 1
            seq.global_denoising_step += 1
            seq.num_to_transfer = num_to_transfer
            
            # Check if block is fully denoised
            is_fully_denoised = (self.mask_token_id not in seq.intermediate_block_tokens) or \
                                (seq.current_denoising_step >= seq.denoising_steps)
            
            if is_fully_denoised:
                seq.status = SequenceStatus.SAVING
            else:
                seq.status = SequenceStatus.DENOISING
        
        # Process SAVING sequences
        for seq in seqs:
            if seq.status == SequenceStatus.SAVING:
                # Commit block and prepare for next
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
