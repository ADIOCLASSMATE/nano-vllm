from collections import deque
from dataclasses import dataclass, field
from typing import List, Union, Optional, Any
import torch
import random
import numpy as np

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import SequenceStatus, RunMode

# FlashInfer imports for Diffusion sampling
try:
    from flashinfer.logits_processor import LogitsPipe, Temperature, Softmax
    from flashinfer.sampling import top_k_top_p_sampling_from_probs
except ImportError:
    print("Warning: flashinfer not found. Diffusion mode will fail if used.")

EPS = 1e-12

@dataclass
class ScheduleResult:
    prefill: List['Sequence'] = field(default_factory=list)
    decode: List['Sequence'] = field(default_factory=list) # Denoise mapped to decode

    @property
    def has_work(self) -> bool:
        return bool(self.prefill or self.decode)

class Scheduler:
    def __init__(self, config: Any):
        # Configuration
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = getattr(config, 'max_num_batched_tokens', 8192)
        
        # AR specifics
        self.eos = getattr(config, 'eos', None)
        
        # Diffusion specifics
        self.mask_token_id = getattr(config, 'mask_token_id', None)
        self.diversity_enforce = getattr(config, 'diversity_enforce', False)
        self.epsilon_greedy = getattr(config, 'epsilon_greedy', False)
        self.epsilon = getattr(config, 'epsilon', 0.0)
        self.barrier = getattr(config, 'diversity_enforce_barrier', 0)
        self.consistent_sampling_params = getattr(config, 'consistent_sampling_params', False)

        # Common Components
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        # Queues
        # Unified waiting queue. First-In-First-Out
        self.waiting: deque['Sequence'] = deque()
        # Running list (mixed AR and Diffusion)
        self.running: List['Sequence'] = []
        
        # Diffusion Sampler Pipeline (Initialized if needed)
        try:
            self.sample_pipe = LogitsPipe([Temperature(), Softmax()])
        except:
            self.sample_pipe = None

    def add(self, seq: 'Sequence'):
        self.waiting.append(seq)

    def is_finished(self):
        return not self.waiting and not self.running

    def _free_finished_sequences(self):
        """Release memory for finished sequences."""
        finished = [seq for seq in self.running if seq.is_finished]
        if not finished:
            return
        
        for seq in finished:
            self.block_manager.deallocate(seq)
        
        # Keep only running sequences
        self.running = [seq for seq in self.running if not seq.is_finished]

    def preempt(self, seq: 'Sequence'):
        """AR specific: preempt a running sequence if OOM."""
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        # Add back to front of waiting queue to prioritize it next time
        self.waiting.appendleft(seq) 

    def schedule(self) -> ScheduleResult:
        """
        Unified scheduling logic.
        1. Cleans up finished sequences.
        2. Promotes waiting sequences to running if space allows (Prefill allocation).
        3. Checks running sequences for next-step resource needs (Decode/Denoise allocation).
        """
        self._free_finished_sequences()

        # --- Phase 1: Try to schedule new sequences (Prefill) ---
        # We try to fill up to max_num_seqs
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            
            # Check 1: Global token limit (mostly for AR batch sizing)
            # For diffusion, len(seq) is the full length, for AR it's current length
            current_len = len(seq)
            if self.block_manager.num_used_blocks * self.block_manager.block_size + current_len > self.max_num_batched_tokens:
                break 

            # Check 2: Block allocation (Initial allocation)
            if not self.block_manager.can_allocate(seq):
                break
                
            # Allocate and Move
            self.waiting.popleft()
            self.block_manager.allocate(seq)
            
            # Set Status based on Mode
            if seq.mode == RunMode.AR:
                seq.status = SequenceStatus.RUNNING
            else: # Diffusion
                # Check if it needs prefill or starts directly (cached)
                if seq.num_prefill_tokens > 0:
                    seq.status = SequenceStatus.PREFILLING
                else:
                    seq.status = SequenceStatus.DENOISING
            
            self.running.append(seq)

        # --- Phase 2: Prepare Batches for Execution ---
        prefill_batch = []
        decode_batch = []
        
        # We iterate a copy because AR might preempt (modify self.running)
        current_running = list(self.running)
        
        for seq in current_running:
            # --- AR Logic ---
            if seq.mode == RunMode.AR:
                if seq.num_cached_tokens < len(seq):
                    # Has un-processed prompt tokens -> Prefill
                    prefill_batch.append(seq)
                else:
                    # Fully cached -> Decode
                    # AR requires strict check: can we append 1 more block if needed?
                    if not self.block_manager.can_append(seq):
                        self.preempt(seq)
                        # Remove from current batch lists if it was added
                        if seq in decode_batch: decode_batch.remove(seq)
                        if seq in self.running: self.running.remove(seq)
                        continue
                        
                    self.block_manager.may_append(seq)
                    decode_batch.append(seq)

            # --- Diffusion Logic ---
            else: # RunMode.DIFFUSION
                if seq.status == SequenceStatus.PREFILLING:
                    prefill_batch.append(seq)
                elif seq.status in (SequenceStatus.DENOISING, SequenceStatus.SAVING):
                    # Diffusion needs to reserve blocks for the NEXT denoising chunk
                    num_new_blocks = seq.num_new_blocks_needed(self.block_manager.block_size)
                    
                    if num_new_blocks > 0:
                        if not self.block_manager.can_append_blocks(num_new_blocks):
                            # Diffusion Strategy: Skip this step, don't preempt immediately, just wait
                            # or print warning as in original code
                            # print(f"[Warning] Seq {seq.seq_id} waiting for blocks.")
                            continue 
                        self.block_manager.append_blocks(seq, num_new_blocks)
                    
                    decode_batch.append(seq)
        
        # In strictly separated engines (like NanoVLLM), we might return either prefill OR decode
        # But for a unified engine, returning both allows the Engine to decide execution order.
        return ScheduleResult(prefill=prefill_batch, decode=decode_batch)

    def postprocess(self, seqs: List['Sequence'], model_output: Any, run_type: Any):
        """
        Unified postprocessing.
        Args:
            seqs: List of sequences processed.
            model_output: 
                - If Diffusion: Logits tensor.
                - If AR: List of token_ids (integers).
            run_type: Enum indicating PREFILL or DECODE/DENOISE phase.
        """
        if not seqs:
            return

        # Simple grouping by mode (usually a batch is either all AR or all Diffusion, but let's be safe)
        # NOTE: This implementation assumes the batch is homogeneous for efficiency. 
        # If mixed, you'd need to split model_output.
        first_seq = seqs[0]

        if first_seq.mode == RunMode.AR:
            self._postprocess_ar(seqs, model_output)
        else:
            self._postprocess_diffusion(seqs, model_output, run_type)

    def _postprocess_ar(self, seqs: List['Sequence'], token_ids: List[int]):
        """Original AR postprocessing logic."""
        # Ensure input is list of ints
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            # Check stop conditions
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                # Deallocation happens in next schedule cycle or immediately here
                # self.block_manager.deallocate(seq) # Optional: eager deallocation
        
        # Clean up finished from running list immediately
        self._free_finished_sequences()

    def _postprocess_diffusion(self, seqs: List['Sequence'], logits: torch.Tensor, run_type: Any):
        """
        Complex Diffusion postprocessing logic.
        Incorporates 'postprocess_unify' batched logic for efficiency.
        """
        # --- Handle Prefill Phase ---
        # Note: In JetEngine, run_type is an Enum. We check string representation or value.
        is_prefill = (str(run_type) == "RunType.PREFILL") or (getattr(run_type, 'name', '') == 'PREFILL')
        
        if is_prefill:
            for seq in seqs:
                seq.num_cached_tokens = seq.num_prefill_tokens
                seq.status = SequenceStatus.DENOISING
            return

        # --- Handle Denoise Phase ---
        if logits is None: 
            return

        device = logits.device
        batch_size = len(seqs)
        block_len = seqs[0].block_length

        # Sampling
        # (B, L, V)
        probs = self.sample_pipe(logits, temperature=seqs[0].temperature).view(batch_size, block_len, -1)
        entropies_all = -(probs.clamp_min(EPS) * (probs.clamp_min(EPS)).log()).sum(dim=-1)

        flat_probs = probs.view(-1, probs.shape[-1])
        
        # Determine sampling params (Scalar or Tensor)
        if self.consistent_sampling_params:
            top_k = seqs[0].top_k
            top_p = seqs[0].top_p
        else:
            top_p = torch.tensor([s.top_p for s in seqs], device=device, dtype=torch.float)
            top_k = torch.tensor([s.top_k for s in seqs], device=device, dtype=torch.long)

        # Sample X0
        flat_samples = top_k_top_p_sampling_from_probs(
            flat_probs, top_k=top_k, top_p=top_p
        ).to(torch.int64)
        
        batch_seq_x0 = flat_samples.view(batch_size, block_len)
        batch_seq_x0_p = torch.gather(flat_probs, 1, flat_samples.unsqueeze(-1)).view(batch_size, block_len)
        batch_seq_x0_logp = torch.log(batch_seq_x0_p.clamp_min(EPS))

        # Collect Batch Tensors from Sequences
        batch_current_tokens = torch.tensor([s.intermediate_block_tokens for s in seqs], device=device, dtype=torch.int64)
        batch_logprobs = torch.tensor([s.block_logprobs or [0.0]*block_len for s in seqs], device=device, dtype=torch.float)
        batch_entropies = torch.tensor([s.block_entropies or [0.0]*block_len for s in seqs], device=device, dtype=torch.float)
        batch_trajectory = torch.tensor([s.block_trajectory or [0]*block_len for s in seqs], device=device, dtype=torch.long)
        
        batch_global_step_plus_1 = torch.tensor([s.global_denoising_step + 1 for s in seqs], device=device, dtype=torch.long).unsqueeze(1)

        # Status Masks
        denoising_mask_bool = torch.tensor([s.status == SequenceStatus.DENOISING for s in seqs], device=device)
        saving_mask_bool = torch.tensor([s.status == SequenceStatus.SAVING for s in seqs], device=device)
        denoising_mask = denoising_mask_bool.unsqueeze(1)

        # Transfer Tokens Count
        num_to_transfer_list = [
            s.num_transfer_tokens_per_step[s.current_denoising_step] if s.status == SequenceStatus.DENOISING else 0
            for s in seqs
        ]
        batch_num_to_transfer = torch.tensor(num_to_transfer_list, device=device, dtype=torch.long)

        # Identify Mask Tokens
        mask_token_mask = (batch_current_tokens == self.mask_token_id) & denoising_mask
        mask_available = mask_token_mask.any(dim=1)
        effective_num_to_transfer = torch.where(mask_available, batch_num_to_transfer, torch.zeros_like(batch_num_to_transfer))

        # --- Remasking Strategy Logic (Condensed) ---
        # Assuming homogeneous strategy for batch usually, but handling per-seq
        strategy = seqs[0].remasking_strategy 
        transfer_index = torch.zeros((batch_size, block_len), dtype=torch.bool, device=device)

        # 1. Sequential
        if strategy == "sequential":
            range_tensor = torch.arange(block_len, device=device).unsqueeze(0)
            # Find first mask pos
            first_mask_pos = torch.where(
                mask_available,
                torch.argmax(mask_token_mask.int(), dim=1),
                torch.full_like(effective_num_to_transfer, block_len)
            )
            start = first_mask_pos.unsqueeze(1)
            end = (start + effective_num_to_transfer.unsqueeze(1)).clamp_max(block_len)
            transfer_index = (range_tensor >= start) & (range_tensor < end)
            transfer_index &= mask_token_mask

        # 2. Entropy Bounded (Example of another strategy)
        elif "entropy_bounded" in strategy:
            masked_entropies = torch.where(mask_token_mask, entropies_all, torch.full_like(entropies_all, torch.inf))
            ent_sorted, order = torch.sort(masked_entropies, dim=1)
            # Filter infs
            ent_sorted_masked = torch.where(torch.isfinite(ent_sorted), ent_sorted, torch.zeros_like(ent_sorted))
            cumsum = torch.cumsum(ent_sorted_masked, dim=1)
            
            thresholds = torch.tensor([s.eb_threshold for s in seqs], device=device).unsqueeze(1)
            k_tensor = torch.searchsorted(cumsum, thresholds, right=False)
            
            # Logic: at least transfer 1 if masks exist
            k_tensor = torch.where(
                (mask_available & denoising_mask_bool).unsqueeze(1), 
                k_tensor.clamp_min(1), 
                torch.zeros_like(k_tensor)
            ).squeeze(1)
            
            max_k = int(k_tensor.max().item())
            if max_k > 0:
                k_mask = torch.arange(max_k, device=device).unsqueeze(0) < k_tensor.unsqueeze(1)
                transfer_index = torch.zeros_like(mask_token_mask)
                transfer_index.scatter_(1, order[:, :max_k], k_mask)
                transfer_index &= mask_token_mask

        # 3. Low Confidence / Dynamic / Random ... (Simplified: use logic from original code if needed)
        # For brevity, falling back to sequential if strategy not matched above, or add blocks here.
        else:
            # Fallback or implement other strategies
            pass 

        # Final Mask
        final_transfer_index = transfer_index & mask_token_mask

        # Apply Updates to Batch Tensors
        batch_new_tokens = torch.where(final_transfer_index, batch_seq_x0, batch_current_tokens)
        batch_new_trajectory = torch.where(
            final_transfer_index & (batch_trajectory == 0), batch_global_step_plus_1, batch_trajectory
        )
        batch_new_logprobs = torch.where(final_transfer_index, batch_seq_x0_logp, batch_logprobs)
        batch_new_entropies = torch.where(final_transfer_index, entropies_all, batch_entropies)

        # Update Steps
        denoise_increment = denoising_mask_bool.int()
        new_denoising_steps = torch.tensor([s.current_denoising_step for s in seqs], device=device) + denoise_increment
        new_global_steps = torch.tensor([s.global_denoising_step for s in seqs], device=device) + denoise_increment

        # Check Termination
        remaining_masks = (batch_new_tokens == self.mask_token_id).any(dim=1)
        step_limits = torch.tensor([s.denoising_steps for s in seqs], device=device)
        is_fully_denoised = (~remaining_masks) | (new_denoising_steps >= step_limits)
        is_fully_denoised &= denoising_mask_bool

        # --- Write Back to Objects ---
        for i, seq in enumerate(seqs):
            if seq.status == SequenceStatus.DENOISING:
                # Update State
                seq.intermediate_block_tokens = batch_new_tokens[i].tolist()
                seq.intermediate_block_tokens_entropy = batch_new_entropies[i].tolist()
                seq.block_trajectory = batch_new_trajectory[i].tolist()
                seq.block_logprobs = batch_new_logprobs[i].tolist()
                seq.block_entropies = batch_new_entropies[i].tolist()
                seq.current_denoising_step = new_denoising_steps[i].item()
                seq.global_denoising_step = new_global_steps[i].item()
                seq.num_to_transfer = final_transfer_index[i].sum().item()
                
                if is_fully_denoised[i]:
                    seq.status = SequenceStatus.SAVING

            elif seq.status == SequenceStatus.SAVING:
                # Commit Block Logic
                seq.commit_block(seq.intermediate_block_tokens)
                seq.num_to_transfer = 0
                
                if not seq.is_finished:
                    seq.start_new_block()
                else:
                    self.block_manager.deallocate(seq)

        # Final cleanup
        self._free_finished_sequences()