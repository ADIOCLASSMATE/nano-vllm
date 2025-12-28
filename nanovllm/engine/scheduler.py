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
            run_type: RunType.PREFILL or RunType.DENOISE
        """
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

        # Decode phase: standard token-by-token decoding
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
            # No sequences available - this should not happen in normal flow
            # as step() should only be called when there's work
            return [], False, RunType.DENOISE
        
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False, RunType.DENOISE

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """Process model outputs and update sequence states.
        
        For denoise: token_ids contains block_length tokens per sequence (flattened)
        For decode: token_ids contains one token per sequence
        """
        # Check if any sequence is in denoise mode
        has_denoise = any(seq.intermediate_block_tokens is not None and seq.status == SequenceStatus.DENOISING for seq in seqs)
        
        if has_denoise:
            # DENOISE phase: token_ids is flattened (N_seqs * block_length)
            token_idx = 0
            for seq in seqs:
                if seq.intermediate_block_tokens is not None and seq.status == SequenceStatus.DENOISING:
                    block_len = seq.block_length
                    # Extract this seq's tokens from the flattened list
                    seq_tokens = token_ids[token_idx:token_idx + block_len]
                    token_idx += block_len
                    
                    # Update all tokens in intermediate block (simple approach: overwrite all)
                    seq.intermediate_block_tokens = seq_tokens
                    
                    # Commit the block
                    seq.commit_block(seq.intermediate_block_tokens)
                    if seq.num_completion_tokens >= seq.max_tokens or seq.status == SequenceStatus.FINISHED:
                        self.block_manager.deallocate(seq)
                    else:
                        seq.start_new_block()
                        self.denoise_ready.append(seq)
        else:
            # Standard DECODE phase: one token per sequence
            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    if seq in self.running:
                        self.running.remove(seq)
