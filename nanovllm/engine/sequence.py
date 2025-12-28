from copy import copy
from enum import Enum, auto
from itertools import count
from typing import List, Optional, Any, Union

class SequenceStatus(Enum):
    # Common / Shared
    WAITING = auto()      # Waiting to be scheduled
    FINISHED = auto()     # Completed generation
    
    # Diffusion Specific
    PREFILLING = auto()   # Currently in prefill run
    DENOISING = auto()    # Currently in denoise run
    SAVING = auto()       # Saving output
    
    # AR Specific (Mapped to RUNNING in original AR code)
    RUNNING = auto()      # Standard AR generation loop

class RunType(Enum):
    PREFILL = auto()
    DENOISE = auto()
    
class RunMode(Enum):
    AR = auto()
    DIFFUSION = auto()

class Sequence:
    default_block_size = 256
    counter = count()

    def __init__(
        self, 
        token_ids: List[int], 
        sampling_params: Any, 
        block_size: int | None = None,
        mask_token_id: int | None = None,
        mode: RunMode = RunMode.AR
    ):
        """
        Unified Sequence init.
        
        Args:
            token_ids: The initial input tokens (prompt).
            sampling_params: The parameters object (should contain attributes needed for the specific mode).
            block_size: KV cache block size.
            mask_token_id: Required for RunMode.DIFFUSION.
            mode: RunMode.AR or RunMode.DIFFUSION.
        """
        self.seq_id = next(Sequence.counter)
        self.mode = mode
        self.block_size = block_size if block_size is not None else Sequence.default_block_size
        
        # Common State
        self.num_cached_tokens = 0
        self.block_table = []
        
        # Extract common sampling params (using getattr to be safe if params objects differ)
        self.temperature = getattr(sampling_params, 'temperature', 1.0)
        self.max_tokens = getattr(sampling_params, 'max_tokens', 128)
        self.ignore_eos = getattr(sampling_params, 'ignore_eos', False)
        
        # --- AR Initialization ---
        if self.mode == RunMode.AR:
            self.status = SequenceStatus.WAITING
            self.token_ids = copy(token_ids)
            self.last_token = token_ids[-1] if token_ids else None
            self.num_tokens = len(self.token_ids)
            self.num_prompt_tokens = len(token_ids)
            
            # AR specific params
            self.top_p = getattr(sampling_params, 'top_p', 1.0)
            
            # Placeholder for Diffusion attributes to avoid AttributeErrors
            self.intermediate_block_tokens = []
            self.trajectory = []
            
        # --- Diffusion Initialization ---
        elif self.mode == RunMode.DIFFUSION:
            if mask_token_id is None:
                raise ValueError("mask_token_id is required for Diffusion mode")
            
            self.mask_token_id = mask_token_id
            self.block_length = getattr(sampling_params, 'block_length', 128)
            self.prompt_token_ids = token_ids # Keep original prompt ref
            prompt_len = len(token_ids)
            
            # Calculate Prefill vs Denoise parts
            self.num_prefill_tokens = (prompt_len // self.block_length) * self.block_length
            prefill_part = token_ids[:self.num_prefill_tokens]
            first_denoise_part = token_ids[self.num_prefill_tokens:]
            
            self.generation_start_index = len(first_denoise_part)
            
            self.token_ids = prefill_part
            self.num_tokens = len(self.token_ids)
            self.num_prompt_tokens = prompt_len
            self.num_generated_tokens = self.num_tokens - self.num_prompt_tokens
            
            # Init Intermediate State
            self.intermediate_block_tokens = first_denoise_part + [mask_token_id] * (self.block_length - len(first_denoise_part))
            self.intermediate_block_tokens_entropy = [0.0] * self.block_length
            
            # Logging / Stats
            self.trajectory = []
            self.block_trajectory = [0] * len(self.intermediate_block_tokens)
            self.logprobs = []
            self.block_logprobs = [0.0] * len(self.intermediate_block_tokens)
            self.entropies = []
            self.block_entropies = [0.0] * len(self.intermediate_block_tokens)
            
            self.current_denoising_step = 0
            self.global_denoising_step = 0
            
            # Diffusion specific params
            self.denoising_steps = getattr(sampling_params, 'denoising_steps', 1)
            stop_words = getattr(sampling_params, 'stop_words', None)
            self.stop_words = set(stop_words) if stop_words else set()
            self.top_k = getattr(sampling_params, 'topk', -1)
            self.top_p = getattr(sampling_params, 'topp', 1.0) # Notice diff spelling in diffusion params
            self.remasking_strategy = getattr(sampling_params, 'remasking_strategy', None)
            self.dynamic_threshold = getattr(sampling_params, 'dynamic_threshold', None)
            self.eb_threshold = getattr(sampling_params, 'eb_threshold', None)
            self.num_transfer_tokens_per_step = self._get_num_transfer_tokens()

            # Set initial status
            if self.num_prefill_tokens > 0:
                self.status = SequenceStatus.WAITING # Logic handles transition to PREFILLING
            else:
                self.status = SequenceStatus.DENOISING

    # ------------------------------------------------------------------
    # Common Properties
    # ------------------------------------------------------------------

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size
        
    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    # ------------------------------------------------------------------
    # AR Specific Methods
    # ------------------------------------------------------------------

    def append_token(self, token_id: int):
        """Standard autoregressive token append."""
        if self.mode != RunMode.AR:
            raise RuntimeError("append_token is mostly for AR mode. Use commit_block for Diffusion.")
            
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        
        # Simple stop check for AR (can be expanded)
        if self.num_completion_tokens >= self.max_tokens:
            self.status = SequenceStatus.FINISHED

    # ------------------------------------------------------------------
    # Diffusion Specific Methods
    # ------------------------------------------------------------------

    def _get_num_transfer_tokens(self):
        """Helper for diffusion steps calculation."""
        base = self.block_length // self.denoising_steps
        remainder = self.block_length % self.denoising_steps
        num_tokens = [base] * self.denoising_steps
        for i in range(remainder):
            num_tokens[i] += 1
        return num_tokens

    def start_new_block(self):
        """Prepare sequence for the next diffusion block."""
        if self.mode != RunMode.DIFFUSION:
            return # No-op or Error based on preference
            
        self.current_denoising_step = 0
        self.intermediate_block_tokens = [self.mask_token_id] * self.block_length
        self.intermediate_block_tokens_entropy = [0.0] * self.block_length
        self.block_trajectory = [0] * self.block_length
        self.block_logprobs = [0.0] * self.block_length
        self.block_entropies = [0.0] * self.block_length
        self.status = SequenceStatus.DENOISING

    def commit_block(self, block_tokens: list[int]):
        """Commit a finished diffusion block to the sequence."""
        if self.mode != RunMode.DIFFUSION:
             raise RuntimeError("commit_block is for Diffusion mode.")

        start = self.generation_start_index
        remaining = self.max_tokens - self.num_completion_tokens
        
        if remaining <= 0:
            self.status = SequenceStatus.FINISHED
            return []

        # Stop word logic
        stop_pos = None
        if (not self.ignore_eos) and self.stop_words:
            stop_pos = next(
                (j for j, tok in enumerate(block_tokens[start:], start)
                if tok in self.stop_words),
                None
            )

        end_by_tokens = start + remaining
        end_by_stop = (stop_pos + 1) if stop_pos is not None else len(block_tokens)
        end = min(len(block_tokens), end_by_tokens, end_by_stop)

        if end == end_by_tokens or (stop_pos is not None and end == stop_pos + 1):
            self.status = SequenceStatus.FINISHED
            
        self.generation_start_index = 0
            
        before_ntok = self.num_tokens
        self.token_ids.extend(block_tokens)
        self.num_tokens = len(self.token_ids)
        self.num_generated_tokens = self.num_tokens - self.num_prompt_tokens
        
        # Cleanup block state
        self.intermediate_block_tokens = []
        self.intermediate_block_tokens_entropy = []
        
        # Stats update
        block_len = len(block_tokens)
        if self.block_trajectory is not None:
            block_start = max(0, self.num_prompt_tokens - before_ntok)
            start_idx = min(block_start, block_len)
            if start_idx < block_len:
                self.trajectory.extend(self.block_trajectory[start_idx:block_len])
                if self.block_logprobs: self.logprobs.extend(self.block_logprobs[start_idx:block_len])
                if self.block_entropies: self.entropies.extend(self.block_entropies[start_idx:block_len])
 
            self.block_trajectory = None
            self.block_logprobs = None
            self.block_entropies = None

        if self.num_tokens >= self.num_prompt_tokens + self.max_tokens:
             self.status = SequenceStatus.FINISHED

    def get_len_for_next_step(self):
        """Crucial for Diffusion memory allocation."""
        if self.mode == RunMode.AR:
            return self.num_tokens + 1
        return self.num_tokens + self.block_length

    def num_new_blocks_needed(self, block_size: int) -> int:
        len_next = self.get_len_for_next_step()
        needed_total_blocks = (len_next + block_size - 1) // block_size
        return max(0, needed_total_blocks - len(self.block_table))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def __getstate__(self):
        # We need a robust state that covers both, but respects the AR optimization
        # if feasible.
        state = {
            'seq_id': self.seq_id,
            'status': self.status,
            'mode': self.mode,
            'num_tokens': self.num_tokens,
            'num_prompt_tokens': self.num_prompt_tokens,
            'num_cached_tokens': self.num_cached_tokens,
            'block_table': self.block_table,
            'block_size': self.block_size
        }
        
        if self.mode == RunMode.AR:
            # Optimization: only store full IDs if prefill, else last token
            if self.num_completion_tokens == 0:
                state['data'] = self.token_ids
            else:
                state['data'] = self.last_token
                state['is_last_token'] = True
                
        else: # Diffusion
            state['token_ids'] = self.token_ids
            state['intermediate_block_tokens'] = self.intermediate_block_tokens
            state['current_denoising_step'] = self.current_denoising_step
            # Add other diffusion specific fields as needed
            
        return state

    def __setstate__(self, state):
        self.seq_id = state.get('seq_id', next(Sequence.counter))
        self.status = state['status']
        self.mode = state.get('mode', RunMode.AR) # Default to AR for backward compat if loading old states
        self.num_tokens = state['num_tokens']
        self.num_prompt_tokens = state['num_prompt_tokens']
        self.num_cached_tokens = state['num_cached_tokens']
        self.block_table = state['block_table']
        self.block_size = state.get('block_size', Sequence.default_block_size)
        
        if self.mode == RunMode.AR:
            if state.get('is_last_token', False):
                self.last_token = state['data']
                # Note: self.token_ids won't be fully restored here if you rely on just last_token
                # for worker processes. If you need full history on workers, remove the optimization.
                self.token_ids = [] # Or handle reconstruction logic if needed
            else:
                self.token_ids = state['data']
                if self.token_ids:
                    self.last_token = self.token_ids[-1]
        else:
            self.token_ids = state['token_ids']
            self.intermediate_block_tokens = state.get('intermediate_block_tokens', [])
            self.current_denoising_step = state.get('current_denoising_step', 0)