from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    DENOISING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams(), mask_token_id: int = -1):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.top_p = sampling_params.top_p
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        
        # Block Diffusion parameters (for LLADA/SDAR)
        self.mask_token_id = mask_token_id
        self.block_length = sampling_params.block_length
        self.denoising_steps = sampling_params.denoising_steps
        self.remasking_strategy = sampling_params.remasking_strategy
        self.dynamic_threshold = sampling_params.dynamic_threshold
        self.eb_threshold = sampling_params.eb_threshold
        self.top_k = sampling_params.topk
        self.top_p_param = sampling_params.topp
        self.stop_words = sampling_params.stop_words if sampling_params.stop_words is not None else []
        
        # Initialize denoise-specific attributes
        if mask_token_id > 0:
            self._init_denoise_params()
        else:
            self.intermediate_block_tokens = None
            self.current_denoising_step = 0
    
    def _init_denoise_params(self):
        """Initialize denoise-specific parameters for LLADA/SDAR models."""
        # Split prompt into prefill and first denoise block
        prompt_len = len(self.token_ids)
        self.num_prefill_tokens = (prompt_len // self.block_length) * self.block_length
        prefill_part = self.token_ids[:self.num_prefill_tokens]
        
        first_denoise_part = self.token_ids[self.num_prefill_tokens:]
        self.generation_start_index = len(first_denoise_part)
        
        # Keep only prefill part in token_ids, denoise part goes to intermediate_block
        self.token_ids = prefill_part
        self.num_tokens = len(self.token_ids)
        
        # Intermediate block = first denoise part + mask tokens
        self.intermediate_block_tokens = first_denoise_part + [self.mask_token_id] * (self.block_length - len(first_denoise_part))
        self.intermediate_block_tokens_entropy = [0.0] * self.block_length
        self.block_trajectory = [0] * self.block_length
        self.block_logprobs = [0.0] * self.block_length
        self.block_entropies = [0.0] * self.block_length
        
        self.current_denoising_step = 0
        self.num_to_transfer = 0
        
        # Determine initial status
        if self.num_prefill_tokens > 0:
            self.status = SequenceStatus.WAITING
        else:
            self.status = SequenceStatus.DENOISING

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
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

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

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
    
    def start_new_block(self):
        """Start a new denoise block with all mask tokens."""
        self.current_denoising_step = 0
        self.intermediate_block_tokens = [self.mask_token_id] * self.block_length
        self.intermediate_block_tokens_entropy = [0.0] * self.block_length
        self.block_trajectory = [0] * self.block_length
        self.block_logprobs = [0.0] * self.block_length
        self.block_entropies = [0.0] * self.block_length
        self.status = SequenceStatus.DENOISING
    
    def commit_block(self, block_tokens: list[int]):
        """Commit confirmed denoise block tokens to token_ids."""
        for token_id in block_tokens:
            if (not self.ignore_eos) and (token_id in self.stop_words):
                self.status = SequenceStatus.FINISHED
                self.token_ids.append(token_id)
                break
            if self.num_completion_tokens >= self.max_tokens:
                self.status = SequenceStatus.FINISHED
                break
            self.token_ids.append(token_id)
        
        if self.num_completion_tokens >= self.max_tokens:
            self.status = SequenceStatus.FINISHED

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
