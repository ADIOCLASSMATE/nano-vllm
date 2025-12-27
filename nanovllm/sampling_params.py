from dataclasses import dataclass
from typing import Literal


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    # Block Diffusion Parameters (for SDAR models)
    block_length: int = 4
    denoising_steps: int = 4
    dynamic_threshold: float = 0.9
    eb_threshold: float = 0.35
    topk: int = 0
    topp: float = 1.0
    remasking_strategy: Literal['sequential', 'low_confidence_static', 'low_confidence_dynamic', 'entropy_bounded'] = 'low_confidence_static'
    stop_words: list[int] | None = None
    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
        assert 0.0 < self.top_p <= 1.0, "top_p must be in (0, 1]"
