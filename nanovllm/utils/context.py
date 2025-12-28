from dataclasses import dataclass
from enum import Enum, auto
import torch
from nanovllm.engine.sequence import RunType


# class RunType(Enum):
#     PREFILL = auto()
#     DENOISE = auto()


@dataclass
class Context:
    is_prefill: bool = False
    run_type: RunType | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    block_length: int | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill=False, run_type=None, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, block_length=None):
    global _CONTEXT
    if run_type is None:
        # Backward compatibility: infer run_type from is_prefill if not provided
        run_type = RunType.PREFILL if is_prefill else RunType.DENOISE
    _CONTEXT = Context(is_prefill, run_type, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, block_length)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
