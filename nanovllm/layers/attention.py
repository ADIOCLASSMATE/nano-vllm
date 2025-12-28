import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context, RunType

try:
    from nanovllm.kernels.triton.attention import sparse_attn_varlen
    SPARSE_ATTN_AVAILABLE = True
except ImportError:
    SPARSE_ATTN_AVAILABLE = False


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o

class BlockAttention(Attention):
    """Block-aware attention for SDAR models using sparse attention patterns.
    
    This attention mechanism implements the block-based sparse attention pattern
    used by SDAR models, with sparse attention during prefill and local attention
    during decode phases.
    
    Expected input shapes:
        q: (seq_len, num_heads*head_dim) - flattened, will be reshaped internally
        k: (seq_len, num_kv_heads*head_dim) - flattened, will be reshaped internally
        v: (seq_len, num_kv_heads*head_dim) - flattened, will be reshaped internally
    """
    
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__(num_heads, head_dim, scale, num_kv_heads)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Forward pass for block-aware attention.
        
        Expected input shapes (flattened):
            q: (seq_len, num_heads*head_dim)
            k: (seq_len, num_kv_heads*head_dim)
            v: (seq_len, num_kv_heads*head_dim)
        
        Returns:
            o: (seq_len, num_heads*head_dim)
        """
        # Reshape inputs to separate head dimension
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # Store k, v to cache only during PREFILL phase (matching jetengine)
        should_store_whole = (context.run_type == RunType.PREFILL)
        if should_store_whole and k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        # Prefill phase: use sparse attention with block pattern
        if context.run_type == RunType.PREFILL:
            if SPARSE_ATTN_AVAILABLE:
                # Use sparse attention with staircase block pattern
                o = sparse_attn_varlen(q, k, v,
                                    cu_seqlens_q=context.cu_seqlens_q,
                                    cu_seqlens_k=context.cu_seqlens_k,
                                    staircase_size=context.block_length)
            else:
                # Fallback: use standard full attention
                if context.block_tables is not None:
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q,
                                           cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k,
                                           cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True,
                                           block_table=context.block_tables)
        # Decode/Denoise phase: use block-local attention with cached keys/values
        else:
            # For BlockAttention, always use block-local attention in DENOISE phase
            # Reshape q, k, v to block format: (seq_len, num_heads, head_dim) -> (num_blocks, block_len, num_heads, head_dim)
            q = q.view(-1, context.block_length, self.num_heads, self.head_dim)
            k = k.view(-1, context.block_length, self.num_kv_heads, self.head_dim)
            v = v.view(-1, context.block_length, self.num_kv_heads, self.head_dim)
            o = flash_attn_with_kvcache(q, k_cache=k_cache, v_cache=v_cache, k=k, v=v,
                                        cache_seqlens=context.context_lens,
                                        block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=False)
        
        # Reshape output to flattened format for compatibility with o_proj
        o = o.view(-1, self.num_heads * self.head_dim)
        return o


class LladaBlockAttention(Attention):
    """Llada variant of block-aware attention.
    
    Unlike BlockAttention which uses sparse_attn_varlen during prefill,
    LladaBlockAttention uses standard flash_attn_varlen_func (full attention).
    The decode phase remains the same with block-local attention.
    
    Expected input shapes:
        q: (seq_len, num_heads*head_dim) - flattened, will be reshaped internally
        k: (seq_len, num_kv_heads*head_dim) - flattened, will be reshaped internally
        v: (seq_len, num_kv_heads*head_dim) - flattened, will be reshaped internally
    """
    
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__(num_heads, head_dim, scale, num_kv_heads)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Forward pass for Llada block-aware attention.
        
        Expected input shapes (flattened):
            q: (seq_len, num_heads*head_dim)
            k: (seq_len, num_kv_heads*head_dim)
            v: (seq_len, num_kv_heads*head_dim)
        
        Returns:
            o: (seq_len, num_heads*head_dim)
        """
        # Reshape inputs to separate head dimension
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # Store k, v to cache only during PREFILL phase (matching jetengine)
        should_store_whole = (context.run_type == RunType.PREFILL)
        if should_store_whole and k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        # Prefill phase: use full attention (no sparse pattern like BlockAttention)
        if context.run_type == RunType.PREFILL:
            # Use standard full attention (not sparse like SDAR's BlockAttention)
            max_seqlen_q = None
            max_seqlen_k = None
            if hasattr(context, 'cu_seqlens_q') and hasattr(context, 'cu_seqlens_k'):
                # Calculate max sequence length from cu_seqlens if available
                if context.cu_seqlens_q is not None and context.cu_seqlens_k is not None:
                    max_seqlen_q = (context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]).max().item()
                    max_seqlen_k = (context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]).max().item()
            
            o = flash_attn_varlen_func(q, k, v,
                                    max_seqlen_q=max_seqlen_q,
                                    cu_seqlens_q=context.cu_seqlens_q,
                                    max_seqlen_k=max_seqlen_k,
                                    cu_seqlens_k=context.cu_seqlens_k,
                                    softmax_scale=self.scale)
        # Decode/Denoise phase: use block-local attention with cached keys/values (same as BlockAttention)
        else:
            # For LladaBlockAttention, always use block-local attention in DENOISE phase
            # Reshape q, k, v to block format: (seq_len, num_heads, head_dim) -> (num_blocks, block_len, num_heads, head_dim)
            q = q.view(-1, context.block_length, self.num_heads, self.head_dim)
            k = k.view(-1, context.block_length, self.num_kv_heads, self.head_dim)
            v = v.view(-1, context.block_length, self.num_kv_heads, self.head_dim)
            o = flash_attn_with_kvcache(q, k_cache=k_cache, v_cache=v_cache, k=k, v=v,
                                        cache_seqlens=context.context_lens,
                                        block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=False)
        
        # Reshape output to flattened format for compatibility with o_proj
        o = o.view(-1, self.num_heads * self.head_dim)
        return o