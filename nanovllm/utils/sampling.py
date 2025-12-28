"""Sampling utilities for nanovllm, compatible with jetengine's top_k_top_p sampling."""

import torch


def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_k: int | torch.Tensor = 0,
    top_p: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Sample from probabilities with top-k and top-p (nucleus) filtering.
    
    Compatible with flashinfer's top_k_top_p_sampling_from_probs API.
    
    Args:
        probs: (batch_size, vocab_size) probability distribution
        top_k: int or tensor of shape (batch_size,). If 0, no top-k filtering.
        top_p: float or tensor of shape (batch_size,). Nucleus sampling threshold.
    
    Returns:
        sampled_token_ids: (batch_size,) tensor of sampled token indices
    """
    batch_size, vocab_size = probs.shape
    device = probs.device
    dtype = probs.dtype
    
    # Handle scalar vs tensor inputs
    if isinstance(top_k, int):
        top_k = torch.tensor([top_k] * batch_size, device=device, dtype=torch.long)
    if isinstance(top_p, (float, int)):
        top_p = torch.tensor([float(top_p)] * batch_size, device=device, dtype=dtype)
    
    # Ensure top_k and top_p are on the same device
    top_k = top_k.to(device)
    top_p = top_p.to(device)
    
    # Apply top-k filtering
    if top_k.max() > 0:
        # Get top-k indices for each sample
        top_k_values, top_k_indices = torch.topk(probs, k=min(top_k.max().item(), vocab_size), dim=-1)
        
        # Create mask for each sample's top-k
        top_k_mask = torch.zeros_like(probs, dtype=torch.bool)
        for i in range(batch_size):
            k = min(top_k[i].item(), vocab_size)
            top_k_mask[i, top_k_indices[i, :k]] = True
        
        # Zero out probabilities outside top-k
        probs = probs.masked_fill(~top_k_mask, 0.0)
    
    # Apply top-p (nucleus) filtering
    if top_p.max() < 1.0:
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # Compute cumulative probabilities
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Create mask: keep tokens until cumulative prob exceeds top_p
        # For each sample, find where cumsum exceeds top_p
        top_p_expanded = top_p.unsqueeze(1)  # (batch_size, 1)
        mask = cumsum_probs > top_p_expanded
        
        # Zero out probabilities beyond the top-p threshold
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        
        # Renormalize
        sorted_probs_sum = sorted_probs.sum(dim=-1, keepdim=True)
        sorted_probs = sorted_probs / sorted_probs_sum.clamp_min(1e-10)
        
        # Scatter back to original indices
        probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)
    
    # Renormalize to ensure valid probability distribution
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-10)
    
    # Sample using multinomial
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return sampled

