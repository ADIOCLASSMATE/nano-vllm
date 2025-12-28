#!/usr/bin/env python3
"""
Test script to verify SDAR attention shapes and forward pass.
"""
import torch
import torch.distributed as dist
from nanovllm.models.sdar import SDARForCausalLM
from nanovllm.config import Config


def test_sdar_shapes():
    """Test SDAR model forward pass shapes."""
    # Mock config
    class MockConfig:
        def __init__(self):
            self.vocab_size = 32000
            self.hidden_size = 4096
            self.num_attention_heads = 32
            self.num_key_value_heads = 8
            self.intermediate_size = 14336
            self.num_hidden_layers = 8
            self.max_position_embeddings = 4096
            self.rms_norm_eps = 1e-6
            self.hidden_act = "silu"
            self.tie_word_embeddings = False
            self.attention_bias = False
            self.rope_theta = 1000000
            self.rope_scaling = None
            self.model_type = "sdar"
    
    config = MockConfig()
    
    # Test with single GPU (no dist)
    try:
        dist.init_process_group("gloo", rank=0, world_size=1, init_method="tcp://localhost:12355")
    except:
        print("Warning: Could not initialize dist, skipping multi-GPU test")
    
    # Create model
    model = SDARForCausalLM(config)
    model.eval()
    
    # Test PREFILL mode
    print("\n=== PREFILL Test ===")
    batch_size = 2
    seq_len = 100
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    positions = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0).expand(batch_size, -1).reshape(-1)
    
    from nanovllm.utils.context import set_context, RunType
    set_context(True, run_type=RunType.PREFILL)
    
    with torch.no_grad():
        try:
            # Forward through embedding and transformer
            hidden_states = model.model.embed_tokens(input_ids.reshape(-1))
            print(f"After embedding: {hidden_states.shape}")
            
            # Process through attention layer
            layer = model.model.blocks[0]
            print(f"Attention q_size: {layer.self_attn.q_size}")
            print(f"Attention kv_size: {layer.self_attn.kv_size}")
            print(f"Attention head_dim: {layer.self_attn.head_dim}")
            
            residual = None
            hidden_states, residual = layer(positions, hidden_states, residual)
            print(f"After attention layer: {hidden_states.shape}")
            
            # Compute logits
            logits = model.compute_logits(hidden_states)
            print(f"Logits shape: {logits.shape}")
            print("✓ PREFILL test passed")
        except Exception as e:
            print(f"✗ PREFILL test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Test DENOISE mode
    print("\n=== DENOISE Test ===")
    batch_size = 2
    block_len = 4
    input_ids = torch.randint(0, config.vocab_size, (batch_size * block_len,))
    positions = torch.arange(block_len, dtype=torch.int64).unsqueeze(0).expand(batch_size, -1).reshape(-1) + 100
    
    set_context(False, run_type=RunType.DENOISE, block_length=block_len)
    
    with torch.no_grad():
        try:
            # Forward through embedding and transformer
            hidden_states = model.model.embed_tokens(input_ids)
            print(f"After embedding: {hidden_states.shape}")
            
            # Process through attention layer
            layer = model.model.blocks[0]
            residual = None
            hidden_states, residual = layer(positions, hidden_states, residual)
            print(f"After attention layer: {hidden_states.shape}")
            
            # Compute logits
            logits = model.compute_logits(hidden_states)
            print(f"Logits shape: {logits.shape}")
            print("✓ DENOISE test passed")
        except Exception as e:
            print(f"✗ DENOISE test failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_sdar_shapes()
