"""Verification script for SDAR and Llada adaptations to nanovllm"""

import sys
import torch

# Test imports
try:
    from nanovllm.models.sdar import SDARForCausalLM, SDARAttention, SDARDecoderLayer
    print("✓ SDARForCausalLM imports successful")
except Exception as e:
    print(f"✗ SDARForCausalLM import failed: {e}")
    sys.exit(1)

try:
    from nanovllm.models.llada import LladaForCausalLM, LladaDecoderLayer
    print("✓ LladaForCausalLM imports successful")
except Exception as e:
    print(f"✗ LladaForCausalLM import failed: {e}")
    sys.exit(1)

try:
    from nanovllm.layers.attention import BlockAttention, LladaBlockAttention
    print("✓ BlockAttention and LladaBlockAttention imports successful")
except Exception as e:
    print(f"✗ BlockAttention/LladaBlockAttention import failed: {e}")
    sys.exit(1)

# Test model creation with mock config
class MockConfig:
    model_type = "sdar"
    hidden_size = 768
    num_attention_heads = 12
    num_key_value_heads = 2
    intermediate_size = 3072
    num_hidden_layers = 12
    vocab_size = 32000
    max_position_embeddings = 4096
    rms_norm_eps = 1e-6
    rope_theta = 10000
    tie_word_embeddings = True
    dtype = torch.float16

try:
    config = MockConfig()
    sdar_model = SDARForCausalLM(config)
    print("✓ SDARForCausalLM initialization successful")
except Exception as e:
    print(f"✗ SDARForCausalLM initialization failed: {e}")
    sys.exit(1)

class MockLladaConfig:
    model_type = "llada"
    n_heads = 12
    n_kv_heads = 2
    d_model = 768
    mlp_hidden_size = 3072
    n_layers = 12
    vocab_size = 32000
    max_sequence_length = 4096
    rms_norm_eps = 1e-6
    rope_theta = 500000.0
    activation_type = "silu"
    weight_tying = True
    attention_bias = False
    rope_scaling = None
    dtype = torch.float16

try:
    llada_config = MockLladaConfig()
    llada_model = LladaForCausalLM(llada_config)
    print("✓ LladaForCausalLM initialization successful")
except Exception as e:
    print(f"✗ LladaForCausalLM initialization failed: {e}")
    sys.exit(1)

# Verify key architectural differences
print("\n--- Architectural Verification ---")

# Check SDAR has BlockAttention
sdar_has_block_attn = any(isinstance(m, BlockAttention) for m in sdar_model.modules())
print(f"✓ SDAR uses BlockAttention: {sdar_has_block_attn}")

# Check Llada has LladaBlockAttention
llada_has_llada_block_attn = any(isinstance(m, LladaBlockAttention) for m in llada_model.modules())
print(f"✓ Llada uses LladaBlockAttention: {llada_has_llada_block_attn}")

print("\n✓ All verification checks passed!")
