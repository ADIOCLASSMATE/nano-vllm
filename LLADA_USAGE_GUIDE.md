# LLaDA Block Denoising Integration - Usage Guide

## Quick Start

### For LLaDA Models

```python
from nanovllm import LLM, SamplingParams

# Initialize with mask_token_id to enable block denoising
llm = LLM(
    "path/to/llada-model",
    mask_token_id=126336,  # LLaDA specific mask token ID
    block_length=4,        # Block size for denoising
    tensor_parallel_size=1,
)

# Configure sampling with denoising parameters
sampling_params = SamplingParams(
    temperature=1.0,
    max_tokens=4096,
    block_length=4,
    denoising_steps=4,
    remasking_strategy="low_confidence_dynamic",
    dynamic_threshold=0.9,
)

# Generate as usual
prompts = ["Your prompt here"]
outputs = llm.generate(prompts, sampling_params)
```

### For SDAR Models

```python
llm = LLM(
    "path/to/sdar-model",
    mask_token_id=151669,  # SDAR specific mask token ID
    block_length=4,
)

sampling_params = SamplingParams(
    temperature=1.0,
    max_tokens=4096,
    block_length=4,
    denoising_steps=4,
    remasking_strategy="sequential",
)

outputs = llm.generate(prompts, sampling_params)
```

### For Qwen3 (No Block Denoising)

```python
# Leave mask_token_id at default (-1)
llm = LLM("path/to/qwen3-model")

sampling_params = SamplingParams(
    temperature=1.0,
    max_tokens=4096,
)

outputs = llm.generate(prompts, sampling_params)
```

## How It Works

### Execution Flow for Block Denoising

```
1. PREFILL Phase
   ├─ Input: Full prompt tokens
   ├─ Attention: Full (prompt can attend to all positions)
   └─ Output: Hidden states for last prefill batch

2. DENOISE Phase (repeats for each block)
   ├─ Input: [prompt_part] + [mask_tokens]
   │   (intermediate_block_tokens)
   ├─ Attention: Full (entire sequence can attend)
   ├─ Output: Logits for mask positions
   ├─ Sampling: Select best tokens for masked positions
   └─ Update: Replace mask tokens with sampled tokens
   
3. Block Complete
   └─ Commit block tokens to main sequence
      Start new block with fresh mask tokens
      (or finish if max_tokens reached)
```

### Execution Flow for Standard Autoregressive

```
1. PREFILL Phase
   └─ Input: Full prompt, Output: Hidden states

2. DECODE Phase (one token at a time)
   ├─ Input: Last token
   ├─ Attention: Local (only on previous context)
   ├─ Output: Logits for next position
   ├─ Sampling: Select next token
   └─ Update: Append token to sequence
```

## Parameters

### SamplingParams for Block Denoising

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_length` | 4 | Number of tokens to generate per block |
| `denoising_steps` | 4 | Number of denoising iterations per block |
| `remasking_strategy` | `'low_confidence_static'` | How to select tokens to update: <br/>`'sequential'`: Left-to-right <br/>`'low_confidence_static'`: Update lowest confidence tokens <br/>`'low_confidence_dynamic'`: Threshold-based low confidence <br/>`'entropy_bounded'`: Entropy-based selection |
| `dynamic_threshold` | 0.9 | Confidence threshold for dynamic remasking |
| `eb_threshold` | 0.35 | Entropy threshold for entropy-bounded remasking |

### Config Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mask_token_id` | int | -1 | Special token ID for masking (enables block denoising if > 0) |
| `block_length` | int | 4 | Size of generation blocks |

## Troubleshooting

### "intermediate_block_tokens is None"
- Ensure `mask_token_id > 0` is set in LLM initialization
- Check that the model config has the correct mask_token_id value

### "Block generation seems slow"
- Adjust `denoising_steps` - fewer steps = faster but potentially lower quality
- Reduce `block_length` for smaller blocks

### "Tokens don't match expected output"
- Verify `remasking_strategy` parameter is correct for your use case
- Check `temperature` - lower values = more confident predictions
- Ensure `max_tokens` is not causing premature termination

## Advanced Usage

### Custom Remasking Strategy

The remasking strategy determines which tokens in the block get updated in each iteration:

1. **Sequential**: Updates tokens from left to right
   - Fast, deterministic, simulates standard autoregressive decoding
   
2. **Low Confidence Static**: Updates tokens with lowest confidence scores
   - Iteratively fixes uncertain predictions
   - More adaptive than sequential

3. **Low Confidence Dynamic**: Uses dynamic threshold
   - Threshold adjusts based on distribution
   - Best for variable difficulty content

4. **Entropy Bounded**: Uses entropy as confidence metric
   - Information-theoretic approach
   - Good for diversity-seeking generation

### Monitoring Block Progress

Each sequence tracks:
- `current_denoising_step`: Which iteration within block (0 to block_length-1)
- `intermediate_block_tokens`: Current state of mask tokens
- `block_trajectory`: History of token updates

## Performance Notes

- **Block denoising is most beneficial** when block_length ≥ 4
- **Smaller blocks (1-2)** approach standard autoregressive speed
- **Larger blocks (8+)** may have higher quality but slower speed
- **KV cache** is efficiently managed across denoise iterations

