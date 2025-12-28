# LLaDA Block Denoising Integration to nanoVLLM

## Overview

Successfully integrated block-based denoising support from jetengine to nanovllm, enabling proper LLaDA and SDAR model inference with mask token handling and iterative block generation.

## Key Changes

### 1. **nanovllm/utils/context.py** - Context Management
- ✅ Added `RunType` enum (`PREFILL`, `DENOISE`)
- ✅ Extended `Context` dataclass with:
  - `run_type`: Tracks execution phase (PREFILL vs DENOISE)
  - `block_length`: Block size for denoising
- ✅ Updated `set_context()` signature to support `run_type` parameter with backward compatibility

**Why:** To distinguish between prompt prefilling and block denoising phases for proper attention computation.

---

### 2. **nanovllm/engine/sequence.py** - Sequence State Management
- ✅ Added `SequenceStatus.DENOISING` to status enum
- ✅ Extended `Sequence.__init__()` to accept `mask_token_id` parameter
- ✅ Added `_init_denoise_params()` method:
  - Splits prompt into prefill and first denoise block
  - Creates `intermediate_block_tokens` with mask tokens
  - Initializes tracking arrays: `block_trajectory`, `block_logprobs`, `block_entropies`
  - Sets initial status based on whether prefill is needed
- ✅ Added `start_new_block()` method - resets denoise block with mask tokens
- ✅ Added `commit_block()` method - commits confirmed tokens and checks stop conditions
- ✅ Added block denoising parameters from SamplingParams:
  - `block_length`, `denoising_steps`, `remasking_strategy`, etc.

**Why:** To track intermediate block tokens separately from confirmed tokens during denoising.

---

### 3. **nanovllm/engine/llm_engine.py** - Request Addition
- ✅ Modified `add_request()` to pass `mask_token_id` to Sequence:
```python
seq = Sequence(prompt, sampling_params, mask_token_id=self.scheduler.config.mask_token_id)
```
- ✅ Updated `step()` method signature to handle new return value from scheduler:
```python
seqs, is_prefill, run_type = self.scheduler.schedule()
token_ids = self.model_runner.call("run", seqs, is_prefill, run_type)
```

**Why:** To enable scheduler to return the execution phase for model runner selection.

---

### 4. **nanovllm/engine/scheduler.py** - Scheduling Logic
- ✅ Added `mask_token_id` and `denoise_ready` deque to Scheduler
- ✅ Extended `schedule()` return signature:
  - Returns: `(seqs, is_prefill, run_type)` instead of `(seqs, is_prefill)`
  - Phases: PREFILL → DENOISE → DENOISE (loop) with standard DECODE as fallback
- ✅ Added denoise batch scheduling between prefill and decode
- ✅ Extended `postprocess()` to handle:
  - DENOISE: Update `intermediate_block_tokens`, manage step progression
  - DECODE: Standard token appending
  - Move sequences to `denoise_ready` when block complete

**Why:** To separate prefill, denoise, and standard decode phases with proper state management.

---

### 5. **nanovllm/engine/model_runner.py** - Model Execution
- ✅ Modified `run()` signature to accept `run_type` parameter
- ✅ Added `prepare_denoise()` method:
  - Query: `intermediate_block_tokens` (with mask tokens)
  - Context: Confirmed tokens from previous prefill/iterations
  - Positions: Global sequence positions
  - Full attention between prompt and current block
- ✅ Updated `warmup_model()` to pass `mask_token_id` to Sequence
- ✅ Updated `set_context()` calls to include `run_type=RunType.PREFILL` and `run_type=RunType.DENOISE`

**Why:** To properly prepare inputs for denoise phase where query is the intermediate block.

---

## How It Works

### Standard Autoregressive (Qwen3, default):
```
Prompt → Prefill → Token 1 → Token 2 → ... → EOS
```

### Block Denoising (LLaDA, SDAR):
```
Prompt → Prefill → [Mask tokens] → Denoise Step 1 → Denoise Step 2 → ... → Complete Block
         ↓                           ↓                ↓
    Keep in token_ids          intermediate_block   Commit to token_ids
                               
Then repeat for next block until max_tokens reached
```

### Key Differences:
1. **Input to attention**: 
   - Prefill: Prompt tokens with full attention
   - Denoise: `intermediate_block_tokens` (prompt + block mask) with full attention between all positions

2. **Output processing**:
   - Prefill: Single prefill run
   - Denoise: Iterative runs updating mask positions until block complete

3. **Scheduling**:
   - `prefill_ready`: Sequences awaiting prompt processing
   - `denoise_ready`: Sequences awaiting block denoising (new!)
   - `running`: Standard decode sequences (fallback if no denoise blocks)

---

## Backward Compatibility

✅ All changes maintain backward compatibility:
- Default `mask_token_id = -1` disables denoising (standard autoregressive)
- `set_context()` infers `run_type` from `is_prefill` if not provided
- Scheduler handles both old `(seqs, is_prefill)` and new `(seqs, is_prefill, run_type)` returns

## Testing Checklist

- [ ] Verify LLaDA model with `mask_token_id=126336` generates correctly
- [ ] Verify SDAR model with proper mask token generates correctly
- [ ] Verify Qwen3 (no mask token) still works unchanged
- [ ] Check intermediate block tokens are correctly masked
- [ ] Verify block completion triggers token commitment
- [ ] Verify KV cache is properly managed across denoise steps

## References

- **Original jetengine implementation**: `jetengine/engine/sequence.py`, `jetengine/engine/scheduler.py`
- **LLaDA model**: `nanovllm/models/llada.py` (forward logic already fixed)
- **Attention layers**: `nanovllm/layers/attention.py` (LladaBlockAttention, BlockAttention)

