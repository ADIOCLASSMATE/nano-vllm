"""
Test script for update_model_param functionality.

This script:
1. Starts nanovllm LLM on GPU 1 & 2 (GPU 1 as rank 0) with Qwen2.5-1.5B
2. Generates some responses
3. Loads Qwen2.5-1.5B-Instruct on GPU 0 using transformers
4. Syncs weights from the Instruct model to nanovllm
5. Generates responses again to verify the update worked
"""

import os
import time

# Set environment variables before importing any other modules
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Note: We don't set CUDA_VISIBLE_DEVICES, instead we use gpu_offset in nanovllm

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams

def apply_template(tokenizer:AutoTokenizer, prompts:list[str]) -> list[str]:
    """
    Apply chat template to a list of prompts using the provided tokenizer.
    
    Example:
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    >>> prompts = ["Hello, how are you?", "What is the capital of France?"]
    >>> apply_template(tokenizer, prompts)
    ['<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n<|im_start|>user\\nHello, how are you?<|im_end|>\\n<|im_start|>assistant\\n',
     '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n<|im_start|>user\\nWhat is the capital of France?<|im_end|>\\n<|im_start|>assistant\\n']
    """
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

def main():
    # Model paths (use absolute paths)
    model_base = "/inspire/hdd/global_user/donglinkang-253108120084/.cache/modelscope/hub/models/Qwen"
    base_model_path = f"{model_base}/Qwen2.5-1.5B"
    instruct_model_path = f"{model_base}/Qwen2.5-1.5B-Instruct"

    print("\n" + "=" * 60)
    print("Step 0: Load Instruct model on GPU 0")
    print("=" * 60)
    
    # Load the Instruct model on GPU 0 (nanovllm uses GPU 1, 2)
    instruct_model = AutoModelForCausalLM.from_pretrained(
        instruct_model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_path)
    
    print(f"Loaded Instruct model with {sum(p.numel() for p in instruct_model.parameters())} parameters")
    
    print("=" * 60)
    print("Step 1: Initialize nanovllm with Qwen2.5-1.5B on GPU 1,2")
    print("=" * 60)

    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Initialize nanovllm LLM with tensor parallelism across 2 GPUs
    # gpu_offset=1 means use GPU 1, 2 (leaving GPU 0 for transformers)
    llm = LLM(
        base_model_path,
        tensor_parallel_size=2,
        gpu_offset=1,  # Use GPU 1, 2 instead of GPU 0, 1
        max_model_len=512,
    )
    
    # Test prompt
    prompts_str = [
        "Hello, how are you today?",
        "What is the capital of France?",
    ]
    prompts = apply_template(base_tokenizer, prompts_str)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=50)
    
    print("\n" + "=" * 60)
    print("Step 2: Generate with base model (before weight update)")
    print("=" * 60)
    
    outputs_before = llm.generate(prompts, sampling_params, use_tqdm=False)
    for i, output in enumerate(outputs_before):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Response: {output['text'][:200]}...")
    
    print("\n" + "=" * 60)
    print("Step 3: Sync weights from Instruct model to nanovllm")
    print("=" * 60)
    
    # Get the state dict from the Instruct model
    instruct_state_dict = instruct_model.state_dict()
    
    # Update nanovllm model with Instruct weights
    # The weights will be transferred to GPU and broadcast via NCCL
    print(f"Syncing {len(instruct_state_dict)} parameters...")
    start_time = time.perf_counter()
    llm.update_model_param(instruct_state_dict)
    end_time = time.perf_counter()
    print(f"Weight sync complete! Time taken: {end_time - start_time:.2f} seconds")
    
    # Free the CPU model to save memory
    del instruct_model
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("Step 5: Generate with updated model (after weight update)")
    print("=" * 60)
    prompts = apply_template(instruct_tokenizer, prompts_str)
    outputs_after = llm.generate(prompts, sampling_params, use_tqdm=False)
    for i, output in enumerate(outputs_after):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Response: {output['text'][:200]}...")
    
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print("\nBefore update (base model) vs After update (instruct model):")
    print("You should see different response styles - the instruct model")
    print("typically gives more helpful and structured responses.")
    
    # Cleanup
    llm.exit()
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
