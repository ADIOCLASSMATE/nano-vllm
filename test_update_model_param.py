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

# Set environment variables before importing any other modules
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams

def main():
    # Model paths (use absolute paths)
    model_base = "/inspire/hdd/global_user/donglinkang-253108120084/.cache/modelscope/hub/models/Qwen"
    base_model_path = f"{model_base}/Qwen2.5-1.5B"
    instruct_model_path = f"{model_base}/Qwen2.5-1.5B-Instruct"
    
    print("=" * 60)
    print("Step 1: Initialize nanovllm with Qwen2.5-1.5B on GPU 1,2")
    print("=" * 60)

    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Initialize nanovllm LLM with tensor parallelism across 2 GPUs
    llm = LLM(
        base_model_path,
        tensor_parallel_size=2,
        max_model_len=512,
    )
    
    # Test prompt
    prompts_str = [
        "Hello, how are you today?",
        "What is the capital of France?",
    ]
    prompts = [
        base_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts_str
    ]
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=50)
    
    print("\n" + "=" * 60)
    print("Step 2: Generate with base model (before weight update)")
    print("=" * 60)
    
    outputs_before = llm.generate(prompts, sampling_params, use_tqdm=False)
    for i, output in enumerate(outputs_before):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Response: {output['text'][:200]}...")
    
    print("\n" + "=" * 60)
    print("Step 3: Load Instruct model on CPU (to avoid GPU memory conflict)")
    print("=" * 60)
    
    # Load the Instruct model on CPU first, then we'll transfer weights
    # Note: We load on CPU because GPU 0 is not visible in this process
    instruct_model = AutoModelForCausalLM.from_pretrained(
        instruct_model_path,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_path)
    
    print(f"Loaded Instruct model with {sum(p.numel() for p in instruct_model.parameters())} parameters")
    
    print("\n" + "=" * 60)
    print("Step 4: Sync weights from Instruct model to nanovllm")
    print("=" * 60)
    
    # Get the state dict from the Instruct model
    instruct_state_dict = instruct_model.state_dict()
    
    # Update nanovllm model with Instruct weights
    # The weights will be transferred to GPU and broadcast via NCCL
    print(f"Syncing {len(instruct_state_dict)} parameters...")
    llm.update_model_param(instruct_state_dict)
    print("Weight sync complete!")
    
    # Free the CPU model to save memory
    del instruct_model
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("Step 5: Generate with updated model (after weight update)")
    print("=" * 60)
    prompts = [
        instruct_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts_str
    ]
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
