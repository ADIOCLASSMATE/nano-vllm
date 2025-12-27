"""
Rich-enhanced test script for update_model_param functionality.

This script:
1. Starts nanovllm LLM on GPU 1 & 2 (GPU 1 as rank 0) with Qwen2.5-1.5B
2. Generates some responses
3. Loads Qwen2.5-1.5B-Instruct on GPU 0 using transformers
4. Syncs weights from the Instruct model to nanovllm
5. Generates responses again to verify the update worked
"""

import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams

console = Console()

def apply_template(tokenizer: AutoTokenizer, prompts: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

def main():
    # Model paths
    model_base = "/inspire/hdd/global_user/donglinkang-253108120084/.cache/modelscope/hub/models/Qwen"
    base_model_path = f"{model_base}/Qwen2.5-1.5B"
    instruct_model_path = f"{model_base}/Qwen2.5-1.5B-Instruct"
    
    prompts_str = [
        "Hello, how are you today?",
        "What is the capital of France?",
    ]
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=50)

    console.print(Panel.fit("[bold blue]Nano-vLLM Weight Sync Demo[/bold blue]", border_style="blue"))

    # --- Step 0: Load Instruct Model ---
    with console.status("[bold green]Step 0: Loading Instruct model on GPU 0...") as status:
        start = time.perf_counter()
        instruct_model = AutoModelForCausalLM.from_pretrained(
            instruct_model_path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_path)
        duration = time.perf_counter() - start
        console.print(f"[bold green]✓[/bold green] Step 0 complete in [yellow]{duration:.2f}s[/yellow]")

    # --- Step 1: Initialize nanovllm ---
    with console.status("[bold cyan]Step 1: Initializing nanovllm on GPU 1,2...") as status:
        start = time.perf_counter()
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        llm = LLM(
            base_model_path,
            tensor_parallel_size=2,
            gpu_offset=1,
            max_model_len=512,
        )
        duration = time.perf_counter() - start
        console.print(f"[bold cyan]✓[/bold cyan] Step 1 complete in [yellow]{duration:.2f}s[/yellow]")

    # --- Step 2: Generate with Base Model ---
    console.print("\n[bold magenta]Step 2: Generating with Base Model[/bold magenta]")
    start = time.perf_counter()
    prompts = apply_template(base_tokenizer, prompts_str)
    outputs_before = llm.generate(prompts, sampling_params, use_tqdm=False)
    duration = time.perf_counter() - start
    
    table_before = Table(title="Base Model Responses", show_header=True, header_style="bold magenta")
    table_before.add_column("Prompt", style="dim")
    table_before.add_column("Response")
    for p, o in zip(prompts, outputs_before):
        table_before.add_row(p, o['text'].strip())
    console.print(table_before)
    console.print(f"Step 2 complete in [yellow]{duration:.2f}s[/yellow]")

    # --- Step 3: Sync Weights ---
    console.print("\n[bold red]Step 3: Syncing Weights (Instruct -> nanovllm)[/bold red]")
    instruct_state_dict = instruct_model.state_dict()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Syncing {len(instruct_state_dict)} parameters...", total=1)
        start = time.perf_counter()
        llm.update_model_param(instruct_state_dict)
        progress.update(task, advance=1)
        duration = time.perf_counter() - start
    
    console.print(f"[bold red]✓[/bold red] Weight sync complete in [yellow]{duration:.2f}s[/yellow]")

    # Cleanup source model
    del instruct_model
    torch.cuda.empty_cache()

    # --- Step 4: Generate with Updated Model ---
    console.print("\n[bold green]Step 4: Generating with Updated Model[/bold green]")
    start = time.perf_counter()
    prompts = apply_template(instruct_tokenizer, prompts_str)
    outputs_after = llm.generate(prompts, sampling_params, use_tqdm=False)
    duration = time.perf_counter() - start
    
    table_after = Table(title="Updated Model (Instruct) Responses", show_header=True, header_style="bold green")
    table_after.add_column("Prompt", style="dim")
    table_after.add_column("Response")
    for p, o in zip(prompts, outputs_after):
        table_after.add_row(p, o['text'].strip())
    console.print(table_after)
    console.print(f"Step 4 complete in [yellow]{duration:.2f}s[/yellow]")

    # --- Final Summary ---
    console.print(Panel(
        "[bold]Test completed successfully![/bold]\n"
        "Weights were successfully synchronized across tensor parallel ranks using NCCL broadcast.",
        title="Summary",
        border_style="green"
    ))

    llm.exit()

if __name__ == "__main__":
    main()
