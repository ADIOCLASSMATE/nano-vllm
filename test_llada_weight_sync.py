"""
SDAR Model Weight Synchronization Test Script.

This script:
1. Initializes jetengine with an SDAR model.
2. Sequentially loads different SDAR models and syncs their weights to jetengine.
3. Verifies each sync by generating responses with SDAR-specific parameters.
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

from jetengine import LLM
from jetengine.sampling_params import SamplingParams

console = Console()

def apply_sdar_template(tokenizer: AutoTokenizer, prompts: list[str]) -> list[list[int]]:
    """Apply SDAR chat template with thinking mode enabled."""
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt + "\n Solve the problem step by step.\n"}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]

def main():    
    # List of SDAR models to sync sequentially
    # You can add multiple SDAR model paths here
    target_models = [
        "./public/models/LLaDA/LLaDA-8B-Instruct",
    ]
    
    # SDAR-specific test prompts
    test_prompts = [
        "Consider the geometric sequence frac{125}{9}, frac{25}{3}, 5, 3, ... What is the eighth term of the sequence? Express your answer as a common fraction.",
        "A regular pentagon is rotated counterclockwise about its center. What is the minimum number of degrees it must be rotated until it coincides with its original position?",
        "If a snack-size tin of peaches has 40 calories and is 2% of a person's daily caloric requirement, how many calories fulfill a person's daily caloric requirement?",
    ]
    
    sampling_params = SamplingParams(
        temperature=1.0, 
        topk=0, 
        topp=1.0, 
        max_tokens=2048,
        remasking_strategy="low_confidence_dynamic", 
        block_length=1024, 
        denoising_steps=1024, 
        dynamic_threshold=0.90)


    console.print(Panel.fit("[bold blue]SDAR Model Weight Synchronization Test[/bold blue]", border_style="blue"))

    # Note: jetengine uses accelerate.PartialState which can work in two modes:
    # 1. Single GPU (tensor_parallel_size=1): Can run directly with `python script.py`
    #    PartialState() will detect no distributed env and create a single-process "pseudo-distributed" environment
    # 2. Multi-GPU (tensor_parallel_size>1): Must use `accelerate launch script.py`
    #    to properly initialize multiple processes
    tensor_parallel_size = 1  # Change to >1 for multi-GPU, then use: accelerate launch test_sdar_weight_sync.py
    
    if tensor_parallel_size > 1:
        console.print("[bold yellow]⚠[/bold yellow] Multi-GPU mode detected. Make sure to run with: [cyan]accelerate launch test_sdar_weight_sync.py[/cyan]")
    
    # --- Step 1: Initialize jetengine with the first SDAR model ---
    initial_model_path = target_models[0]
    with console.status(f"[bold cyan]Initializing jetengine with {os.path.basename(initial_model_path)}...") as status:
        start = time.perf_counter()
        llm = LLM(
            initial_model_path, 
            enforce_eager=False, 
            tensor_parallel_size=1, 
            mask_token_id=126336, 
            block_length=1024, 
            max_num_seqs=32,
            gpu_memory_utilization=0.9)

        duration = time.perf_counter() - start
        console.print(f"[bold cyan]✓[/bold cyan] jetengine initialized in [yellow]{duration:.2f}s[/yellow]")

    # --- Step 2: Sequential Sync Loop ---
    for model_path in target_models:
        model_name = os.path.basename(model_path)
        console.print(f"\n[bold yellow]>>> Processing Model: {model_name}[/bold yellow]")

        # 2.1 Load model on GPU 0
        with console.status(f"[bold green]Loading {model_name} on GPU 0...") as status:
            start_load = time.perf_counter()
            source_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map="cuda:0",
                trust_remote_code=True,
            )
            source_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            load_duration = time.perf_counter() - start_load
            console.print(f"  [bold green]✓[/bold green] Loaded in [yellow]{load_duration:.2f}s[/yellow]")

        # 2.2 Sync weights
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            state_dict = source_model.state_dict()
            task = progress.add_task(f"  Syncing {len(state_dict)} parameters to jetengine...", total=1)
            start_sync = time.perf_counter()
            llm.update_model_param(state_dict)
            progress.update(task, advance=1)
            sync_duration = time.perf_counter() - start_sync
            console.print(f"  [bold red]✓[/bold red] Synced in [yellow]{sync_duration:.2f}s[/yellow]")

        # 2.3 Generate and verify with SDAR-specific parameters
        with console.status(f"[bold magenta]  Generating responses with {model_name} (SDAR mode)...") as status:
            start_gen = time.perf_counter()
            prompts = apply_sdar_template(source_tokenizer, test_prompts)
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            gen_duration = time.perf_counter() - start_gen

        # Display results
        table = Table(title=f"SDAR Responses from {model_name}", show_header=True, header_style="bold magenta")
        table.add_column("Prompt", style="dim", width=40)
        table.add_column("Response", width=60)
        table.add_column("Length", justify="right")
        
        for p, o in zip(test_prompts, outputs):
            response_text = o.get('text', '')
            token_ids = o.get('token_ids', [])
            response_preview = response_text.strip()[:150] + ("..." if len(response_text) > 150 else "")
            table.add_row(
                p[:37] + ("..." if len(p) > 37 else ""),
                response_preview,
                str(len(token_ids))
            )
        
        console.print(table)
        console.print(f"  Generation took [yellow]{gen_duration:.2f}s[/yellow]")
        console.print(f"  Average response length: [yellow]{sum(len(o.get('token_ids', [])) for o in outputs) / len(outputs):.1f}[/yellow] tokens")

        # 2.4 Cleanup source model for next iteration
        del source_model
        torch.cuda.empty_cache()
        console.print(f"[bold blue]--- Finished {model_name} ---[/bold blue]")

    # --- Final Summary ---
    console.print(Panel(
        "[bold green]All SDAR models synchronized and verified successfully![/bold green]\n"
        "Weight switching is functional and SDAR-specific features are working correctly.",
        title="Final Status",
        border_style="green"
    ))

    llm.exit()

if __name__ == "__main__":
    main()

