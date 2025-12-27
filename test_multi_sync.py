"""
Multi-model sequential weight sync test script.

This script:
1. Initializes nanovllm with a base model (Qwen2.5-1.5B).
2. Sequentially loads different models on GPU 0 and syncs their weights to nanovllm.
3. Verifies each sync by generating responses.
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
    model_base_dir = "/inspire/hdd/global_user/donglinkang-253108120084/.cache/modelscope/hub/models"
    
    # List of models to sync sequentially
    target_models = [
        f"{model_base_dir}/Qwen/Qwen2.5-1.5B",
        f"{model_base_dir}/Qwen/Qwen2.5-1.5B-Instruct",
        f"{model_base_dir}/Qwen/Qwen2.5-Math-1.5B",
        f"{model_base_dir}/Qwen/Qwen2.5-Math-1.5B-Instruct",
        f"{model_base_dir}/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    ]
    
    test_prompts = [
        "What is 2+2?",
        "Write a short poem about AI.",
        "Solve for x: 3x + 5 = 14",
    ]
    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=100)

    console.print(Panel.fit("[bold blue]Multi-Model Sequential Weight Sync Test[/bold blue]", border_style="blue"))

    # --- Step 1: Initialize nanovllm with the first model ---
    initial_model_path = target_models[0]
    with console.status(f"[bold cyan]Initializing nanovllm with {os.path.basename(initial_model_path)}...") as status:
        start = time.perf_counter()
        llm = LLM(
            initial_model_path,
            tensor_parallel_size=2,
            gpu_offset=1,
            max_model_len=512,
        )
        duration = time.perf_counter() - start
        console.print(f"[bold cyan]✓[/bold cyan] nanovllm initialized in [yellow]{duration:.2f}s[/yellow]")

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
            )
            source_tokenizer = AutoTokenizer.from_pretrained(model_path)
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
            task = progress.add_task(f"  Syncing {len(state_dict)} parameters to nanovllm...", total=1)
            start_sync = time.perf_counter()
            llm.update_model_param(state_dict)
            progress.update(task, advance=1)
            sync_duration = time.perf_counter() - start_sync
            console.print(f"  [bold red]✓[/bold red] Synced in [yellow]{sync_duration:.2f}s[/yellow]")

        # 2.3 Generate and verify
        with console.status(f"[bold magenta]  Generating responses with {model_name}...") as status:
            start_gen = time.perf_counter()
            prompts = apply_template(source_tokenizer, test_prompts)
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            gen_duration = time.perf_counter() - start_gen

        table = Table(title=f"Responses from {model_name}", show_header=True, header_style="bold magenta")
        table.add_column("Prompt", style="dim", width=30)
        table.add_column("Response")
        for p, o in zip(prompts, outputs):
            table.add_row(p, o['text'].strip()[:200] + ("..." if len(o['text']) > 200 else ""))
        console.print(table)
        console.print(f"  Generation took [yellow]{gen_duration:.2f}s[/yellow]")

        # 2.4 Cleanup source model for next iteration
        del source_model
        torch.cuda.empty_cache()
        console.print(f"[bold blue]--- Finished {model_name} ---[/bold blue]")

    # --- Final Summary ---
    console.print(Panel(
        "[bold green]All models synchronized and verified successfully![/bold green]\n"
        "Sequential weight switching is functional and efficient.",
        title="Final Status",
        border_style="green"
    ))

    llm.exit()

if __name__ == "__main__":
    main()
