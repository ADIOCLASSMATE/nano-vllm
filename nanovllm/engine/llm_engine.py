from __future__ import annotations


import atexit
from dataclasses import fields
from time import perf_counter
from typing import TYPE_CHECKING
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

if TYPE_CHECKING:
    import torch

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if not hasattr(self, 'model_runner') or self.model_runner is None:
            return  # Already exited
        self.model_runner.call("exit")
        del self.model_runner
        self.model_runner = None
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params, mask_token_id=self.scheduler.config.mask_token_id)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill, run_type = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill, run_type)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def update_model_param(self, param_dict: dict[str, "torch.Tensor"]):
        """Update model parameters and sync across all model runners using NCCL broadcast.
        
        Args:
            param_dict: A dictionary mapping parameter names to new tensor values.
                       Parameter names should match the source model's state_dict keys.
                       These should be FULL (unsharded) weights.
        
        Note:
            This method broadcasts full weights via NCCL (not shared memory).
            Each rank then applies its own weight_loader to handle tensor parallel sharding.
        """
        # For each parameter, broadcast full weight via NCCL then each rank shards locally
        self.model_runner.update_model_param_with_broadcast(param_dict)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
    def generate_streaming(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: "SamplingParams" | list["SamplingParams"],
        max_active: int | None = None,
        use_tqdm: bool = True,
    ) -> list[dict]:
        """
        Stream prompts through the engine while keeping up to `max_active` sequences running.
        As sequences finish, new prompts are added from the pending list to maximize GPU utilization.
        """
        total = len(prompts)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * total

        if max_active is None:
            max_active = self.scheduler.config.max_num_seqs

        if use_tqdm:
            pbar = tqdm(total=total, desc="Generating", dynamic_ncols=True)

        outputs: dict[int, dict] = {}
        pending_idx = 0
        stall_counter = 0

        # Prime initial requests up to capacity
        initial = min(max_active, total)
        for i in range(initial):
            self.add_request(prompts[i], sampling_params[i])
        pending_idx = initial

        total_generated_tokens = 0
        start_time = perf_counter()

        while not self.is_finished() or pending_idx < total:
            # Top up to capacity before each step
            running = getattr(self.scheduler, "running", [])
            deficit = max_active - len(running)
            while deficit > 0 and pending_idx < total:
                self.add_request(
                    prompts[pending_idx], sampling_params[pending_idx])
                pending_idx += 1
                deficit -= 1

            output, num_processed = self.step()

            if not output and not self.is_finished() and num_processed == 0 and pending_idx == total:
                stall_counter += 1
                if stall_counter > 3:
                    print("\n[Warning] Deadlock detected: No progress can be made because all sequences are waiting for KV cache blocks, but no blocks are free. This can happen if the batch size is too large for the available KV cache. Try reducing the number of concurrent requests.")
                    break
            else:
                stall_counter = 0

            total_generated_tokens += num_processed

            if use_tqdm:
                throughput = total_generated_tokens / \
                    (perf_counter() - start_time)
                pbar.set_postfix(
                    {"Throughput": f"{int(throughput)} tok/s"})
                pbar.update(len(output))

            for seq_id, token_ids in output:
                outputs[seq_id] = {
                    "token_ids": token_ids,
                    "Throughput": f"{int(throughput)} tok/s"
                }

        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]

        safe_outputs = []
        for output in outputs:
            token_ids = output["token_ids"]
            throughput = output["Throughput"]
            try:
                text = self.tokenizer.decode(token_ids)
            except Exception as e:
                print(f"[Warning] Decode failed for token_ids={token_ids}. Set the token to EOS.")
                token_ids = [self.tokenizer.eos_token_id]
                text = self.tokenizer.decode(token_ids)
            safe_outputs.append({
                "text": text,
                "token_ids": token_ids,
                "Throughput": throughput
            })

        if use_tqdm:
            pbar.close()
        return safe_outputs