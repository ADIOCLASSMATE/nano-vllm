import atexit
import math
import pickle
from dataclasses import fields
from time import perf_counter
from typing import Union, List, Optional, Dict, Any
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, RunType, SequenceStatus, RunMode
from nanovllm.engine.scheduler import Scheduler, ScheduleResult
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.loader import load_from_hf_model

# Optional: Import helper for KV estimation if available, else define simple fallback
try:
    from nanovllm.utils.statics import _estimate_kv_cache_usage, _actual_estimate_kv_cache_usage
except ImportError:
    def _estimate_kv_cache_usage(config): return 0, 0
    def _actual_estimate_kv_cache_usage(max_tokens, max_active, config): return 0, 0


class LLMEngine:

    def __init__(self, model: str, **kwargs):
        # 1. Parse Configuration
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = Config(model, **config_kwargs)

        # 2. Determine Run Mode (AR vs Diffusion)
        model_type = self.config.hf_config.model_type.lower()
        if any(x in model_type for x in ["sdar", "llada", "dream"]):
            self.mode = RunMode.DIFFUSION
        else:
            self.mode = RunMode.AR
        
        print(f"[LLMEngine] Mode detected: {self.mode.name} (Model type: {model_type})")

        # 3. KV Cache Estimation (UX only)
        try:
            est_blocks, est_bytes = _estimate_kv_cache_usage(self.config)
            if est_blocks > 0:
                est_gib = est_bytes / (1024 ** 3)
                print(
                    f"[KVCache] Estimating {est_blocks:,} blocks "
                    f"({est_gib:.2f} GiB) for up to {self.config.max_num_seqs:,} active sequences."
                )
        except Exception:
            pass

        # 4. Initialize Multiprocessing (Tensor Parallelism)
        self.ps = []
        self.events = []
        
        # Only spawn workers if TP > 1
        if self.config.tensor_parallel_size > 1:
            ctx = mp.get_context("spawn")
            for i in range(1, self.config.tensor_parallel_size):
                event = ctx.Event()
                process = ctx.Process(target=ModelRunner, args=(self.config, i, event))
                process.start()
                self.ps.append(process)
                self.events.append(event)
        
        # 5. Initialize Local ModelRunner (Rank 0)
        self.model_runner = ModelRunner(self.config, 0, self.events)

        # 6. Initialize Tokenizer & Scheduler
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model, use_fast=True, trust_remote_code=True
        )
        self.config.eos = self.tokenizer.eos_token_id
        
        # Handle Mask Token for Diffusion
        if self.mode == RunMode.DIFFUSION:
            if self.config.mask_token_id == -1:
                if self.tokenizer.mask_token_id is not None:
                    self.config.mask_token_id = self.tokenizer.mask_token_id
                elif self.tokenizer.pad_token_id is not None:
                    self.config.mask_token_id = self.tokenizer.pad_token_id
            assert self.config.mask_token_id is not None, "Diffusion models require a mask_token_id"

        self.scheduler = Scheduler(self.config)
        
        # Default consistency (Diffusion usually overrides this in generate, AR uses True)
        self.scheduler.consistent_sampling_params = True 
        
        atexit.register(self.exit)

    def exit(self):
        if not hasattr(self, 'model_runner') or self.model_runner is None:
            return  # Already exited
        
        # Signal workers to exit
        try:
            if self.config.tensor_parallel_size > 1:
                self.model_runner.call("exit")
        except Exception:
            pass # Handle race conditions during shutdown
            
        del self.model_runner
        self.model_runner = None
        
        for p in self.ps:
            p.join()
        
        torch.cuda.empty_cache()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        
        # Diffusion specific: remove existing padding if present to handle raw inputs
        if self.mode == RunMode.DIFFUSION and isinstance(prompt, list):
             if self.tokenizer.pad_token_id in prompt:
                try:
                    start = prompt.index(self.tokenizer.pad_token_id) + 1
                    prompt = prompt[start:]
                except ValueError:
                    pass

        # Ensure stop words
        if sampling_params.stop_words is None:
            sampling_params.stop_words = [self.tokenizer.eos_token_id]

        # Create Unified Sequence
        # Pass masking token only if Diffusion
        mask_id = self.config.mask_token_id if self.mode == RunMode.DIFFUSION else None
        
        seq = Sequence(
            token_ids=prompt, 
            sampling_params=sampling_params,
            mask_token_id=mask_id,
            mode=self.mode
        )
        
        self.scheduler.add(seq)

    def step(self):
        """
        Unified Step Logic:
        1. Schedule (Get batches)
        2. Run Prefill (if any)
        3. Run Decode/Denoise (if any)
        4. Collect Results
        """
        # 1. Schedule
        schedule_result: ScheduleResult = self.scheduler.schedule()
        
        if not schedule_result.has_work:
            return [], 0

        finished_sequences = []
        tokens_generated = 0

        # 2. Run Prefill
        if schedule_result.prefill:
            # ModelRunner dispatch (Rank 0 gets output, others get None)
            output = self.model_runner.call("run", schedule_result.prefill, RunType.PREFILL)
            # Postprocess
            # AR: output is token_ids (list[int]), Diffusion: output is logits (Tensor)
            self.scheduler.postprocess(schedule_result.prefill, output, RunType.PREFILL)

        # 3. Run Decode / Denoise
        if schedule_result.decode:
            output = self.model_runner.call("run", schedule_result.decode, RunType.DENOISE)
            
            # Postprocess
            # Note: scheduler.postprocess handles the logic split internally based on seq.mode
            self.scheduler.postprocess(schedule_result.decode, output, RunType.DENOISE)
            
            # Stats counting
            if self.mode == RunMode.AR:
                tokens_generated += len(schedule_result.decode) # 1 token per seq
            else:
                tokens_generated += sum(getattr(seq, "num_to_transfer", 0) for seq in schedule_result.decode)

        # 4. Collect Outputs (Deduplicated)
        seen_seq_ids = set()
        finished_outputs = []
        
        # Combine lists to check for completion
        all_active = schedule_result.prefill + schedule_result.decode
        
        for seq in all_active:
            if seq.seq_id in seen_seq_ids:
                continue
            if not seq.is_finished:
                # For streaming, we might want intermediate results, 
                # but let's stick to returning only when finished or for monitoring
                # Current logic: return data for external consumer to print/stream
                pass
            
            seen_seq_ids.add(seq.seq_id)
            
            # Prepare Output Dict
            out_data = {
                "seq_id": seq.seq_id,
                "token_ids": seq.token_ids,
                "is_finished": seq.is_finished
            }
            
            # Diffusion extras
            if self.mode == RunMode.DIFFUSION:
                 out_data.update({
                     "trajectory": getattr(seq, "trajectory", []),
                     "logprobs": getattr(seq, "logprobs", []),
                     "entropies": getattr(seq, "entropies", [])
                 })
            
            finished_outputs.append(out_data)

        return finished_outputs, tokens_generated

    def is_finished(self):
        return self.scheduler.is_finished()

    # -------------------------------------------------------------------------
    # Speculative / Parameter Update
    # -------------------------------------------------------------------------

    def update_model_param(self, param_dict: dict[str, torch.Tensor]):
        """
        Update model parameters and sync across all model runners.
        Useful for speculative decoding or online updates.
        """
        self.model_runner.update_model_param_with_broadcast(param_dict)

    # -------------------------------------------------------------------------
    # Generation Methods
    # -------------------------------------------------------------------------

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        """
        Unified generation entry point.
        """
        # Standardize inputs
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
            self.scheduler.consistent_sampling_params = True
        else:
            self.scheduler.consistent_sampling_params = False

        # Add requests
        for p, sp in zip(prompts, sampling_params):
            self.add_request(p, sp)

        # UI Setup
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        
        outputs = {}
        stall_counter = 0
        total_tokens = 0
        start_time = perf_counter()

        while not self.is_finished():
            step_outputs, num_tokens = self.step()
            total_tokens += num_tokens
            
            # Deadlock Check (Crucial for Diffusion)
            if not step_outputs and not self.is_finished() and num_tokens == 0:
                stall_counter += 1
                if stall_counter > 5:
                    print("\n[Warning] Deadlock detected: No progress possible. Batch size may be too large.")
                    break
            else:
                stall_counter = 0

            # Process Outputs
            current_time = perf_counter()
            throughput = total_tokens / (current_time - start_time + 1e-6)
            
            for out in step_outputs:
                # Update persistent storage
                outputs[out["seq_id"]] = out
                
                # Update Progress Bar (only count finished sequences)
                if use_tqdm and out["is_finished"]:
                    # Check if we haven't counted this one yet to avoid double update
                    # (This logic is simplified, usually you'd track finished set)
                    pass 

            if use_tqdm:
                # Update stats
                finished_count = sum(1 for o in outputs.values() if o["is_finished"])
                pbar.n = finished_count
                pbar.set_postfix({"Throughput": f"{int(throughput)} tok/s"})
                pbar.refresh()

        if use_tqdm:
            pbar.close()

        # Format Final Result
        final_results = []
        sorted_ids = sorted(outputs.keys())
        
        for seq_id in sorted_ids:
            data = outputs[seq_id]
            token_ids = data["token_ids"]
            
            try:
                text = self.tokenizer.decode(token_ids)
            except Exception:
                text = ""
                
            res = {
                "text": text,
                "token_ids": token_ids,
            }
            if self.mode == RunMode.DIFFUSION:
                res["trajectory"] = data.get("trajectory")
                res["logprobs"] = data.get("logprobs")
                res["entropies"] = data.get("entropies")
                
            final_results.append(res)
            
        return final_results

    def generate_streaming(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        max_active: int | None = None,
        use_tqdm: bool = True,
    ) -> list[dict]:
        """
        Streaming generation that keeps a constant number of active sequences.
        Recommended for large batch Diffusion generation to avoid OOM.
        """
        # UX Estimation
        if self.mode == RunMode.DIFFUSION:
            try:
                max_toks = sampling_params.max_tokens if not isinstance(sampling_params, list) else sampling_params[0].max_tokens
                _actual_estimate_kv_cache_usage(max_toks, max_active or 32, self.config)
            except: pass

        total_requests = len(prompts)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * total_requests
            self.scheduler.consistent_sampling_params = True
        
        if max_active is None:
            max_active = getattr(self.scheduler, "max_num_seqs", 32)

        if use_tqdm:
            pbar = tqdm(total=total_requests, desc="Generating (Stream)", dynamic_ncols=True)

        outputs = {}
        pending_idx = 0
        stall_counter = 0
        total_tokens = 0
        start_time = perf_counter()

        # Initial Fill
        initial_fill = min(max_active, total_requests)
        for i in range(initial_fill):
            self.add_request(prompts[i], sampling_params[i])
        pending_idx = initial_fill

        # Main Loop
        while not self.is_finished() or pending_idx < total_requests:
            # Refill
            active_count = len(getattr(self.scheduler, "running", [])) # Accessing scheduler internal
            while active_count < max_active and pending_idx < total_requests:
                self.add_request(prompts[pending_idx], sampling_params[pending_idx])
                pending_idx += 1
                active_count += 1

            # Step
            step_outputs, num_tokens = self.step()
            total_tokens += num_tokens
            
            # Deadlock
            if not step_outputs and not self.is_finished() and num_tokens == 0 and pending_idx == total_requests:
                stall_counter += 1
                if stall_counter > 5:
                    print("\n[Warning] Deadlock detected.")
                    break
            else:
                stall_counter = 0

            # Tracking
            current_time = perf_counter()
            throughput = total_tokens / (current_time - start_time + 1e-6)

            finished_this_step = 0
            for out in step_outputs:
                if out["is_finished"]:
                    # Store only finished results to save memory if needed, 
                    # or update intermediate. Here we overwrite.
                    outputs[out["seq_id"]] = out
                    finished_this_step += 1

            if use_tqdm:
                pbar.update(finished_this_step)
                pbar.set_postfix({"Throughput": f"{int(throughput)} tok/s"})

        if use_tqdm:
            pbar.close()

        # Format Results (Same as generate)
        final_results = []
        for seq_id in sorted(outputs.keys()):
            data = outputs[seq_id]
            token_ids = data["token_ids"]
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            res = {"text": text, "token_ids": token_ids}
            if self.mode == RunMode.DIFFUSION:
                res.update({k: data.get(k) for k in ["trajectory", "logprobs", "entropies"]})
            final_results.append(res)
            
        return final_results

    # -------------------------------------------------------------------------
    # Resource Management
    # -------------------------------------------------------------------------
    
    def offload_parameters(self, include_buffers: bool = False):
        """Move parameters to meta device to free GPU memory."""
        # Only works on the local rank 0 model runner instance currently
        if self.model_runner:
            self.model_runner.model.to_empty(device=torch.device("meta"))
            torch.cuda.empty_cache()
            print("Model parameters offloaded to meta device.")

    def reload_parameters(self, hf_model):
        """Reload weights from a HF model instance."""
        if self.model_runner:
            self.model_runner.reinit_model()
            load_from_hf_model(self.model_runner.model, hf_model=hf_model)
            # Re-allocate cache not strictly needed if shapes didn't change, 
            # but safe to do if reinit cleared it.