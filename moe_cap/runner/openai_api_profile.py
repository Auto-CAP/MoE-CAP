import torch
import argparse
import os
import asyncio
import aiohttp
import sys
import time
import traceback
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from tqdm.asyncio import tqdm as async_tqdm

from moe_cap.model_loader import HFModelInfoRetriever
from moe_cap.utils.continuous_batching_utils import _calculate_continuous_metrics
from moe_cap.utils.acc_metrics import compute_accuracy_metrics, format_accuracy_summary
from moe_cap.configs import CAPConfig
from moe_cap.data_loader.loader_registry import get_loader_for_task

import json
from transformers import AutoTokenizer
import re


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=100 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    output_len: int
    model: str
    extra_request_body: Dict[str, Any]


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    error: str = ""
    output_len: int = 0
    prompt_len: int = 0


def get_auth_headers() -> Dict[str, str]:
    """Get authorization headers from environment variable."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    else:
        return {}


def remove_prefix(text: str, prefix: str) -> str:
    """Remove prefix from text if it exists."""
    return text[len(prefix):] if text.startswith(prefix) else text


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[async_tqdm] = None,
) -> RequestFuncOutput:
    """Send async request to OpenAI-compatible completions API."""
    api_url = request_func_input.api_url
    prompt = request_func_input.prompt

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": prompt,
            "temperature": 0.0,
            "best_of": 1,
            "max_tokens": request_func_input.output_len,
            "stream": False,
            "ignore_eos": False,
            **request_func_input.extra_request_body,
        }
        headers = get_auth_headers()

        output = RequestFuncOutput()
        generated_text = ""
        output_len = 0
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        if chunk == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(chunk)
                            
                            # Check if token was generated
                            if data["choices"][0].get("text"):
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.ttft = ttft
                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += data["choices"][0]["text"]
                                output_len += 1
                                
                        except json.JSONDecodeError:
                            continue

                    latency = time.perf_counter() - st
                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.output_len = output_len
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


class OpenAIAPIMoEProfiler:
    def __init__(self, config: CAPConfig, output_dir: str = None, api_url: str = None):
        """Initialize profiler from a CAPConfig object.

        Args:
            config: CAPConfig instance containing model and dataset info.
            output_dir: optional output directory. If not provided, will use './output'.
            api_url: OpenAI-compatible API endpoint URL.
        """
        # store config
        self.config = config
        self.api_url = api_url
        
        # Extract base URL for control endpoints
        # e.g., http://localhost:8000/v1/completions -> http://localhost:8000
        from urllib.parse import urlparse
        parsed = urlparse(api_url)
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"

        # dataset names (can be multiple)
        self.dataset_names = config.dataset_names or ["gsm8k"]

        # output dir
        self.output_dir = output_dir or "./output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Build HF model info retriever using the CAPConfig API
        self.hf_model_name = config.model_id
        self.model_info = HFModelInfoRetriever(config=config)
        moe_info = self.model_info.get_moe_info()
        attn_info = self.model_info.get_attention_info()

        # precision and dtype
        self.precision = self.model_info.get_model_precision_bytes()
        self.used_dtype = config.precision or "bfloat16"

        # architecture info
        arch = self.model_info.get_architecture_info()
        self.d_model = arch.get("hidden_size")
        self.n_layers = arch.get("num_hidden_layers")
        self.n_vocab = arch.get("vocab_size")

        # moe/attention info
        self.d_ff = moe_info.get("ffn_dim")
        self.total_experts = moe_info.get("num_experts_per_layer")
        self.used_experts = moe_info.get("moe_top_k")
        self.n_kv_heads = attn_info.get("num_key_value_heads")
        self.n_attn_heads = attn_info.get("num_attention_heads", self.n_kv_heads)
        self.d_head = attn_info.get("head_dim")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name, trust_remote_code=True)

    def _load_data_for_task(self, task_name: str):
        """Load data for a single task name using the modern data loader APIs."""
        try:
            loader, max_new_tokens = get_loader_for_task(task_name, self.config)
        except KeyError:
            raise ValueError(f"Unsupported task '{task_name}'. No loader registered.")

        all_input_raw = loader.get_input()
        return all_input_raw, max_new_tokens
        
    def _prepare_inputs(self, all_input_raw, max_new_tokens):
        """Prepare inputs for the model"""       
        system_prompt = "You are an expert problem solver. Provide concise answers."
        chat_prompts = [[{"role": "system", "content": system_prompt},
                        {"role": "user", "content": q}] for q in all_input_raw]
        chat_prompts = self.tokenizer.apply_chat_template(
            chat_prompts, 
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Calculate prompt lengths
        prompt_lengths = [len(self.tokenizer.encode(p)) for p in chat_prompts]
        
        return chat_prompts, prompt_lengths, max_new_tokens
    
    def _check_batch_recording_status(self):
        """Check if batch recording endpoints are available."""
        import requests
        try:
            response = requests.get(f"{self.base_url}/batch_recording_status", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Warning: Batch recording endpoints not available: {e}")
            print("Make sure you're using the custom vllm server (moe_cap.systems.vllm)")
            return None
    
    def _start_batch_recording(self):
        """Start batch statistics recording on the server."""
        import requests
        try:
            response = requests.post(f"{self.base_url}/start_batch_recording", timeout=10)
            response.raise_for_status()
            print("Started batch recording on server")
            return True
        except Exception as e:
            print(f"Warning: Could not start batch recording: {e}")
            return False
    
    def _stop_batch_recording(self):
        """Stop batch statistics recording on the server."""
        import requests
        try:
            response = requests.post(f"{self.base_url}/stop_batch_recording", timeout=10)
            response.raise_for_status()
            print("Stopped batch recording on server")
            return True
        except Exception as e:
            print(f"Warning: Could not stop batch recording: {e}")
            return False
    
    def _dump_batch_recording(self):
        """Dump and retrieve batch statistics from the server."""
        import requests
        try:
            response = requests.post(f"{self.base_url}/dump_batch_recording", timeout=10)
            response.raise_for_status()
            data = response.json()
            records = data.get("records", [])
            print(f"Retrieved {len(records)} batch records from server")
            return records
        except Exception as e:
            print(f"Warning: Could not dump batch recording: {e}")
            return []
        except Exception as e:
            print(f"Warning: Could not dump batch recording: {e}")
            return []
    
    def get_metrics(self, results: List[RequestFuncOutput], prompt_lengths: List[int], batch_size: int = 1, server_records: List[dict] = None):
        """Calculate metrics from profiling results.
        
        Args:
            results: List of request outputs
            prompt_lengths: List of prompt lengths
            batch_size: Batch size used
            server_records: Optional list of batch records from server
        """
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {"error": "No successful requests"}
        
        # Use server records if available, otherwise create from results
        if server_records:
            output_data = server_records
            print(f"Using {len(output_data)} server-recorded batch statistics")
        else:
            # Fallback: Convert results to continuous batching format
            # WARNING: This fallback assumes batch_size=1 (no batching) which is inaccurate
            # for continuous batching servers. Server records should be used for accurate metrics.
            print("WARNING: No server records available. Using fallback with batch_size=1 approximation.")
            print("This will NOT reflect actual continuous batching behavior!")
            output_data = []
            for i, r in enumerate(successful_results):
                # Add prefill record (assuming no batching - batch_size=1)
                output_data.append({
                    'expert_activation': 0,  # Will be populated with actual expert data later
                    'latency': r.ttft,
                    'seq_lens_sum': prompt_lengths[i] if i < len(prompt_lengths) else 0,
                    'batch_size': 1,  # Approximation: client doesn't know actual server batch size
                    'forward_mode': 'prefill',
                    'gpu_num': "N/A"
                })
                
                # Add decoding record (assuming no batching - batch_size=1)
                if r.output_len > 0:
                    # Average time per token for decoding
                    decode_time = r.latency - r.ttft
                    tpot = decode_time / r.output_len if r.output_len > 0 else 0
                    output_data.append({
                        'expert_activation': 0,  # Will be populated with actual expert data later
                        'latency': tpot,
                        'seq_lens_sum': r.output_len,
                        'batch_size': 1,  # Approximation: client doesn't know actual server batch size
                        'forward_mode': 'decoding',
                        'gpu_num': "N/A"
                    })

        
        # Use continuous batching metrics calculation
        try:
            gpu_raw_type = output_data[0].get("gpu_raw_type", None)
            res_dict = _calculate_continuous_metrics(
                n_layers=self.n_layers,
                d_model=self.d_model,
                gpu_raw_type=gpu_raw_type,
                n_attn_heads=self.n_attn_heads,
                d_head=self.d_head,
                n_kv_heads=self.n_kv_heads,
                d_ff=self.d_ff,
                hf_config=getattr(self.model_info, "hf_config", None),
                num_gpus=output_data[0].get("gpu_num", 1) if output_data else 1,
                model_name=self.hf_model_name,
                used_dtype=self.used_dtype,
                precision=self.precision,
                output_data=output_data
            )
        except Exception as e:
            print(f"Warning: Could not calculate continuous batching metrics: {e}")
            import traceback
            traceback.print_exc()
            res_dict = {}
        
        res_dict.update({
            # "avg_ttft": total_ttft / len(successful_results),
            # "avg_latency": sum(r.latency for r in successful_results) / len(successful_results),
            # "avg_output_len": avg_output_len,
            # "avg_context_len": avg_context_len,
            # "decode_throughput_tokens_per_sec": total_output_tokens / total_decode_time if total_decode_time > 0 else 0,
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": len(results) - len(successful_results),
        })


        
        return res_dict

    def get_model_simple_name(self):
        """Get simplified model name for output directory."""
        norm_path = os.path.normpath(self.hf_model_name)
        parts = norm_path.split(os.sep)
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1]}"
        else:
            return self.hf_model_name

    async def run_benchmark(
        self,
        prompts: List[str],
        max_output_len: int,
        batch_size: Optional[int] = None,
    ) -> Tuple[List[RequestFuncOutput], float]:
        """
        Send all prompts to the API and collect results.

        Args:
            prompts: List of prompts to send
            max_output_len: Maximum number of tokens to generate
            batch_size: Number of requests per batch. If None, send all at once.

        Returns:
            Tuple of (results, total_time)
        """
        # If no batch_size specified, send all at once
        if batch_size is None or batch_size >= len(prompts):
            tasks = []
            pbar = async_tqdm(total=len(prompts), desc="Processing requests")

            for prompt in prompts:
                request_input = RequestFuncInput(
                    prompt=prompt,
                    api_url=self.api_url,
                    output_len=max_output_len,
                    model=self.hf_model_name,
                    extra_request_body={},
                )
                tasks.append(async_request_openai_completions(request_input, pbar))

            torch.cuda.synchronize()
            start_time = time.perf_counter()
            results = await asyncio.gather(*tasks)
            torch.cuda.synchronize()
            total_time = time.perf_counter() - start_time
            pbar.close()
            return results, total_time

        # Batched execution with 50% overlap
        all_results = [None] * len(prompts)
        pbar = async_tqdm(total=len(prompts), desc="Processing requests")

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        batch_start_idx = 0
        active_tasks = {}  # Maps task to its original index

        while batch_start_idx < len(prompts) or active_tasks:
            # Launch new batch if we haven't processed all prompts yet
            if batch_start_idx < len(prompts):
                batch_end_idx = min(batch_start_idx + batch_size, len(prompts))

                for idx in range(batch_start_idx, batch_end_idx):
                    prompt = prompts[idx]
                    request_input = RequestFuncInput(
                        prompt=prompt,
                        api_url=self.api_url,
                        output_len=max_output_len,
                        model=self.hf_model_name,
                        extra_request_body={},
                    )
                    task = asyncio.create_task(
                        async_request_openai_completions(request_input, pbar)
                    )
                    active_tasks[task] = idx

                current_batch_size = batch_end_idx - batch_start_idx
                threshold = current_batch_size // 2  # 50% of current batch
                completed_in_batch = 0

                # Wait until 50% of current batch is complete before launching next batch
                while completed_in_batch < threshold and active_tasks:
                    done, pending = await asyncio.wait(
                        active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
                    )

                    for task in done:
                        result = await task
                        idx = active_tasks.pop(task)
                        all_results[idx] = result

                        # Count if this task belongs to the current batch
                        if batch_start_idx <= idx < batch_end_idx:
                            completed_in_batch += 1

                batch_start_idx = batch_end_idx

            else:
                # No more batches to launch, just wait for remaining tasks
                done, pending = await asyncio.wait(
                    active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    result = await task
                    idx = active_tasks.pop(task)
                    all_results[idx] = result

        torch.cuda.synchronize()
        total_time = time.perf_counter() - start_time
        pbar.close()

        return all_results, total_time

    async def run_async(self, batch_size: Optional[int] = None):
        """Run profiling for all configured datasets."""
        # iterate over all datasets in the CAPConfig
        for dataset_name in self.dataset_names:
            print(f"Running profiling for dataset: {dataset_name}")

            # Load and prepare inputs
            all_input_raw, max_new_tokens = self._load_data_for_task(dataset_name)
            prompts, prompt_lengths, max_output_len = self._prepare_inputs(all_input_raw, max_new_tokens)

            # Get ground truth targets for evaluation
            try:
                loader, _ = get_loader_for_task(dataset_name, self.config)
                ground_truth = loader.get_target()
            except Exception as e:
                print(f"Warning: Could not load ground truth for {dataset_name}: {e}")
                ground_truth = None

            # Start batch recording on server
            self._start_batch_recording()

            # Run benchmark
            print(f"Sending {len(prompts)} requests to {self.api_url}")
            results, total_time = await self.run_benchmark(
                prompts=prompts,
                max_output_len=max_output_len,
                batch_size=batch_size,
            )

            # Stop batch recording and retrieve records
            self._stop_batch_recording()
            server_records = self._dump_batch_recording()

            num_gpus = 1
            if server_records and len(server_records) > 0:
                first_record = server_records[0]
                num_gpus = first_record.get("gpu_num", 1)
                print(f"Detected num_gpus from records: {num_gpus}")

            # Calculate metrics
            res_dict = self.get_metrics(results, prompt_lengths, batch_size=batch_size or 1, server_records=server_records)

            # Compute accuracy metrics if ground truth is available
            if ground_truth is not None:
                try:
                    # Extract predictions from results
                    predictions = [r.generated_text for r in results if r.success]
                    
                    # Compute accuracy using utility function
                    accuracy_metrics = compute_accuracy_metrics(
                        predictions=predictions,
                        targets=ground_truth[:len(predictions)],  # Match length in case some failed
                        dataset_name=dataset_name,
                        extract_answers=True
                    )
                    res_dict.update(accuracy_metrics)
                    
                    # Print formatted accuracy summary
                    summary = format_accuracy_summary(accuracy_metrics)
                    print(f"Accuracy for {dataset_name}: {summary}")
                except Exception as e:
                    print(f"Warning: Could not compute accuracy metrics: {e}")
            
            # Auto-detect GPU type and number from hardware_utils
            gpu_raw_type = res_dict.get("gpu_raw_type", None)
            if gpu_raw_type:
                gpu_name_pattern = re.compile(r'NVIDIA[\s-]+(RTX[\s-]+)?([A-Z0-9]+)')
                match = gpu_name_pattern.search(gpu_raw_type)  
                if match:
                    gpu_type = ''.join(filter(None, match.groups())).strip()
                else:
                    gpu_type = "Unknown"
            else:
                gpu_type = "Unknown"
            
            # Remove gpu_raw_type from metrics if present
            if "gpu_raw_type" in res_dict:
                del res_dict["gpu_raw_type"]

            # Add metadata fields to the output
            res_dict["model_name"] = self.hf_model_name
            res_dict["method"] = "vllm" ## Current hardcoded to vllm
            res_dict["precision"] = self.used_dtype
            res_dict["e2e_s"] = round(total_time, 2)
            res_dict["batch_size"] = batch_size if batch_size else None  # None indicates all inputs sent at once
            res_dict["gpu_type"] = f"{num_gpus}x{gpu_type}"
            res_dict["dataset"] = dataset_name
            # Determine model type based on model name (heuristic)
            res_dict["model_type"] = "instruct" if any(x in self.hf_model_name.lower() for x in ["instruct", "chat"]) else "thinking"
            
            print(f"Metrics for {dataset_name}: {res_dict}")

            # Save results
            dest_dir = os.path.join(self.output_dir, self.get_model_simple_name())
            os.makedirs(dest_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(dest_dir, f"cap_metrics_{dataset_name}_{timestamp}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(res_dict, f, indent=4)
            print(f"Metrics written to {output_path}")

            # Save detailed results
            detailed_output_path = os.path.join(dest_dir, f"detailed_results_{dataset_name}.jsonl")
            with open(detailed_output_path, 'w', encoding='utf-8') as f:
                for i, result in enumerate(results):
                    record = {
                        "index": i,
                        "prompt_length": prompt_lengths[i] if i < len(prompt_lengths) else 0,
                        "success": result.success,
                        "output_len": result.output_len,
                        "ttft": result.ttft,
                        "latency": result.latency,
                        "itl": result.itl,
                        "error": result.error,
                    }
                    f.write(json.dumps(record) + '\n')
            print(f"Detailed results written to {detailed_output_path}")

    def run(self, batch_size: Optional[int] = None):
        """Synchronous wrapper for run_async."""
        asyncio.run(self.run_async(batch_size=batch_size))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="HuggingFace model ID (required unless specified in config file)")
    parser.add_argument("--datasets", nargs='+', help="One or more dataset names (e.g. gsm8k), required unless specified in config file")
    parser.add_argument("--config-file", type=str, help="Path to a JSON or YAML config file that contains CAPConfig fields")
    parser.add_argument("--api-url", type=str, required=True, help="OpenAI-compatible API endpoint URL (e.g., http://localhost:8000/v1/completions)")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--batch-size", type=int, default=None, help="Number of requests per batch. If not set, all requests are sent at once.")
    args = parser.parse_args()

    # Load config file if provided (JSON or YAML). CLI args override file values.
    file_cfg = {}
    if args.config_file:
        cf = args.config_file
        if cf.endswith('.json'):
            with open(cf, 'r', encoding='utf-8') as f:
                file_cfg = json.load(f)
        else:
            # try yaml first (if installed), fall back to json
            try:
                import yaml
                with open(cf, 'r', encoding='utf-8') as f:
                    file_cfg = yaml.safe_load(f)
            except Exception:
                # try json fallback
                with open(cf, 'r', encoding='utf-8') as f:
                    file_cfg = json.load(f)

    # Merge CLI args over file config
    merged = dict(file_cfg or {})
    merged['model_id'] = args.model_name or merged.get('model_id')
    merged['dataset_names'] = args.datasets or merged.get('dataset_names')

    # Validate required fields
    if not merged.get('model_id'):
        parser.error("--model_name is required (or 'model_id' must be specified in the config file)")
    if not merged.get('dataset_names'):
        parser.error("--datasets is required (or 'dataset_names' must be specified in the config file)")

    # Validate that all datasets have registered loaders
    from moe_cap.data_loader.loader_registry import _REGISTRY
    unsupported = [ds for ds in merged['dataset_names'] if ds.lower() not in _REGISTRY]
    if unsupported:
        available = sorted(_REGISTRY.keys())
        parser.error(
            f"Unsupported dataset(s): {', '.join(unsupported)}. "
            f"Available datasets: {', '.join(available)}"
        )

    # Build CAPConfig and pass it to the profiler
    cap_cfg = CAPConfig(
        dataset_names=merged.get('dataset_names'),
        metrics=merged.get('metrics', []),
        model_id=merged.get('model_id'),
        precision=merged.get('precision', 'bfloat16'),
        dataset_subset=merged.get('dataset_subset'),
        dataset_split=merged.get('dataset_split', 'test')
    )

    profiler = OpenAIAPIMoEProfiler(
        config=cap_cfg,
        output_dir=args.output_dir,
        api_url=args.api_url,
    )

    profiler.run(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
