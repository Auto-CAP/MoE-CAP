import torch
import argparse
import math
import os
import asyncio
import aiohttp
import sys
import time
import traceback
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Literal, cast
from dataclasses import dataclass, field
from enum import Enum
from tqdm.asyncio import tqdm as async_tqdm

from moe_cap.model_loader import HFModelInfoRetriever
from moe_cap.utils.continuous_batching_utils import _calculate_continuous_metrics
from moe_cap.utils.acc_metrics import (
    compute_accuracy_metrics,
    extract_answer,
    format_accuracy_summary,
)
from moe_cap.utils.cost_utils import calculate_cost
from moe_cap.configs import CAPConfig
from moe_cap.data_loader.loader_registry import get_loader_for_task

import json
from transformers import AutoTokenizer
import re


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=100 * 60 * 60)


class BackendType(Enum):
    """Backend server type for API calls."""

    VLLM = "vllm"
    SGLANG = "sglang"
    AUTO = "auto"  # Auto-detect


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    output_len: int
    model: str
    extra_request_body: Dict[str, Any]
    prompt_len: int = 0  # Tokenized input length for per-request raw records
    ignore_eos: bool = False  # Whether to ignore EOS token for fixed-length generation


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
    return text[len(prefix) :] if text.startswith(prefix) else text


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
            "ignore_eos": request_func_input.ignore_eos,  # Use the ignore_eos setting
            **request_func_input.extra_request_body,
        }
        headers = get_auth_headers()

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        generated_text = ""
        output_len = 0
        usage = None
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
                            if data.get("usage"):
                                usage = data["usage"]

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
                    if usage:
                        output.prompt_len = usage.get("prompt_tokens", output.prompt_len) or output.prompt_len
                        output.output_len = usage.get("completion_tokens", output_len) or output_len
                    else:
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


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[async_tqdm] = None,
) -> RequestFuncOutput:
    """Send async request to OpenAI-compatible chat completions API.

    Non-streaming: TTFT/TPOT are measured server-side from ModelRunner forward
    records (sparse: expert recorder; dense/profiling-only: the profile-only
    timing recorder). Client stream timing is intentionally not used because it
    includes scheduler/request queueing, which is invalid at high concurrency.
    """
    api_url = request_func_input.api_url
    messages = json.loads(
        request_func_input.prompt
    )  # prompt stores JSON-encoded messages

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "messages": messages,
            "temperature": 0.0,
            "max_completion_tokens": request_func_input.output_len,
            "stream": False,
            "ignore_eos": request_func_input.ignore_eos,  # Force fixed-length chat outputs when requested
            **request_func_input.extra_request_body,
        }
        headers = get_auth_headers()

        output = RequestFuncOutput()
        st = time.perf_counter()

        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    choice = data["choices"][0]
                    # Handle both regular and reasoning model responses
                    msg = choice.get("message", {})
                    generated_text = msg.get("content") or ""
                    latency = time.perf_counter() - st
                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    usage = data.get("usage", {})
                    output.prompt_len = usage.get("prompt_tokens", request_func_input.prompt_len) or request_func_input.prompt_len
                    output.output_len = usage.get("completion_tokens", 0)
                else:
                    error_body = await response.text()
                    output.error = f"{response.status}: {error_body[:200]}"
                    output.success = False
        except Exception:
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            output.success = False

    if pbar:
        pbar.update(1)
    return output


def build_failure_zero_predictions(
    results: List[RequestFuncOutput],
    ground_truth: List[Any],
) -> Tuple[List[str], List[Any]]:
    """Failure-as-zero prediction/target pairing aligned to every sample.

    Evaluate EVERY ground-truth sample while preserving original index
    alignment. At index ``i`` the model's generated text is used only if result
    ``i`` exists and succeeded; otherwise an empty prediction is emitted, which
    naturally scores zero. Missing result objects (``results`` shorter than the
    ground truth) also count as zero. Failed requests are never silently
    excluded, so the returned lists always have ``len(ground_truth)`` entries and
    the reported total equals the full requested/evaluable sample count.
    """
    predictions: List[str] = []
    targets: List[Any] = list(ground_truth)
    for i in range(len(ground_truth)):
        if i < len(results) and results[i].success:
            predictions.append(results[i].generated_text)
        else:
            predictions.append("")
    return predictions, targets


def arena_hard_success_failure_indices(
    results: List[RequestFuncOutput],
    n_eval: int,
) -> Tuple[List[int], List[int]]:
    """Split ``n_eval`` evaluable samples into successful and failed indices.

    A sample is successful only if result ``i`` exists and succeeded; every other
    evaluable index (failed request or missing result object) is a failure that
    must still contribute a zero to the Arena-Hard aggregate. Indices are
    original dataset indices, so downstream question/UID/baseline alignment is
    preserved.
    """
    success_indices = [
        i for i in range(n_eval) if i < len(results) and results[i].success
    ]
    failed_indices = [
        i for i in range(n_eval) if not (i < len(results) and results[i].success)
    ]
    return success_indices, failed_indices


class OpenAIAPIMoEProfiler:
    def __init__(
        self,
        config: CAPConfig,
        output_dir: Optional[str] = None,
        api_url: Optional[str] = None,
        backend: str = "auto",
        ignore_eos: Optional[bool] = None,
        server_batch_size: Optional[int] = None,
        profiling_only: bool = False,
        use_chat_api: bool = False,
    ):
        """Initialize profiler from a CAPConfig object.

        Args:
            config: CAPConfig instance containing model and dataset info.
            output_dir: optional output directory. If not provided, will use './output'.
            api_url: OpenAI-compatible API endpoint URL.
            backend: Backend type - "vllm", "sglang", or "auto" (auto-detect).
            ignore_eos: Whether to ignore EOS token for fixed-length generation.
                       If None, auto-set based on fixed_length_mode in config.
            server_batch_size: Number of concurrent requests to send. If None, send all at once.
            profiling_only: Skip expert distribution recording, only collect TTFT/TPOT/throughput.
            use_chat_api: Use /v1/chat/completions endpoint instead of /v1/completions.
        """
        self.server_batch_size = server_batch_size
        self.profiling_only = profiling_only or config.profiling_only
        # store config
        self.config = config
        if api_url is None:
            raise ValueError("api_url must be provided")
        self.api_url = api_url
        self.use_chat_api = use_chat_api
        if (
            self.use_chat_api
            and "/v1/completions" in self.api_url
            and "/v1/chat/completions" not in self.api_url
        ):
            self.api_url = self.api_url.replace(
                "/v1/completions", "/v1/chat/completions"
            )
        self.request_fn = (
            async_request_openai_chat_completions
            if self.use_chat_api
            else async_request_openai_completions
        )

        # Extract base URL for control endpoints
        # e.g., http://localhost:8000/v1/completions -> http://localhost:8000
        from urllib.parse import urlparse

        parsed = urlparse(api_url)
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"

        # Backend detection and configuration
        self.backend_type = self._detect_or_set_backend(backend)
        print(f"Using backend: {self.backend_type.value}")
        if self.profiling_only:
            print(
                "Profiling-only mode: expert distribution recording disabled. "
                "Make sure the server was started with MOE_CAP_PROFILING_ONLY=1"
            )

        # Auto-set ignore_eos based on fixed_length_mode if not explicitly set
        if ignore_eos is None:
            self.ignore_eos = config.fixed_length_mode
            if self.ignore_eos:
                print(
                    "Fixed-length mode detected: enabling ignore_eos for accurate output length"
                )
        else:
            self.ignore_eos = ignore_eos

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
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_name, trust_remote_code=True
        )

    def _detect_or_set_backend(self, backend: str) -> BackendType:
        """Detect or set the backend type.

        Args:
            backend: "vllm", "sglang", or "auto"

        Returns:
            BackendType enum value
        """
        if backend.lower() == "vllm":
            return BackendType.VLLM
        elif backend.lower() == "sglang":
            return BackendType.SGLANG
        elif backend.lower() == "auto":
            # Try to auto-detect by checking available endpoints
            return self._auto_detect_backend()
        else:
            print(f"Warning: Unknown backend '{backend}', defaulting to auto-detect")
            return self._auto_detect_backend()

    def _auto_detect_backend(self) -> BackendType:
        """Auto-detect backend by probing available endpoints."""
        import requests

        # Try SGLang endpoint first (more specific)
        try:
            response = requests.get(f"{self.base_url}/get_model_info", timeout=5)
            if response.status_code == 200:
                data = response.json()
                # SGLang typically returns model_path in get_model_info
                if "model_path" in data:
                    print("Auto-detected SGLang backend")
                    return BackendType.SGLANG
        except Exception:
            pass

        # Try vLLM endpoint
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                print("Auto-detected vLLM backend")
                return BackendType.VLLM
        except Exception:
            pass

        # Default to vLLM if detection fails
        print("Warning: Could not auto-detect backend, defaulting to vLLM")
        return BackendType.VLLM

    def _load_data_for_task(self, task_name: str):
        """Load data for a single task name using the modern data loader APIs."""
        loader: Any
        try:
            loader, max_new_tokens = get_loader_for_task(task_name, self.config)
        except KeyError:
            raise ValueError(f"Unsupported task '{task_name}'. No loader registered.")

        all_input_raw = loader.get_input()
        system_prompts = getattr(loader, "system_prompts", None)
        return all_input_raw, max_new_tokens, system_prompts

    def _prepare_inputs(self, all_input_raw, max_new_tokens, system_prompts=None):
        """Prepare inputs for the model"""
        default_system = "Output the answer directly without description."

        if self.use_chat_api:
            prompts = []
            for i, q in enumerate(all_input_raw):
                sys_msg = system_prompts[i] if system_prompts else default_system
                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": q},
                ]
                prompts.append(json.dumps(messages))
            prompt_lengths = [len(self.tokenizer.encode(q)) for q in all_input_raw]
            return prompts, prompt_lengths, max_new_tokens
        else:
            chat_prompts = []
            for i, q in enumerate(all_input_raw):
                sys_msg = system_prompts[i] if system_prompts else default_system
                chat_prompts.append(
                    [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": q},
                    ]
                )
            chat_prompts = self.tokenizer.apply_chat_template(
                chat_prompts, add_generation_prompt=True, tokenize=False
            )
            prompt_lengths = [len(self.tokenizer.encode(p)) for p in chat_prompts]
            return chat_prompts, prompt_lengths, max_new_tokens

    def _check_batch_recording_status(self):
        """Check if batch recording endpoints are available."""
        import requests

        try:
            if self.backend_type == BackendType.SGLANG:
                # SGLang doesn't have a status endpoint, just return True
                return {"available": True, "backend": "sglang"}
            else:
                response = requests.get(
                    f"{self.base_url}/batch_recording_status", timeout=5
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Warning: Batch recording endpoints not available: {e}")
            if self.backend_type == BackendType.VLLM:
                print(
                    "Make sure you're using the custom vllm server (moe_cap.systems.vllm)"
                )
            return None

    def _start_batch_recording(self):
        """Start batch statistics recording on the server."""
        import requests

        try:
            if self.backend_type == BackendType.SGLANG:
                response = requests.post(
                    f"{self.base_url}/start_expert_distribution_record", timeout=300
                )
            else:
                response = requests.post(
                    f"{self.base_url}/start_batch_recording", timeout=300
                )
            response.raise_for_status()
            print(
                f"Started {'expert distribution' if self.backend_type == BackendType.SGLANG else 'batch'} recording on server"
            )
            return True
        except Exception as e:
            print(f"Warning: Could not start recording: {e}")
            return False

    def _stop_batch_recording(self):
        """Stop batch statistics recording on the server."""
        import requests

        try:
            if self.backend_type == BackendType.SGLANG:
                response = requests.post(
                    f"{self.base_url}/stop_expert_distribution_record", timeout=300
                )
            else:
                response = requests.post(
                    f"{self.base_url}/stop_batch_recording", timeout=300
                )
            response.raise_for_status()
            print(
                f"Stopped {'expert distribution' if self.backend_type == BackendType.SGLANG else 'batch'} recording on server"
            )
            return True
        except Exception as e:
            print(f"Warning: Could not stop recording: {e}")
            return False

    def _dump_batch_recording(self):
        """Dump and retrieve batch statistics from the server."""
        import requests

        try:
            if self.backend_type == BackendType.SGLANG:
                # SGLang dumps to file, need to trigger dump and then read from file
                response = requests.post(
                    f"{self.base_url}/dump_expert_distribution_record", timeout=300
                )
                response.raise_for_status()
                print("Expert distribution record dumped to file")

                # Read the dumped records from the file location
                server_output_base = os.environ.get(
                    "SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR",
                    os.path.join(os.getcwd(), "expert_records"),
                )
                record_file = os.path.join(
                    server_output_base,
                    self.hf_model_name,
                    "expert_distribution_record.jsonl",
                )

                if os.path.exists(record_file):
                    records = []
                    with open(record_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                records.append(json.loads(line))
                    print(
                        f"Retrieved {len(records)} expert distribution records from file"
                    )
                    return records
                else:
                    print(f"Warning: Expected record file not found at {record_file}")
                    return []
            else:
                response = requests.post(
                    f"{self.base_url}/dump_batch_recording", timeout=300
                )
                response.raise_for_status()
                data = response.json()
                records = data.get("records", [])
                print(f"Retrieved {len(records)} batch records from server")
                return records
        except Exception as e:
            print(f"Warning: Could not dump recording: {e}")
            return []

    def get_metrics(
        self,
        results: List[RequestFuncOutput],
        prompt_lengths: List[int],
        batch_size: int = 1,
        server_records: Optional[List[dict]] = None,
    ):
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
            print(
                "WARNING: No server records available. Using fallback with batch_size=1 approximation."
            )
            print("This will NOT reflect actual continuous batching behavior!")
            output_data = []
            for i, r in enumerate(successful_results):
                # Add prefill record (assuming no batching - batch_size=1)
                output_data.append(
                    {
                        "expert_activation": 0,  # Will be populated with actual expert data later
                        "latency": r.ttft,
                        "seq_lens_sum": prompt_lengths[i]
                        if i < len(prompt_lengths)
                        else 0,
                        "batch_size": 1,  # Approximation: client doesn't know actual server batch size
                        "forward_mode": "prefill",
                        "gpu_num": "N/A",
                    }
                )

                # Add decoding record (assuming no batching - batch_size=1)
                if r.output_len > 0:
                    # Average time per token for decoding
                    decode_time = r.latency - r.ttft
                    tpot = decode_time / r.output_len if r.output_len > 0 else 0
                    output_data.append(
                        {
                            "expert_activation": 0,  # Will be populated with actual expert data later
                            "latency": tpot,
                            "seq_lens_sum": r.output_len,
                            "batch_size": 1,  # Approximation: client doesn't know actual server batch size
                            "forward_mode": "decoding",
                            "gpu_num": "N/A",
                        }
                    )

        # Profiling-only (dense) runs still get authoritative TTFT/TPOT from the
        # server-side prefill/decode forward-latency records, but the MoE
        # hardware-utilization formulas in _calculate_continuous_metrics assume
        # sparse/expert structure and are not valid for dense models. Skip them
        # and only carry gpu_raw_type through (for cost/GPU labeling).
        if self.profiling_only:
            res_dict = {}
            if server_records:
                gpu_raw_type = server_records[0].get("gpu_raw_type")
                if gpu_raw_type is not None:
                    res_dict["gpu_raw_type"] = gpu_raw_type
        else:
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
                    output_data=output_data,
                )
            except Exception as e:
                print(f"Warning: Could not calculate continuous batching metrics: {e}")
                import traceback

                traceback.print_exc()
                res_dict = {}

        res_dict.update(
            {
                # "avg_ttft": total_ttft / len(successful_results),
                # "avg_latency": sum(r.latency for r in successful_results) / len(successful_results),
                # "avg_output_len": avg_output_len,
                # "avg_context_len": avg_context_len,
                # "decode_throughput_tokens_per_sec": total_output_tokens / total_decode_time if total_decode_time > 0 else 0,
                "total_requests": len(results),
                "successful_requests": len(successful_results),
                "failed_requests": len(results) - len(successful_results),
            }
        )

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
        prompt_lengths: Optional[List[int]] = None,
    ) -> Tuple[List[RequestFuncOutput], float]:
        """
        Send all prompts to the API and collect results.

        Args:
            prompts: List of prompts to send
            max_output_len: Maximum number of tokens to generate
            batch_size: Number of requests per batch. If None, send all at once.
            prompt_lengths: Optional tokenized input length per request for raw records.

        Returns:
            Tuple of (results, total_time)
        """
        # If no batch_size specified, send all at once
        if batch_size is None or batch_size >= len(prompts):
            tasks = []
            pbar = async_tqdm(total=len(prompts), desc="Processing requests")

            for idx, prompt in enumerate(prompts):
                request_input = RequestFuncInput(
                    prompt=prompt,
                    api_url=self.api_url,
                    output_len=max_output_len,
                    model=self.hf_model_name,
                    extra_request_body={},
                    prompt_len=prompt_lengths[idx] if prompt_lengths and idx < len(prompt_lengths) else 0,
                    ignore_eos=self.ignore_eos,  # Pass ignore_eos setting
                )
                tasks.append(self.request_fn(request_input, pbar))

            torch.cuda.synchronize()
            start_time = time.perf_counter()
            results = await asyncio.gather(*tasks)
            torch.cuda.synchronize()
            total_time = time.perf_counter() - start_time
            pbar.close()
            return results, total_time

        # Batched execution with 50% overlap
        all_results: List[Optional[RequestFuncOutput]] = [None] * len(prompts)
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
                        prompt_len=prompt_lengths[idx] if prompt_lengths and idx < len(prompt_lengths) else 0,
                        ignore_eos=self.ignore_eos,  # Pass ignore_eos setting
                    )
                    task = asyncio.create_task(self.request_fn(request_input, pbar))
                    active_tasks[task] = idx

                current_batch_size = batch_end_idx - batch_start_idx
                # Wait for at least one request from the current batch. The old
                # `current_batch_size // 2` becomes 0 for batch_size=1, which floods
                # all 256 requests despite --server-batch-size 1.
                threshold = max(1, current_batch_size // 2)  # 50% overlap, but sequential for batch_size=1
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

        return cast(List[RequestFuncOutput], all_results), total_time

    async def run_async(self):
        """Run profiling for all configured datasets."""
        # iterate over all datasets in the CAPConfig
        for dataset_name in self.dataset_names:
            print(f"Running profiling for dataset: {dataset_name}")

            # Load and prepare inputs
            all_input_raw, max_new_tokens, system_prompts = self._load_data_for_task(
                dataset_name
            )
            prompts, prompt_lengths, max_output_len = self._prepare_inputs(
                all_input_raw, max_new_tokens, system_prompts
            )

            # Get ground truth targets for evaluation
            try:
                loader: Any
                loader, _ = get_loader_for_task(dataset_name, self.config)
                ground_truth = loader.get_target()
            except Exception as e:
                print(f"Warning: Could not load ground truth for {dataset_name}: {e}")
                ground_truth = None

            # Both sparse and dense/profiling-only runs collect authoritative
            # TTFT/TPOT from server-side ModelRunner forward records via the same
            # control endpoints. In profiling-only mode the server installs a
            # dense-safe timing recorder (MOE_CAP_PROFILING_ONLY=1), so
            # start/stop/dump work without ExpertLocationMetadata. Client stream
            # timing is intentionally not used (it includes request queueing).
            self._start_batch_recording()

            # Run benchmark
            print(f"Sending {len(prompts)} requests to {self.api_url}")
            results, total_time = await self.run_benchmark(
                prompts=prompts,
                max_output_len=max_output_len,
                batch_size=self.server_batch_size,
                prompt_lengths=prompt_lengths,
            )

            # Stop recording and retrieve server-side forward records.
            self._stop_batch_recording()
            server_records = self._dump_batch_recording()

            num_gpus = 1
            if server_records and len(server_records) > 0:
                first_record = server_records[0]
                num_gpus = first_record.get("gpu_num", 1)
                print(f"Detected num_gpus from records: {num_gpus}")

            # Calculate metrics
            res_dict: Dict[str, Any] = self.get_metrics(
                results,
                prompt_lengths,
                batch_size=self.server_batch_size or 1,
                server_records=server_records,
            )

            # Compute accuracy metrics if ground truth is available
            if ground_truth is not None:
                try:
                    # Failure-as-zero policy: evaluate EVERY ground-truth sample.
                    # At index i use the generated text only if result i exists
                    # and succeeded; otherwise score an empty prediction (which
                    # naturally scores zero). Missing result objects also count
                    # zero. This keeps original index alignment intact and makes
                    # the reported total equal the full evaluable sample count --
                    # failed requests are never silently excluded.
                    predictions, targets = build_failure_zero_predictions(
                        results, ground_truth
                    )

                    # Compute accuracy using utility function
                    accuracy_metrics = compute_accuracy_metrics(
                        predictions=predictions,
                        targets=targets,
                        dataset_name=dataset_name,
                        extract_answers=True,
                    )
                    res_dict.update(accuracy_metrics)

                    # Print formatted accuracy summary
                    summary = format_accuracy_summary(accuracy_metrics)
                    print(f"Accuracy for {dataset_name}: {summary}")
                except Exception as e:
                    print(f"Warning: Could not compute accuracy metrics: {e}")

            # Arena-Hard: run LLM-as-a-judge evaluation if configured
            if dataset_name.lower() == "arena-hard" and self.config.judge_api_url:
                try:
                    from moe_cap.utils.arena_hard_judge import (
                        evaluate_arena_hard,
                        load_baseline_answers,
                        merge_failures_as_zero,
                    )

                    # Failure-as-zero policy: every evaluable Arena-Hard sample
                    # must contribute to the final aggregate. Successful
                    # generations are sent to the judge; failed/missing
                    # generations are scored zero (counted as losses) and merged
                    # back so the reported total stays the full evaluable sample
                    # count. Everything is indexed off the ORIGINAL dataset index
                    # so question / UID / baseline alignment is preserved.
                    n_eval = len(all_input_raw)
                    success_indices, failed_indices = (
                        arena_hard_success_failure_indices(results, n_eval)
                    )
                    predictions = [
                        extract_answer(results[i].generated_text, dataset_name)
                        for i in success_indices
                    ]
                    questions = [all_input_raw[i] for i in success_indices]

                    # Resolve baselines and UIDs for EVERY evaluable sample so
                    # both successful and failed indices stay aligned. Baselines
                    # for failed samples are still required (they are evaluable);
                    # a failed generation just scores zero without a judge call.
                    all_uids: Optional[List[str]] = None
                    baseline_by_index: Optional[Dict[int, str]] = None
                    if self.config.baseline_answers_path:
                        if not os.path.exists(self.config.baseline_answers_path):
                            raise FileNotFoundError(
                                f"Arena-Hard baseline file not found: {self.config.baseline_answers_path}"
                            )
                        baseline_dict = load_baseline_answers(
                            self.config.baseline_answers_path
                        )
                        if not baseline_dict:
                            raise ValueError(
                                f"No baseline answers loaded from {self.config.baseline_answers_path}"
                            )
                        # Match by uid if loader supports it, else fall back to index order
                        loader, _ = get_loader_for_task(dataset_name, self.config)
                        if hasattr(loader, "get_uids"):
                            all_uids = loader.get_uids()
                            eval_uids = [
                                all_uids[i] for i in range(n_eval) if i < len(all_uids)
                            ]
                            missing = sum(1 for uid in eval_uids if not baseline_dict.get(uid))
                            if missing > 0 or len(eval_uids) < n_eval:
                                raise ValueError(
                                    f"Baseline missing for {missing + (n_eval - len(eval_uids))}"
                                    f"/{n_eval} evaluable uids. Ensure baseline file and "
                                    "UIDs cover all dataset samples."
                                )
                            baseline_by_index = {
                                i: baseline_dict.get(all_uids[i], "")
                                for i in range(n_eval)
                            }
                        else:
                            # No UIDs: baseline can only be aligned positionally, so
                            # index it by the SAME original result indices. If any
                            # evaluable index falls outside the baseline list we
                            # cannot align safely -> fail loudly instead of shifting.
                            baseline_values = list(baseline_dict.values())
                            out_of_range = [
                                i for i in range(n_eval) if i >= len(baseline_values)
                            ]
                            if out_of_range:
                                raise ValueError(
                                    "Arena-Hard baseline has "
                                    f"{len(baseline_values)} positional entries but "
                                    f"{n_eval} evaluable samples; cannot align "
                                    "baselines without UIDs. Provide a UID-keyed "
                                    "baseline covering all samples."
                                )
                            baseline_by_index = {
                                i: baseline_values[i] for i in range(n_eval)
                            }
                    else:
                        print(
                            "Warning: No baseline_answers_path configured for Arena-Hard judge evaluation"
                        )
                        baseline_by_index = None

                    if baseline_by_index is not None:
                        # Baselines aligned to the successful generations sent to
                        # the judge, and to the full evaluable set for merging.
                        baseline_answers = [
                            baseline_by_index[i] for i in success_indices
                        ]

                        # Build judge API URL with auth
                        judge_api_url = self.config.judge_api_url
                        judge_model = self.config.judge_model or "gpt-4.1"
                        api_key = self.config.judge_api_key or os.environ.get(
                            "OPENAI_API_KEY"
                        )

                        judge_result = await evaluate_arena_hard(
                            questions=questions,
                            model_answers=predictions,
                            baseline_answers=baseline_answers,
                            judge_api_url=judge_api_url,
                            judge_model=judge_model,
                            api_key=api_key,
                        )

                        # Explicit zero records for failed/missing generations,
                        # aligned to their original dataset index.
                        failed_records = [
                            {
                                "index": i,
                                "question": all_input_raw[i][:500],
                                "uid": all_uids[i] if all_uids is not None else None,
                                "model_answer": "",
                                "baseline_answer": baseline_by_index[i][:500],
                                "generation_error": (
                                    results[i].error[:500]
                                    if i < len(results)
                                    else "missing result"
                                ),
                            }
                            for i in failed_indices
                        ]

                        arena_metrics = merge_failures_as_zero(
                            judge_result=judge_result,
                            success_indices=success_indices,
                            failed_records=failed_records,
                            total=n_eval,
                        )
                        per_question = arena_metrics.pop("arena_hard_per_question", [])
                        res_dict.update(arena_metrics)
                        print(
                            f"Arena-Hard win rate: {arena_metrics['arena_hard_win_rate']}% "
                            f"(total={arena_metrics['arena_hard_total']}, "
                            f"{arena_metrics['arena_hard_failed_generations']} failed "
                            "generations scored 0)"
                        )
                        if per_question:
                            judge_dir = os.path.join(self.output_dir, self.get_model_simple_name())
                            os.makedirs(judge_dir, exist_ok=True)
                            judge_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            judge_log_path = os.path.join(
                                judge_dir,
                                f"arena_hard_judge_{dataset_name}_{judge_ts}.jsonl",
                            )
                            with open(judge_log_path, "w", encoding="utf-8") as f:
                                for rec in per_question:
                                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            print(f"Arena-Hard judge log written to {judge_log_path}")
                    else:
                        print(
                            "Warning: Not enough baseline answers for Arena-Hard evaluation"
                        )
                except Exception as e:
                    print(f"Warning: Arena-Hard judge evaluation failed: {e}")
                    import traceback

                    traceback.print_exc()

            # Auto-detect GPU type and number from hardware_utils
            gpu_raw_type = res_dict.get("gpu_raw_type", None)
            res_dict["cost"] = calculate_cost(
                round(total_time, 2), gpu_raw_type, num_gpus
            )
            gpu_type = gpu_raw_type if gpu_raw_type else "Unknown"

            # Remove gpu_raw_type from metrics if present
            if "gpu_raw_type" in res_dict:
                del res_dict["gpu_raw_type"]

            # Add metadata fields to the output
            res_dict["model_name"] = self.hf_model_name
            res_dict["method"] = (
                self.backend_type.value
            )  # Use detected/configured backend
            res_dict["precision"] = self.used_dtype
            num_requests = len(prompts)
            res_dict["e2e_s"] = round(total_time / max(num_requests, 1), 2)
            res_dict["server_batch_size"] = (
                self.server_batch_size
            )  # None indicates all inputs sent at once
            res_dict["gpu_type"] = f"{num_gpus}x{gpu_type}"
            res_dict["dataset"] = dataset_name
            res_dict["ignore_eos"] = self.ignore_eos  # Track if ignore_eos was used
            res_dict["profiling_only"] = self.profiling_only
            # Determine model type based on model name (heuristic)
            res_dict["model_type"] = (
                "instruct"
                if any(x in self.hf_model_name.lower() for x in ["instruct", "chat"])
                else "thinking"
            )

            # --- Prepare output directory ---
            dest_dir = os.path.join(self.output_dir, self.get_model_simple_name())
            os.makedirs(dest_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # === 1. Output Data (always write per-request raw records) ===
            output_data_path = os.path.join(
                dest_dir, f"output_data_{dataset_name}_{timestamp}.jsonl"
            )
            prefill_tokens_total = 0
            decode_generated_tokens_total = 0
            with open(output_data_path, "w", encoding="utf-8") as f:
                for i, result in enumerate(results):
                    fallback_input_tokens = prompt_lengths[i] if i < len(prompt_lengths) else 0
                    input_token_count = result.prompt_len or fallback_input_tokens
                    if result.success and result.output_len:
                        output_token_count = result.output_len
                    elif result.success and result.generated_text:
                        output_token_count = len(self.tokenizer.encode(result.generated_text))
                    else:
                        output_token_count = 0
                    prefill_tokens_total += input_token_count or 0
                    decode_generated_tokens_total += output_token_count or 0
                    record = {
                        "index": i,
                        # Historical fields kept for compatibility: input_tokens is a count,
                        # output_tokens is the generated text consumed by existing evaluators.
                        "input_tokens": fallback_input_tokens,
                        "output_tokens": result.generated_text if result.success else "",
                        # Explicit numeric per-request token counts for raw TEAS/MoE-CAP records.
                        "input_token_count": input_token_count,
                        "output_token_count": output_token_count,
                        "requested_output_tokens": max_output_len,
                        "success": result.success,
                        # Per-request timing so latency stats can always be
                        # recomputed from raw records (was missing historically
                        # -> old runs' per-request latency is unrecoverable).
                        "latency_s": getattr(result, "latency", 0) or 0,
                        "ttft_s": getattr(result, "ttft", 0) or 0,
                    }
                    if result.error:
                        record["error"] = result.error[:500]
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Output data written to {output_data_path}")

            # === 2. Metrics File (Always required) ===
            # Compute average expert activation for prefill and decode from server records
            prefill_activations = []
            decode_activations = []
            prefill_latencies = []
            decode_latencies = []
            prefill_batch_sizes = []
            decode_batch_sizes = []
            if server_records:
                for sr in server_records:
                    fm = sr.get("forward_mode")
                    is_prefill = fm == "prefill"
                    is_decode = fm in ("decode", "decoding")
                    ea = sr.get("expert_activation")
                    if ea is not None and ea >= 0:
                        if is_prefill:
                            prefill_activations.append(ea)
                        elif is_decode:
                            decode_activations.append(ea)
                    lat = sr.get("latency")
                    if lat is not None and lat >= 0:
                        if is_prefill:
                            prefill_latencies.append(lat)
                        elif is_decode:
                            decode_latencies.append(lat)
                    bs = sr.get("batch_size")
                    if bs is not None and bs >= 0:
                        if is_prefill:
                            prefill_batch_sizes.append(bs)
                        elif is_decode:
                            decode_batch_sizes.append(bs)

            avg_batch_size_prefill = (
                sum(prefill_batch_sizes) / len(prefill_batch_sizes)
                if prefill_batch_sizes
                else 0
            )
            avg_batch_size_decode = (
                sum(decode_batch_sizes) / len(decode_batch_sizes)
                if decode_batch_sizes
                else 0
            )

            successful_for_simple = [r for r in results if getattr(r, "success", False)]
            simple_ttft = (
                sum(getattr(r, "ttft", 0) for r in successful_for_simple) / len(successful_for_simple)
                if successful_for_simple
                else 0
            )
            simple_tpots = []
            for r in successful_for_simple:
                out_len = getattr(r, "output_len", 0) or 0
                latency = getattr(r, "latency", 0) or 0
                ttft = getattr(r, "ttft", 0) or 0
                if out_len > 0 and latency >= ttft:
                    simple_tpots.append((latency - ttft) / out_len)
            simple_tpot = sum(simple_tpots) / len(simple_tpots) if simple_tpots else 0

            metrics_dict = {
                "performance": {
                    "e2e_s": res_dict.get(
                        "e2e_s", round(total_time / max(len(prompts), 1), 2)
                    ),
                    "ttft": (sum(prefill_latencies)/len(prefill_latencies)) if prefill_latencies else (res_dict.get("ttft") or simple_ttft),
                    "tpot": (sum(decode_latencies)/len(decode_latencies)) if decode_latencies else (res_dict.get("tpot") or simple_tpot),
                },
                "expert_activation": {
                    "avg_expert_activation_prefill": (
                        sum(prefill_activations) / len(prefill_activations)
                        if prefill_activations
                        else 0
                    ),
                    "avg_expert_activation_decode": (
                        sum(decode_activations) / len(decode_activations)
                        if decode_activations
                        else 0
                    ),
                },
                "batch_token_profile": {
                    "prefill_tokens": prefill_tokens_total,
                    "prefill_tokens_per_request": prefill_tokens_total / max(num_requests, 1),
                    "prefill_avg_batch_size": avg_batch_size_prefill,
                    "decode_generated_tokens": decode_generated_tokens_total,
                    "decode_generated_tokens_per_request": decode_generated_tokens_total / max(num_requests, 1),
                    "decode_avg_batch_size": avg_batch_size_decode,
                },
            }

            # Quality: unified acc + total
            if "arena_hard_win_rate" in res_dict:
                metrics_dict["quality"] = {
                    "acc": res_dict["arena_hard_win_rate"] / 100.0,
                    "total": res_dict.get("arena_hard_total", 0),
                }
            elif "acc" in res_dict or "exact_match" in res_dict:
                metrics_dict["quality"] = {
                    "acc": res_dict.get("acc", res_dict.get("exact_match", 0.0)),
                    "total": res_dict.get("total", 0),
                }

            metrics_path = os.path.join(
                dest_dir, f"metrics_{dataset_name}_{timestamp}.json"
            )
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_dict, f, indent=4)
            print(f"Metrics written to {metrics_path}")

            # === 3. Metadata (Always required) ===
            # Detect inference engine name and version separately
            engine_name = self.backend_type.value
            engine_ver = "unknown"
            try:
                if self.backend_type == BackendType.SGLANG:
                    import sglang

                    engine_ver = getattr(sglang, "__version__", "unknown")
                else:
                    import vllm

                    engine_ver = getattr(vllm, "__version__", "unknown")
            except ImportError:
                pass

            metadata_dict = {
                "hardware": {
                    "gpu_type": gpu_type,
                    "num_gpus": num_gpus,
                },
                "model_config": {
                    "model_name": self.hf_model_name,
                    "precision": self.used_dtype,
                },
                "system_environment": {
                    "inference_engine": engine_name,
                    "inference_engine_version": engine_ver,
                    "batch_size": self.server_batch_size,
                    "avg_batch_size_prefill": avg_batch_size_prefill,
                    "avg_batch_size_decode": avg_batch_size_decode,
                },
            }

            metadata_path = os.path.join(
                dest_dir, f"metadata_{dataset_name}_{timestamp}.json"
            )
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata_dict, f, indent=4)
            print(f"Metadata written to {metadata_path}")

            # === 4. Detailed Results (per-forward-step from server records) ===
            detailed_output_path = os.path.join(
                dest_dir, f"detailed_results_{dataset_name}_{timestamp}.jsonl"
            )
            with open(detailed_output_path, "w", encoding="utf-8") as f:
                if server_records:
                    for i, sr in enumerate(server_records):
                        record = {
                            "index": i,
                            "forward_mode": sr.get("forward_mode", "unknown"),
                            "expert_activation": sr.get("expert_activation", 0),
                            "batch_size": sr.get("batch_size", 0),
                            "seq_lens_sum": sr.get("seq_lens_sum", 0),
                        }
                        if sr.get("forward_mode") == "prefill":
                            record["ttft"] = sr.get("latency", 0)
                        else:
                            record["tpot"] = sr.get("latency", 0)
                        f.write(json.dumps(record) + "\n")
                else:
                    print("Warning: No server records available for detailed results")
            print(f"Detailed results written to {detailed_output_path}")

    def run(self):
        """Synchronous wrapper for run_async."""
        asyncio.run(self.run_async())


def main():
    parser = argparse.ArgumentParser(
        description="MoE-CAP OpenAI API Profiler - Run benchmarks via OpenAI-compatible API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file:
  python -m moe_cap.runner.openai_api_profile --config-file configs/gsm8k_qwen3_235b.yaml --api-url http://localhost:8000/v1/completions

  # Using command-line arguments:
  python -m moe_cap.runner.openai_api_profile --model_name Qwen/Qwen3-235B-A22B --datasets gsm8k --metrics em f1 \
    --api-url http://localhost:8000/v1/completions

  # Fixed-length benchmarking via CLI:
  python -m moe_cap.runner.openai_api_profile --model_name deepseek-ai/DeepSeek-V2-Lite-Chat \
    --datasets longbench_v2 --fixed-length-mode --target-input-tokens 13000 --target-output-tokens 1000 --num-samples 100 \
    --api-url http://localhost:8000/v1/completions

  # Mix config file and CLI (CLI overrides config):
  python -m moe_cap.runner.openai_api_profile --config-file configs/base.yaml --model_name my/custom-model \
    --api-url http://localhost:8000/v1/completions
""",
    )
    # Config file option
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to a JSON or YAML config file that contains CAPConfig fields. CLI args override config file values.",
    )

    # Required fields (can come from config file, except api-url which is always required)
    parser.add_argument(
        "--model_name",
        type=str,
        help="HuggingFace model ID (required unless specified in config file)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="One or more dataset names (e.g. gsm8k nq), required unless specified in config file",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        required=True,
        help="OpenAI-compatible API endpoint URL (e.g., http://localhost:8000/v1/completions)",
    )
    parser.add_argument(
        "--use-chat-api",
        action="store_true",
        default=False,
        help="Use /v1/chat/completions instead of /v1/completions",
    )

    # CAPConfig optional fields
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Metrics to compute (e.g. em f1). Default: [] (no metrics)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["bfloat16", "float16", "float32", "int8", "int4"],
        help="Model precision. Default: bfloat16",
    )
    parser.add_argument(
        "--dataset-subset",
        type=str,
        default=None,
        help="Dataset subset to use (e.g. 'main' for gsm8k, or a specific LongBench task)",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default=None,
        help="Dataset split to use (e.g. test, validation). Default: test",
    )

    # Fixed-length benchmarking options
    parser.add_argument(
        "--fixed-length-mode",
        action="store_true",
        default=None,
        help="Enable fixed-length benchmarking mode (no accuracy eval, pure performance)",
    )
    parser.add_argument(
        "--target-input-tokens",
        type=int,
        default=None,
        help="Target input token length for fixed-length benchmarking",
    )
    parser.add_argument(
        "--target-output-tokens",
        type=int,
        default=None,
        help="Target output token length for fixed-length benchmarking",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples for fixed-length benchmarking",
    )

    # Server and output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for metrics. Default: ./output",
    )
    parser.add_argument(
        "--server-batch-size",
        type=int,
        default=None,
        help="Number of concurrent requests to send. If not set, all requests are sent at once.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["vllm", "sglang", "auto"],
        help="Backend server type. 'auto' will attempt to detect automatically.",
    )

    # EOS handling
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        default=None,
        help="Ignore EOS token to force fixed-length output. Auto-enabled for fixed_length_mode.",
    )
    parser.add_argument(
        "--no-ignore-eos",
        action="store_true",
        help="Explicitly disable ignore_eos even in fixed_length_mode.",
    )
    # Profiling-only mode
    parser.add_argument(
        "--profiling-only",
        action="store_true",
        default=None,
        help="Profiling-only mode: skip expert distribution recording, only collect "
        "TTFT/TPOT/throughput. Server must be started with MOE_CAP_PROFILING_ONLY=1.",
    )
    parser.add_argument(
        "--judge-api-url",
        type=str,
        default=None,
        help="OpenAI-compatible chat API URL for Arena-Hard judge (e.g. https://api.openai.com/v1/chat/completions)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Judge model name (default: gpt-4.1)",
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        default=None,
        help="API key for judge model (default: OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--baseline-answers-path",
        type=str,
        default=None,
        help="Path to baseline model answers JSONL for Arena-Hard evaluation",
    )
    args = parser.parse_args()

    # Handle ignore_eos logic
    ignore_eos = None  # Let profiler auto-detect based on config
    if args.ignore_eos:
        ignore_eos = True
    elif args.no_ignore_eos:
        ignore_eos = False

    # Load config file if provided (JSON or YAML). CLI args override file values.
    file_cfg = {}
    if args.config_file:
        cf = args.config_file
        if cf.endswith(".json"):
            with open(cf, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
        else:
            # try yaml first (if installed), fall back to json
            try:
                import yaml

                with open(cf, "r", encoding="utf-8") as f:
                    file_cfg = yaml.safe_load(f)
            except Exception:
                # try json fallback
                with open(cf, "r", encoding="utf-8") as f:
                    file_cfg = json.load(f)

    # Merge CLI args over file config
    merged = dict(file_cfg or {})

    # Override with CLI args if provided (None means not specified)
    if args.model_name is not None:
        merged["model_id"] = args.model_name
    if args.datasets is not None:
        merged["dataset_names"] = args.datasets
    if args.metrics is not None:
        merged["metrics"] = args.metrics
    if args.precision is not None:
        merged["precision"] = args.precision
    if args.dataset_subset is not None:
        merged["dataset_subset"] = args.dataset_subset
    if args.dataset_split is not None:
        merged["dataset_split"] = args.dataset_split
    if args.fixed_length_mode is not None:
        merged["fixed_length_mode"] = args.fixed_length_mode
    if args.target_input_tokens is not None:
        merged["target_input_tokens"] = args.target_input_tokens
    if args.target_output_tokens is not None:
        merged["target_output_tokens"] = args.target_output_tokens
    if args.num_samples is not None:
        merged["num_samples"] = args.num_samples
    if args.profiling_only is not None:
        merged["profiling_only"] = args.profiling_only
    if args.judge_api_url is not None:
        merged["judge_api_url"] = args.judge_api_url
    if args.judge_model is not None:
        merged["judge_model"] = args.judge_model
    if args.judge_api_key is not None:
        merged["judge_api_key"] = args.judge_api_key
    if args.baseline_answers_path is not None:
        merged["baseline_answers_path"] = args.baseline_answers_path

    # Validate required fields
    if not merged.get("model_id"):
        parser.error(
            "--model_name is required (or 'model_id' must be specified in the config file)"
        )
    if not merged.get("dataset_names"):
        parser.error(
            "--datasets is required (or 'dataset_names' must be specified in the config file)"
        )

    # Validate that all datasets have registered loaders
    from moe_cap.data_loader.loader_registry import _REGISTRY

    unsupported = [ds for ds in merged["dataset_names"] if ds.lower() not in _REGISTRY]
    if unsupported:
        available = sorted(_REGISTRY.keys())
        parser.error(
            f"Unsupported dataset(s): {', '.join(unsupported)}. "
            f"Available datasets: {', '.join(available)}"
        )

    # Build CAPConfig and pass it to the profiler
    dataset_names = cast(List[str], merged.get("dataset_names"))
    model_id = cast(str, merged.get("model_id"))
    cap_cfg = CAPConfig(
        dataset_names=dataset_names,
        metrics=merged.get("metrics", []),
        model_id=model_id,
        precision=merged.get("precision", "bfloat16"),
        dataset_subset=merged.get("dataset_subset"),
        dataset_split=merged.get("dataset_split", "test"),
        fixed_length_mode=merged.get("fixed_length_mode", False),
        target_input_tokens=merged.get("target_input_tokens"),
        target_output_tokens=merged.get("target_output_tokens"),
        num_samples=merged.get("num_samples"),
        profiling_only=merged.get("profiling_only", False),
        judge_api_url=merged.get("judge_api_url"),
        judge_model=merged.get("judge_model", "gpt-4.1"),
        judge_api_key=merged.get("judge_api_key"),
        baseline_answers_path=merged.get("baseline_answers_path"),
    )

    profiler = OpenAIAPIMoEProfiler(
        config=cap_cfg,
        output_dir=args.output_dir,
        api_url=args.api_url,
        backend=args.backend,
        ignore_eos=ignore_eos,
        server_batch_size=args.server_batch_size,
        profiling_only=cap_cfg.profiling_only,
        use_chat_api=args.use_chat_api,
    )

    profiler.run()


if __name__ == "__main__":
    main()
