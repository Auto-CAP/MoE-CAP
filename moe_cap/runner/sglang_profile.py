import torch
import argparse
import os
from collections import defaultdict, Counter
from datetime import datetime
from moe_cap.model_loader import HFModelInfoRetriever
from moe_cap.utils.continuous_batching_utils import _calculate_continuous_metrics
from moe_cap.utils.acc_metrics import compute_accuracy_metrics, format_accuracy_summary
from moe_cap.utils.cost_utils import calculate_cost
from moe_cap.configs import CAPConfig
from moe_cap.data_loader import GSM8KLoader
from moe_cap.data_loader.loader_registry import get_loader_for_task
import requests
from sglang.lang.api import set_default_backend
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
import sglang as sgl
import json
from transformers import AutoTokenizer
import re

class SGLangMoEActivationAnalyzer:
    def __init__(self, config: CAPConfig, output_dir: str = None, ignore_eos: bool = None):
        """Initialize analyzer from a CAPConfig object.

        Args:
            config: CAPConfig instance containing model and dataset info.
            output_dir: optional output directory for metrics. If not provided, will use './output'.
            ignore_eos: Whether to ignore EOS token for fixed-length generation.
                       If None, auto-set based on fixed_length_mode in config.
        """
        # store config
        self.config = config
        
        # Auto-set ignore_eos based on fixed_length_mode if not explicitly set
        if ignore_eos is None:
            self.ignore_eos = config.fixed_length_mode
            if self.ignore_eos:
                print("Fixed-length mode detected: enabling ignore_eos for accurate output length")
        else:
            self.ignore_eos = ignore_eos

        # dataset names (can be multiple)
        self.dataset_names = config.dataset_names or ["gsm8k"]

        # output dir for metrics
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
        # the HFModelInfoRetriever returns keys like num_experts_per_layer and moe_top_k
        self.total_experts = moe_info.get("num_experts_per_layer")
        self.used_experts = moe_info.get("moe_top_k")
        self.n_kv_heads = attn_info.get("num_key_value_heads")
        self.d_head = attn_info.get("head_dim")

        # safe compute per-token kv size if values are present
        try:
            self.per_token_kv_size = 2 * (self.n_layers or 0) * (self.d_head or 0) * (self.n_kv_heads or 0)
        except Exception:
            self.per_token_kv_size = None

        # Initialize record storage
        self.record = []
        self.prefilling_finished = False

        # Expert activation tracking
        self.layer_expert_counts = defaultdict(Counter)
        self.current_layer = None


    def _load_data(self):
        """Compatibility wrapper: load data for the primary dataset (first in the list).

        Prefer using _load_data_for_task to explicitly load per-dataset.
        """
        return self._load_data_for_task(self.dataset_names[0])

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
        # Create batches
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        system_prompt = "You are an expert problem solver. Provide concise answers."
        chat_prompts = [[{"role": "system", "content": system_prompt},
                        {"role": "user", "content": q}] for q in all_input_raw]
        chat_prompts = tokenizer.apply_chat_template(chat_prompts, 
                                                     add_generation_prompt=True,
                                                     tokenize=False)
        arguments = [{"question": q, "max_new_tokens": max_new_tokens} for q in chat_prompts]
        return arguments  # Single batch for auto mod
    
    def get_metrics(self, records, num_gpus=1):
        gpu_raw_types = records[0].get("gpu_raw_type", None)
        res_dict = _calculate_continuous_metrics(
            n_layers=self.n_layers,
            d_model=self.d_model,
            gpu_raw_type=gpu_raw_types,
            n_attn_heads=self.n_kv_heads,
            d_head=self.d_head,
            n_kv_heads=self.n_kv_heads,
            d_ff=self.d_ff,
            hf_config=getattr(self.model_info, "hf_config", None),
            num_gpus=num_gpus,
            model_name=self.hf_model_name,
            used_dtype=self.used_dtype,
            precision=self.precision,
            output_data=records
        )
        return res_dict

    def get_model_simple_name(self):
        norm_path = os.path.normpath(self.hf_model_name)
        parts = norm_path.split(os.sep)
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1]}"
        else:
            return self.hf_model_name

    @staticmethod
    @sgl.function
    def run_sgl_with_eos(s, question, max_new_tokens):
        """Run SGLang with normal EOS handling."""
        s += question
        s += sgl.gen(
            "answer",
            max_tokens=max_new_tokens,
            stop=["Question", "Assistant:", "<|separator|>"],
        )

    @staticmethod
    @sgl.function
    def run_sgl_ignore_eos(s, question, max_new_tokens):
        """Run SGLang ignoring EOS token for fixed-length output."""
        s += question
        s += sgl.gen(
            "answer",
            max_tokens=max_new_tokens,
            ignore_eos=True,  # Force exact token count
        )

    def run(self, port=30000):
        set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
        
        # Select the appropriate run function based on ignore_eos setting
        run_fn = self.run_sgl_ignore_eos if self.ignore_eos else self.run_sgl_with_eos
        if self.ignore_eos:
            print("Using ignore_eos mode for fixed-length generation")

        # iterate over all datasets in the CAPConfig
        for dataset_name in self.dataset_names:
            print(f"Running analysis for dataset: {dataset_name}")
            
            import time
            start_time = time.time()

            # Load and prepare inputs
            all_input_raw, max_new_tokens = self._load_data_for_task(dataset_name)
            batched_inputs = self._prepare_inputs(all_input_raw, max_new_tokens)

            # Get ground truth targets for evaluation
            try:
                loader, _ = get_loader_for_task(dataset_name, self.config)
                ground_truth = loader.get_target()
            except Exception as e:
                print(f"Warning: Could not load ground truth for {dataset_name}: {e}")
                ground_truth = None

            # Start recording
            response = requests.post(f"http://localhost:{port}/start_expert_distribution_record")
            response.raise_for_status()
            print("Started expert distribution recording.")

            # Run inference (auto mode: single batch request)
            states = run_fn.run_batch(
                batched_inputs,
                temperature=0,
                num_threads=128,
                progress_bar=True)

            end_time = time.time()
            e2e_time = end_time - start_time
            # Stop recording
            response = requests.post(f"http://localhost:{port}/stop_expert_distribution_record")
            response.raise_for_status()
            print("Stopped expert distribution recording.")

            # Dump expert distribution record
            response = requests.post(f"http://localhost:{port}/dump_expert_distribution_record")
            response.raise_for_status()

            # The server dumps the file to: {SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR}/{model_path}/expert_distribution_record.jsonl
            # By default, SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR is set to {server_cwd}/expert_records
            # We need to look for the file where the server actually saved it
            server_output_base = os.environ.get("SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR", 
                                                os.path.join(os.getcwd(), "expert_records"))
            tmp_record = os.path.join(server_output_base, self.hf_model_name, "expert_distribution_record.jsonl")
            
            # Expert records stay in the server location, just rename per dataset
            expert_dest_dir = os.path.join(server_output_base, self.hf_model_name)
            os.makedirs(expert_dest_dir, exist_ok=True)
            dest_record = os.path.join(expert_dest_dir, f"expert_distribution_record_{dataset_name}.jsonl")
            if os.path.exists(tmp_record):
                # atomic replace if possible
                try:
                    os.replace(tmp_record, dest_record)
                except Exception:
                    # fallback to copy
                    import shutil
                    shutil.copy(tmp_record, dest_record)
                    os.remove(tmp_record)
            else:
                print(f"Warning: expected dump at {tmp_record} not found.")

            # Read the dataset-specific record and compute metrics
            with open(dest_record, 'r', encoding='utf-8') as f:
                all_experts_record = [json.loads(line.strip()) for line in f]
            
            # Determine num_gpus from the record if possible
            num_gpus = 1
            if all_experts_record and len(all_experts_record) > 0:
                first_record = all_experts_record[0]
                num_gpus = first_record.get("gpu_num", 1)
                print(f"Detected num_gpus from records: {num_gpus}")
            
            # Pass tensor_parallel_size to get_metrics
            res_dict = self.get_metrics(all_experts_record, num_gpus=num_gpus)
            # Compute accuracy metrics if ground truth is available
            if ground_truth is not None:
                try:
                    # Extract predictions from states
                    predictions = [state["answer"] for state in states]
                    # print(f"Predictions: {predictions}")
                    # print(f"Ground truth: {ground_truth}")
                    
                    # Compute exact match using utility function
                    accuracy_metrics = compute_accuracy_metrics(
                        predictions=predictions,
                        targets=ground_truth,
                        dataset_name=dataset_name,
                        extract_answers=True
                    )
                    res_dict.update(accuracy_metrics)
                    
                    # Print formatted accuracy summary
                    summary = format_accuracy_summary(accuracy_metrics)
                    print(f"Accuracy for {dataset_name}: {summary}")
                except Exception as e:
                    print(f"Warning: Could not compute accuracy metrics: {e}")
            
            
            # Auto-detect GPU type from hardware_utils
            gpu_raw_type = res_dict.get("gpu_raw_type", None)
            
            # Calculate cost
            cost = calculate_cost(round(e2e_time, 2), gpu_raw_type, num_gpus)
            if cost is not None:
                res_dict['cost'] = cost
            
            if gpu_raw_type:
                gpu_name_pattern = re.compile(r'NVIDIA[\s-]+(RTX[\s-]+)?([A-Z0-9]+)')
                match = gpu_name_pattern.search(gpu_raw_type)  
                if match:
                    gpu_type = ''.join(filter(None, match.groups())).strip()
                else:
                    gpu_type = "Unknown"
            else:
                gpu_type = "Unknown"
            
            # Filter out gpu_raw_type from metrics
            if "gpu_raw_type" in res_dict:
                del res_dict["gpu_raw_type"]
            # Add metadata fields to the output
            res_dict["model_name"] = self.hf_model_name
            res_dict["method"] = "sglang"
            res_dict["precision"] = self.used_dtype
            res_dict["e2e_s"] = round(e2e_time, 2)
            res_dict["batch_size"] = None  # None indicates all inputs sent at once
            res_dict["gpu_type"] = f"{num_gpus}x{gpu_type}"
            res_dict["dataset"] = dataset_name
            res_dict["ignore_eos"] = self.ignore_eos  # Track if ignore_eos was used
            # Determine model type based on model name (heuristic)
            res_dict["model_type"] = "instruct" if any(x in self.hf_model_name.lower() for x in ["instruct", "chat"]) else "thinking"
            
            print(f"Metrics for {dataset_name}: {res_dict}")

            # Metrics go to output_dir
            metrics_dest_dir = os.path.join(self.output_dir, self.get_model_simple_name())
            os.makedirs(metrics_dest_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(metrics_dest_dir, f"cap_metrics_{dataset_name}_{timestamp}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(res_dict, f, indent=4)
            print(f"Metrics written to {output_path}")
            print(f"Expert records saved to {dest_record}")

def main():
    parser = argparse.ArgumentParser(
        description="MoE-CAP SGLang Profiler - Run benchmarks using SGLang server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file:
  python -m moe_cap.runner.sglang_profile --config-file configs/gsm8k_qwen3_235b.yaml

  # Using command-line arguments:
  python -m moe_cap.runner.sglang_profile --model_name Qwen/Qwen3-235B-A22B --datasets gsm8k --metrics em f1

  # Fixed-length benchmarking via CLI:
  python -m moe_cap.runner.sglang_profile --model_name deepseek-ai/DeepSeek-V2-Lite-Chat \
    --datasets longbench_v2 --fixed-length-mode --target-input-tokens 13000 --target-output-tokens 1000 --num-samples 100

  # Mix config file and CLI (CLI overrides config):
  python -m moe_cap.runner.sglang_profile --config-file configs/base.yaml --model_name my/custom-model
"""
    )
    # Config file option
    parser.add_argument("--config-file", type=str, help="Path to a JSON or YAML config file that contains CAPConfig fields. CLI args override config file values.")
    
    # Required fields (can come from config file)
    parser.add_argument("--model_name", type=str, help="HuggingFace model ID (required unless specified in config file)")
    parser.add_argument("--datasets", nargs='+', help="One or more dataset names (e.g. gsm8k nq), required unless specified in config file")
    
    # CAPConfig optional fields
    parser.add_argument("--metrics", nargs='*', default=None, help="Metrics to compute (e.g. em f1). Default: [] (no metrics)")
    parser.add_argument("--precision", type=str, default=None, choices=["bfloat16", "float16", "float32", "int8", "int4"],
                       help="Model precision. Default: bfloat16")
    parser.add_argument("--dataset-subset", type=str, default=None,
                       help="Dataset subset to use (e.g. 'main' for gsm8k, or a specific LongBench task)")
    parser.add_argument("--dataset-split", type=str, default=None,
                       help="Dataset split to use (e.g. test, validation). Default: test")
    
    # Fixed-length benchmarking options
    parser.add_argument("--fixed-length-mode", action="store_true", default=None,
                       help="Enable fixed-length benchmarking mode (no accuracy eval, pure performance)")
    parser.add_argument("--target-input-tokens", type=int, default=None,
                       help="Target input token length for fixed-length benchmarking")
    parser.add_argument("--target-output-tokens", type=int, default=None,
                       help="Target output token length for fixed-length benchmarking")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples for fixed-length benchmarking")
    
    # Server and output options
    parser.add_argument("--port", type=int, default=30000, help="Port for the SGLang server. Default: 30000")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for metrics. Default: ./output")
    
    # EOS handling
    parser.add_argument("--ignore-eos", action="store_true", default=None,
                       help="Ignore EOS token to force fixed-length output. Auto-enabled for fixed_length_mode.")
    parser.add_argument("--no-ignore-eos", action="store_true",
                       help="Explicitly disable ignore_eos even in fixed_length_mode.")
    args = parser.parse_args()
    
    # Handle ignore_eos logic
    ignore_eos = None  # Let analyzer auto-detect based on config
    if args.ignore_eos:
        ignore_eos = True
    elif args.no_ignore_eos:
        ignore_eos = False

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
    
    # Override with CLI args if provided (None means not specified)
    if args.model_name is not None:
        merged['model_id'] = args.model_name
    if args.datasets is not None:
        merged['dataset_names'] = args.datasets
    if args.metrics is not None:
        merged['metrics'] = args.metrics
    if args.precision is not None:
        merged['precision'] = args.precision
    if args.dataset_subset is not None:
        merged['dataset_subset'] = args.dataset_subset
    if args.dataset_split is not None:
        merged['dataset_split'] = args.dataset_split
    if args.fixed_length_mode is not None:
        merged['fixed_length_mode'] = args.fixed_length_mode
    if args.target_input_tokens is not None:
        merged['target_input_tokens'] = args.target_input_tokens
    if args.target_output_tokens is not None:
        merged['target_output_tokens'] = args.target_output_tokens
    if args.num_samples is not None:
        merged['num_samples'] = args.num_samples

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

    # Build CAPConfig and pass it to the analyzer (new API)
    cap_cfg = CAPConfig(
        dataset_names=merged.get('dataset_names'),
        metrics=merged.get('metrics', []),
        model_id=merged.get('model_id'),
        precision=merged.get('precision', 'bfloat16'),
        dataset_subset=merged.get('dataset_subset'),
        dataset_split=merged.get('dataset_split', 'test'),
        fixed_length_mode=merged.get('fixed_length_mode', False),
        target_input_tokens=merged.get('target_input_tokens'),
        target_output_tokens=merged.get('target_output_tokens'),
        num_samples=merged.get('num_samples'),
    )

    analyzer = SGLangMoEActivationAnalyzer(
        config=cap_cfg,
        output_dir=args.output_dir,
        ignore_eos=ignore_eos,
    )

    analyzer.run(args.port)


if __name__ == "__main__":
    main()
