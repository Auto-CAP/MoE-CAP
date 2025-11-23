#!/usr/bin/env python3
"""
vLLM Expert Distribution Demo

Usage:
  python test_expert_distribution.py <model_name>   

Examples:
  python test_expert_distribution.py Qwen/Qwen1.5-MoE-A2.7B
  python test_expert_distribution.py deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct

"""

import torch
import gc
import time
import sys
import os
import weakref
from typing import Dict, List, Any, Optional
from vllm import LLM, SamplingParams

# Add current directory to path for monkey patching
sys.path.insert(0, os.path.dirname(__file__))

# Import and apply monkey patching from vllm_integration
try:
    from vllm_integration import apply_vllm_monkey_patching
    apply_vllm_monkey_patching()
except ImportError as e:
    print(f"Warning: Could not import monkey patching: {e}")
    print("Expert distribution recording may not work properly.")
except Exception as e:
    print(f"Warning: Could not apply monkey patching: {e}")
    print("Expert distribution recording may not work properly.")


# Default model configuration
DEFAULT_MODEL_CONFIG = {
    "model_path": "Qwen/Qwen1.5-MoE-A2.7B",
    "tp_size": 4,
    "gpu_memory_utilization": 0.4,
}


def get_default_model_config():
    """Return the default model configuration."""
    return DEFAULT_MODEL_CONFIG


def create_llm_model(model_name: str, gpu_memory_utilization: float = 0.4):
    """Create LLM instance for the specified model."""
    return LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=128,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_log_stats=True,
        tensor_parallel_size=4,
    )


def force_llm_shutdown(llm: LLM):
    """Force shutdown of LLM to free GPU memory."""
    try:
        # Try to shutdown the engine core if it has the method
        if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'engine_core'):
            engine_core = llm.llm_engine.engine_core
            if hasattr(engine_core, 'shutdown'):
                engine_core.shutdown()

        # Force deletion of the LLM object
        del llm

        # Aggressive garbage collection
        gc.collect()

        # Clear CUDA cache multiple times
        for _ in range(3):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Wait longer for cleanup
        time.sleep(5)

    except Exception as e:
        print(f"Warning: Error during LLM shutdown: {e}")
        try:
            del llm
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass


def run_expert_analysis(llm: LLM, model_config: dict) -> dict:
    """Run expert distribution analysis for both modes and collect results."""

    prompt = "Write a simple Python function to calculate factorial."
    model_name = model_config["model_path"].split('/')[-1]
    results = {
        "model_name": model_name,
        "model_path": model_config["model_path"],
        "input_prompt": prompt,
        "modes": {}
    }

    for mode in ["stat", "per_token", "per_pass"]:
        mode_results = {"success": False, "error": None}

        try:
            # Configure and start recording
            llm.collective_rpc("configure_expert_distribution_recording", args=(mode, False)) # args: mode, verbose
            llm.collective_rpc("start_expert_distribution_recording")

            # Generate text
            outputs = llm.generate([prompt], SamplingParams(max_tokens=32, temperature=0.0), use_tqdm=False)
            generated_text = outputs[0].outputs[0].text.strip()

            mode_results["output_text"] = generated_text

            # Dump data BEFORE stopping (stopping clears the recorder state)
            all_data = llm.collective_rpc("dump_expert_distribution_record")
            
            # Stop recording after dumping
            llm.collective_rpc("stop_expert_distribution_recording")

            # Process data based on mode
            if mode == "stat":
                # STAT mode: only rank 0 returns data
                valid_data = [d for d in all_data if d and isinstance(d, dict) and "aggregated_expert_counts" in d]
                if not valid_data:
                    raise ValueError(f"No valid STAT data found (received {len(all_data)} workers)")
                data = valid_data[0]
                mode_results.update(analyze_stat_mode(data))
            elif mode == "per_token":
                # PER_TOKEN mode: aggregate records from all workers
                all_records = []
                for worker_data in all_data:
                    if isinstance(worker_data, dict) and "records" in worker_data:
                        all_records.extend(worker_data["records"])
                
                if not all_records:
                    raise ValueError(f"No records found (received {len(all_data)} workers)")
                
                # Create aggregated data structure
                data = {"records": all_records}
                mode_results.update(analyze_per_token_mode(data))
            elif mode == "per_pass":
                # PER_PASS mode: aggregate records from all workers
                all_records = []
                for worker_data in all_data:
                    if isinstance(worker_data, dict) and "records" in worker_data:
                        all_records.extend(worker_data["records"])
                
                if not all_records:
                    raise ValueError(f"No records found (received {len(all_data)} workers)")
                
                # Create aggregated data structure
                data = {"records": all_records}
                mode_results.update(analyze_per_pass_mode(data))

            mode_results["success"] = True

        except Exception as e:
            mode_results["error"] = str(e)

        results["modes"][mode] = mode_results

    return results


def analyze_stat_mode(data: dict) -> dict:
    """Analyze stat mode results and return data. Matches instructor's SGLang approach."""
    results = {"raw_data": None, "analysis": {}}

    if "aggregated_expert_counts" not in data:
        return results

    counts = data["aggregated_expert_counts"]
    
    # Handle different count formats
    if not counts:
        return results
    
    # Convert to tensor and handle shape
    counts_tensor = torch.tensor(counts)
    
    # STAT mode returns [num_forwards, num_experts] - already aggregated across layers
    if counts_tensor.ndim == 2:
        # Shape: [num_forwards, num_experts] - already summed across layers
        num_forwards, num_experts = counts_tensor.shape
        # Get num_layers from metadata if available, otherwise use 1
        num_layers = data.get("num_layers", 24)  # Default to 24 for Qwen MoE
    elif counts_tensor.ndim == 3:
        # Shape: [num_forwards, num_layers, num_experts] - per-layer data
        num_forwards, num_layers, num_experts = counts_tensor.shape
    else:
        # Unexpected shape
        return results

    # Calculate per-forward-pass expert activation
    if counts_tensor.ndim == 2:
        # [num_forwards, num_experts] - count active experts per forward pass
        activated_experts = (counts_tensor > 0).float()
        avg_activated_per_forward = activated_experts.sum(dim=1)  # [num_forwards] - total active experts per forward
    else:
        # [num_forwards, num_layers, num_experts] - per-layer data
        activated_experts = (counts_tensor > 0).float()
        activated_per_layer_per_forward = activated_experts.sum(dim=2)  # [num_forwards, num_layers]
        avg_activated_per_forward = activated_per_layer_per_forward.mean(dim=1)  # [num_forwards]

    # Separate prefill and decode based on forward pass patterns
    # In vLLM, the first forward pass is typically prefill (processing the full prompt)
    # Subsequent passes are decode (generating one token at a time)
    prefill_forwards = []  # Usually just one forward pass
    decode_forwards = []   # Usually multiple forward passes

    # Simple heuristic: first forward = prefill, rest = decode
    # (This matches the instructor's approach of tracking per forward pass)
    if num_forwards > 0:
        prefill_forwards = avg_activated_per_forward[:1]  # First forward pass
        if num_forwards > 1:
            decode_forwards = avg_activated_per_forward[1:]  # Remaining forward passes

    # Overall statistics (sum across all forwards)
    if counts_tensor.ndim == 2:
        # [num_forwards, num_experts] - sum across forwards
        expert_totals = counts_tensor.sum(dim=0)  # [num_experts]
    else:
        # [num_forwards, num_layers, num_experts] - sum across forwards and layers
        expert_totals = counts_tensor.sum(dim=(0, 1))  # [num_experts]
    
    total_tokens = expert_totals.sum().item()
    active_experts = (expert_totals > 0).sum().item()

    # Store raw data
    results["raw_data"] = {
        "aggregated_expert_counts": data["aggregated_expert_counts"][:2] if len(data["aggregated_expert_counts"]) > 0 else [],
        "num_layers": num_layers,
        "num_experts": num_experts,
        "num_forwards": num_forwards
    }

    # Analysis results
    results["analysis"] = {
        "model_info": f"{num_layers} layers, {num_experts} experts",
        "total_forwards": num_forwards,
        "total_tokens": total_tokens,
        "active_experts": active_experts,
        "expert_utilization": f"{active_experts}/{num_experts} ({active_experts/num_experts:.1%})",
        "expert_totals": expert_totals.tolist(),
    }

    # Per-forward-pass activation analysis (matches instructor's approach)
    forward_activations = []
    for forward_idx in range(num_forwards):
        forward_activation = avg_activated_per_forward[forward_idx].item()
        forward_activations.append(round(forward_activation, 3))

    results["analysis"]["per_forward_activation"] = forward_activations

    # Prefill analysis (first forward pass)
    if len(prefill_forwards) > 0:
        prefill_activation = prefill_forwards[0].item()
        results["analysis"]["prefill"] = {
            "forward_passes": 1,
            "avg_experts_activated": round(prefill_activation, 3),
            "description": "First forward pass (prompt processing)"
        }

    # Decode analysis (remaining forward passes)
    if len(decode_forwards) > 0:
        decode_avg_activation = decode_forwards.mean().item()
        results["analysis"]["decode"] = {
            "forward_passes": len(decode_forwards),
            "avg_experts_activated": round(decode_avg_activation, 3),
            "per_forward_activations": [round(x.item(), 3) for x in decode_forwards],
            "description": "Subsequent forward passes (token generation)"
        }

    # Overall activation statistics
    if len(avg_activated_per_forward) > 0:
        overall_avg_activation = avg_activated_per_forward.mean().item()
        results["analysis"]["overall_avg_activation"] = round(overall_avg_activation, 3)

        # Activation statistics
        nonzero_activations = avg_activated_per_forward[avg_activated_per_forward > 0]
        if len(nonzero_activations) > 1:
            activation_std = nonzero_activations.std().item()
            results["analysis"]["activation_std"] = round(activation_std, 3)
            results["analysis"]["activation_variation"] = round(activation_std / overall_avg_activation, 3)

    if active_experts > 1:
        nonzero = expert_totals[expert_totals > 0].float()
        cv = (nonzero.std() / nonzero.mean()).item()
        balancedness = 1.0 / (1.0 + cv)
        results["analysis"]["load_balancedness"] = round(balancedness, 3)
        results["analysis"]["coefficient_of_variation"] = round(cv, 3)

        # Top experts
        top_k = min(5, active_experts)
        top_indices = expert_totals.topk(top_k).indices.tolist()
        top_loads = expert_totals[top_indices].tolist()
        results["analysis"]["top_experts"] = list(zip(top_indices, top_loads))

    return results


def analyze_per_token_mode(data: dict) -> dict:
    """Analyze per-token mode results and return data."""
    results = {"raw_data": None, "analysis": {}}

    if "records" not in data:
        return results

    records = data["records"]

    # Analysis results
    results["analysis"] = {
        "total_tokens": len(records),
    }

    # Store raw data (first few records)
    if records:
        results["raw_data"] = records[:2]  # First 2 token records

        record = records[0]
        if "topk_ids_of_layer" in record:
            layer_routing = record["topk_ids_of_layer"]
            routed_layers = sum(1 for experts in layer_routing.values()
                              if any(e != -1 for e in experts))
            results["analysis"]["routed_layers_per_token"] = routed_layers

            # Store sample routing for first few layers
            sample_routing = {}
            for layer_idx, experts in list(layer_routing.items())[:3]:  # First 3 layers
                if any(e != -1 for e in experts):
                    sample_routing[layer_idx] = [e for e in experts if e != -1]
            results["analysis"]["sample_routing"] = sample_routing

    return results


def analyze_per_pass_mode(data: dict) -> dict:
    """Analyze per-pass mode results and return data."""
    results = {"raw_data": None, "analysis": {}}

    if "records" not in data:
        return results

    records = data["records"]

    # Analysis results
    total_passes = len(records)
    results["analysis"] = {
        "total_forward_passes": total_passes,
    }

    if records:
        # Store raw data (first few records)
        results["raw_data"] = records[:3]  # First 3 forward pass records

        # Extract activation statistics
        avg_activations = [r.get("avg_activated_per_layer", 0) for r in records]
        total_activations = [r.get("total_activated_experts", 0) for r in records]
        expert_utilizations = [r.get("expert_utilization", 0) for r in records]

        if avg_activations:
            avg_activation_mean = sum(avg_activations) / len(avg_activations)
            results["analysis"]["avg_activation_per_pass"] = round(avg_activation_mean, 3)

            if len(avg_activations) > 1:
                activation_std = (sum((x - avg_activation_mean) ** 2 for x in avg_activations) / len(avg_activations)) ** 0.5
                results["analysis"]["activation_variation"] = round(activation_std, 3)

        # Expert utilization statistics
        if expert_utilizations:
            avg_utilization = sum(expert_utilizations) / len(expert_utilizations)
            results["analysis"]["avg_expert_utilization"] = round(avg_utilization, 4)

        # Forward pass analysis
        results["analysis"]["per_pass_activations"] = [round(a, 3) for a in avg_activations]

        # Prefill vs Decode analysis (similar to stat mode)
        if total_passes > 0:
            # Assume first pass is prefill, rest are decode
            if total_passes >= 1:
                prefill_activation = avg_activations[0]
                results["analysis"]["prefill_activation"] = round(prefill_activation, 3)

            if total_passes > 1:
                decode_activations = avg_activations[1:]
                decode_avg = sum(decode_activations) / len(decode_activations)
                results["analysis"]["decode_avg_activation"] = round(decode_avg, 3)
                results["analysis"]["decode_passes"] = len(decode_activations)

    return results


def print_usage_info():
    """Print usage information and default model."""
    print("vLLM Expert Distribution Test")
    print("=" * 40)
    print(f"Default model: {DEFAULT_MODEL_CONFIG['model_path']}")
    print(f"Tensor Parallel Size: {DEFAULT_MODEL_CONFIG['tp_size']}")
    print(f"GPU Memory Utilization: {DEFAULT_MODEL_CONFIG['gpu_memory_utilization']}")
    print()
    print("Usage: python test_expert_distribution.py <model_name>")
    print("If no model is specified, the default model will be used.")
    print()


def print_results(result: dict):
    """Print results for the tested model."""
    print("\n" + "=" * 80)
    print("EXPERT DISTRIBUTION ANALYSIS RESULTS")
    print("=" * 80)

    print(f"\n MODEL: {result['model_name']}")
    print(f"   Path: {result['model_path']}")
    print("-" * 60)

    print("    INPUT/OUTPUT:")
    print(f"   Input: {result['input_prompt']}")

    for mode_name, mode_data in result['modes'].items():
        print(f"\n  {mode_name.upper()} MODE:")
        print("-" * 40)

        if not mode_data.get('success', False):
            print(f"   ❌ Failed: {mode_data.get('error', 'Unknown error')}")
            continue

        print(f"   Output: {mode_data.get('output_text', 'No output')}")
        print()

        # Print analysis results
        analysis = mode_data.get('analysis', {})
        if analysis:
            if mode_name == "stat":
                print("    STATISTICS:")
                for key, value in analysis.items():
                    if key == "expert_totals":
                        print(f"     tokens per expert: {value}")
                    elif key == "top_experts":
                        print("     Top experts by load:")
                        for idx, load in value:
                            print(f"       Expert {idx}: {load} tokens")
                    elif key == "coefficient_of_variation":
                        print(f"     CV: {value} (lower = more balanced)")
                    elif key == "load_balancedness":
                        print(f"     Load Balancedness: {value} (higher = more balanced)")
                    elif key == "activation_std":
                        print(f"     ACTIVATION VARIATION: ±{value} experts (std dev)")
                    elif key == "activation_variation":
                        print(f"     ACTIVATION COEFFICIENT OF VARIATION: {value:.1%}")
                    elif key == "prefill":
                        print(f"       PREFILL:")
                        print(f"       Forward passes: {value['forward_passes']}")
                        print(f"       Avg experts activated: {value['avg_experts_activated']}")
                    elif key == "decode":
                        print(f"       DECODE PHASE:")
                        print(f"       Forward passes: {value['forward_passes']}")
                        print(f"       Avg experts activated: {value['avg_experts_activated']}")
                        if value['forward_passes'] <= 5:  # Show individual if not too many
                            decode_str = ", ".join([f"F{i+2}:{a}" for i, a in enumerate(value['per_forward_activations'])])
                            print(f"       Per-forward: {decode_str}")
                    else:
                        print(f"     {key.replace('_', ' ').title()}: {value}")
            elif mode_name == "per_token":
                print("    PER-TOKEN ANALYSIS:")
                for key, value in analysis.items():
                    if key == "sample_routing":
                        print("     Sample routing (first 3 layers):")
                        for layer, experts in value.items():
                            print(f"       Layer {layer}: Experts {experts}")
                    else:
                        print(f"     {key.replace('_', ' ').title()}: {value}")
            elif mode_name == "per_pass":
                print("    PER-PASS ANALYSIS:")
                for key, value in analysis.items():
                    if key == "per_pass_activations":
                        print(f"     Per-pass activations: {value}")
                    elif key == "prefill_activation":
                        print(f"     PREFILL: Avg experts activated: {value}")
                    elif key == "decode_avg_activation":
                        print(f"     DECODE: Avg experts activated: {value} ({analysis.get('decode_passes', 0)} passes)")
                    elif key == "activation_variation":
                        print(f"     ACTIVATION VARIATION: ±{value} experts (std dev)")
                    else:
                        print(f"     {key.replace('_', ' ').title()}: {value}")

        # Print raw data (first 10 rows with explanations)
        raw_data = mode_data.get('raw_data')
        if raw_data:
            print(f"\n    RAW DATA ({mode_name.upper()}) - First rows:")
            if mode_name == "stat":
                print("     STAT shows token counts per expert per layer")
                print("     Format: [pass][layer][expert_counts]")
                if raw_data.get('aggregated_expert_counts'):
                    counts = raw_data['aggregated_expert_counts']
                    print(f"     Sample (first pass, 3 layers, 8 experts each):")
                    for pass_idx, pass_data in enumerate(counts[:1]):
                        if isinstance(pass_data, list) and len(pass_data) > 0:
                            for layer_idx, layer_data in enumerate(pass_data[:3]):
                                if isinstance(layer_data, list):
                                    expert_str = ', '.join(map(str, layer_data[:8]))
                                    if len(layer_data) > 8:
                                        expert_str += "..."
                                    print(f"       Layer {layer_idx}: [{expert_str}]")
                            if len(pass_data) > 3:
                                print(f"       ... +{len(pass_data) - 3} more layers")
                print(f"     Model: {raw_data.get('num_layers')} layers × {raw_data.get('num_experts')} experts")
            elif mode_name == "per_token":
                print("     PER_TOKEN shows routing path of each token through layers")
                print("     Each token visits different experts at each layer (top-k routing)")
                print("     Format: Layer X: [expert_ids] means token used these experts at that layer")
                if isinstance(raw_data, list) and len(raw_data) > 0:
                    print("     Sample routing (first 2 tokens, first 4 layers each):")
                    for token_idx in range(min(2, len(raw_data))):  # First 2 tokens
                        print(f"       Token {token_idx}:")
                        record = raw_data[token_idx]
                        if isinstance(record, dict):
                            if 'topk_ids_of_layer' in record and record['topk_ids_of_layer']:
                                layer_data = record['topk_ids_of_layer']
                                for layer_idx, expert_ids in list(layer_data.items())[:4]:  # First 4 layers
                                    valid_experts = [e for e in expert_ids if e != -1]
                                    if valid_experts:
                                        print(f"         Layer {layer_idx}: [{', '.join(map(str, valid_experts))}]")
                            elif 'topk_ids' in record:
                                topk_ids = record['topk_ids']
                                if isinstance(topk_ids, list) and len(topk_ids) > 0:
                                    for layer_idx, layer_experts in enumerate(topk_ids[:4]):  # First 4 layers
                                        # Flatten nested lists and filter out -1 values
                                        if isinstance(layer_experts, list):
                                            if not layer_experts:
                                                valid_experts = []
                                            elif isinstance(layer_experts[0], list):  # Nested structure
                                                valid_experts = []
                                                for sublist in layer_experts:
                                                    if isinstance(sublist, list):
                                                        valid_experts.extend([e for e in sublist if e != -1])
                                                    elif sublist != -1:
                                                        valid_experts.append(sublist)
                                                valid_experts = list(set(valid_experts))  # Remove duplicates
                                                valid_experts.sort()  # Sort for consistency
                                            else:  # Flat list
                                                valid_experts = [e for e in layer_experts if e != -1]
                                        else:
                                            valid_experts = [layer_experts] if layer_experts != -1 else []

                                        if valid_experts:
                                            print(f"         Layer {layer_idx}: [{', '.join(map(str, valid_experts))}]")
                        print()  # Add blank line between tokens
            elif mode_name == "per_pass":
                print("     PER_PASS shows expert activation metrics per forward pass")
                print("     Each record represents one complete forward pass through the model")
                print("     Format: Pass ID, avg experts activated per layer, total experts, utilization")
                if isinstance(raw_data, list) and len(raw_data) > 0:
                    print("     Sample forward passes:")
                    for record in raw_data:
                        if isinstance(record, dict):
                            pass_id = record.get('forward_pass_id', 'N/A')
                            avg_activation = record.get('avg_activated_per_layer', 'N/A')
                            total_activation = record.get('total_activated_experts', 'N/A')
                            utilization = record.get('expert_utilization', 'N/A')
                            print(f"       Pass {pass_id}: Avg={avg_activation}, Total={total_activation}, Utilization={utilization:.1%}")

    print("\n" + "=" * 60)


def main():
    """Main demo function."""
    print("vLLM Expert Distribution Test")
    print("=" * 40)

    # Print usage info
    print_usage_info()

    # Get model configuration
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        model_config = {
            "model_path": model_name,
            "tp_size": 4,
            "gpu_memory_utilization": 0.4,
        }
        print(f"Testing model: {model_name}")
    else:
        model_config = get_default_model_config()
        print(f"Using default model: {model_config['model_path']}")

    print("-" * 50)

    # Test the model
    llm = None
    result = None
    try:
        llm = create_llm_model(model_config['model_path'], model_config['gpu_memory_utilization'])
        result = run_expert_analysis(llm, model_config)
        print(f"✅ Completed testing {model_config['model_path'].split('/')[-1]}")

    except Exception as e:
        print(f"❌ Failed to test {model_config['model_path']}: {e}")
        if "Free memory" in str(e):
            print("💡 Tip: Reduce gpu_memory_utilization or try a different model")

        # Create failed result for reporting
        result = {
            "model_name": model_config['model_path'].split('/')[-1],
            "model_path": model_config['model_path'],
            "input_prompt": "Write a simple Python function to calculate factorial.",
            "modes": {
                "stat": {"success": False, "error": str(e)},
                "per_token": {"success": False, "error": str(e)},
                "per_pass": {"success": False, "error": str(e)}
            }
        }

    finally:
        # Print results
        if result:
            print_results(result)

        # Cleanup
        if llm is not None:
            force_llm_shutdown(llm)


if __name__ == "__main__":
    main()
