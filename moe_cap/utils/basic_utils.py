from typing import List

# Import hardware and model-specific helpers from their own modules
from .hardware_utils import get_gpu_details, get_peak_bw, get_peak_flops
from .qwen_utils import (
    _get_qwen_expert_config,
    _get_qwen3_expert_config,
    _calculate_qwen_prefill,
    _calculate_qwen3_prefill,
    _calculate_qwen_decoding,
    _calculate_qwen3_decoding,
)
from .deepseek_utils import (
    _get_deepseek_expert_config,
    _calculate_deepseek_prefill,
    _calculate_deepseek_decoding,
    _calculate_deepseek_attention_size,
)


def _extract_output_data(outputs):
    """Extract relevant data from outputs (expects list of dict-like items)."""
    prefill_lengths = []
    output_lengths = []
    max_duration = 0.0

    for x in outputs:
        # Support both object and dict styles used elsewhere in repo
        meta = x.get('meta_info') if isinstance(x, dict) else getattr(x, 'meta_info', None)
        if meta is None:
            # try attributes used by some callers
            prompt_tokens = getattr(x, 'prompt_token_ids', None)
            outputs_attr = getattr(x, 'outputs', None)
            completion_tokens = len(outputs_attr[0].token_ids) if outputs_attr else 0
            prefill_lengths.append(len(prompt_tokens) if prompt_tokens is not None else 0)
            output_lengths.append(completion_tokens)
            max_duration = max(max_duration, getattr(x, 'e2e_latency', 0.0))
            continue

        output_lengths.append(meta.get('completion_tokens', 0))
        prefill_lengths.append(meta.get('prompt_tokens', 0))
        max_duration = max(max_duration, meta.get('e2e_latency', 0.0))

    return {
        'prefill_lengths': prefill_lengths,
        'output_lengths': output_lengths,
        'max_duration': max_duration,
    }


def _calculate_kv_size(model_name, hf_config, n_layers, d_head, n_kv_heads):
    """Calculate per-token KV size based on model type."""
    if "DeepSeek" in model_name and hasattr(hf_config, "kv_lora_rank") and hasattr(hf_config, "qk_rope_head_dim"):
        return n_layers * (hf_config.kv_lora_rank + hf_config.qk_rope_head_dim)
    return 2 * n_layers * d_head * n_kv_heads


def _calculate_attention_size(model_name, hf_config, d_model, n_attn_heads, d_head, n_kv_heads):
    """Return attention size per token (in TB units where appropriate).

    For DeepSeek we delegate to its specialized function.
    """
    if (
        "DeepSeek" in model_name
        and hasattr(hf_config, "qk_rope_head_dim")
        and hasattr(hf_config, "qk_nope_head_dim")
        and hasattr(hf_config, "v_head_dim")
        and hasattr(hf_config, "kv_lora_rank")
    ):
        return _calculate_deepseek_attention_size(hf_config, d_model, n_attn_heads)

    return (d_model * (n_attn_heads * d_head + n_kv_heads * d_head * 2) + n_attn_heads * d_head * d_model) / 1e12


def _calculate_throughput_metrics(batch_size, prefill_lengths, max_duration):
    total_prefill = sum(prefill_lengths)
    prefill_tp = total_prefill / max_duration if max_duration > 0 else 0
    ttft = max_duration / batch_size if batch_size > 0 else 0
    return ttft, prefill_tp


def _get_hardware_specs(used_dtype, gpu_type=None):
    return {
        "peak_bandwidth_tb": get_peak_bw(gpu_type) / 1e12,
        "peak_flops_tf": get_peak_flops(gpu_type, precision=used_dtype) / 1e12,
    }


def only_prefill_metrics(outputs, batch_size):
    output_data = _extract_output_data(outputs)
    ttft, prefill_tp = _calculate_throughput_metrics(batch_size, output_data['prefill_lengths'], output_data['max_duration'])
    return {
        'prefill_smbu': 0,
        'prefill_smfu': 0,
        'decoding_smbu': 0,
        'decoding_smfu': 0,
        'kv_size': 0,
        'decoding_throughput': 0,
        'prefill_tp': prefill_tp,
        'ttft': ttft,
    }


def _calculate_prefill_metrics(model_name, n_layers, attention_size_per_token, expert_config, prefill_activation, hardware_specs, num_gpus, precision, ttft, prefill_tp, metrics_data):
    """Dispatch to model-specific prefill calculators when available."""
    if "Qwen" in model_name and "Qwen3" not in model_name:
        return _calculate_qwen_prefill(n_layers, attention_size_per_token, expert_config, prefill_activation, hardware_specs, num_gpus, precision, ttft, prefill_tp, metrics_data)
    if "Qwen3" in model_name:
        return _calculate_qwen3_prefill(n_layers, attention_size_per_token, expert_config, prefill_activation, hardware_specs, num_gpus, precision, ttft, prefill_tp, metrics_data)
    if "DeepSeek" in model_name:
        return _calculate_deepseek_prefill(n_layers, attention_size_per_token, expert_config, prefill_activation, hardware_specs, num_gpus, precision, ttft, prefill_tp, metrics_data)

    return _calculate_default_prefill(n_layers, attention_size_per_token, expert_config, prefill_activation, hardware_specs, num_gpus, precision, ttft, prefill_tp, metrics_data)


def _calculate_decoding_metrics(model_name, n_layers, attention_size_per_token, expert_config, decode_steps_activation, metrics_data, hardware_specs, num_gpus, precision, batch_size, decoding_tp, tpot=None):
    if "Qwen" in model_name and "Qwen3" not in model_name:
        return _calculate_qwen_decoding(n_layers, attention_size_per_token, expert_config, decode_steps_activation, metrics_data, hardware_specs, num_gpus, precision, batch_size, decoding_tp, tpot)
    if "Qwen3" in model_name:
        return _calculate_qwen3_decoding(n_layers, attention_size_per_token, expert_config, decode_steps_activation, metrics_data, hardware_specs, num_gpus, precision, batch_size, decoding_tp, tpot)
    if "DeepSeek" in model_name:
        return _calculate_deepseek_decoding(n_layers, attention_size_per_token, expert_config, decode_steps_activation, metrics_data, hardware_specs, num_gpus, precision, batch_size, decoding_tp, tpot)

    return _calculate_default_decoding(n_layers, attention_size_per_token, expert_config, decode_steps_activation, metrics_data, hardware_specs, num_gpus, precision, batch_size, decoding_tp, tpot)


# Default implementations kept here for generic models
def _calculate_default_prefill(n_layers, attention_size_per_token, expert_config, prefill_activation, hardware_specs, num_gpus, precision, ttft, prefill_tp, metrics_data):
    smbu_numerator = (n_layers * (prefill_activation * expert_config['expert_size'] + attention_size_per_token) + metrics_data['kv_size']) * precision / (ttft or 1)
    smbu = smbu_numerator / (num_gpus * hardware_specs['peak_bandwidth_tb'])

    smfu_numerator = (n_layers * (attention_size_per_token + expert_config['expert_size']) + metrics_data['attention_score']) * 2 * prefill_tp
    smfu = smfu_numerator / (num_gpus * hardware_specs['peak_flops_tf'] / 2)

    return smbu, smfu


def _calculate_default_decoding(n_layers, attention_size_per_token, expert_config, activation, metrics_data, hardware_specs, num_gpus, precision, batch_size, decoding_tp, tpot=None):
    if tpot is None:
        tpot = batch_size / decoding_tp if decoding_tp and batch_size else 1

    smbu_numerator = ((n_layers * (activation * expert_config['expert_size'] + attention_size_per_token) + metrics_data['kv_size']) * precision / tpot)
    smbu = smbu_numerator / (num_gpus * hardware_specs['peak_bandwidth_tb'])

    smfu_numerator = ((n_layers * (attention_size_per_token + expert_config['expert_size']) + metrics_data['attention_score']) * 2 * decoding_tp)
    smfu = smfu_numerator / (num_gpus * hardware_specs['peak_flops_tf'] / 2)

    return smbu, smfu


# Exported small helper used by other modules
def _calculate_expert_config(model_name, hf_config, d_ff, d_model, n_layers):
    """Calculate expert configuration based on model type."""
    config = {
        'expert_size': d_ff * 3 * d_model / 1e12,
        'shared_experts_size_total': 0,
        'deepseek_dense_ffn_size': 0,
        'deepseek_sparse_layer_num': 0,
        'deepseek_num_dense_layer': 0
    }
    
    if "Qwen" in model_name and not "Qwen3" in model_name:
        config.update(_get_qwen_expert_config(hf_config, d_model))
    elif "Qwen3" in model_name:
        config.update(_get_qwen3_expert_config(hf_config, d_model))
    elif "DeepSeek" in model_name:
        config.update(_get_deepseek_expert_config(hf_config, d_model, n_layers))
    
    return config


def _calculate_attention_score(model_name, hf_config, prefill_len, output_len, n_layers, n_attn_heads, d_head):
    """Calculate attention score for a single output."""
    if (
        "DeepSeek" in model_name
        and hasattr(hf_config, "qk_rope_head_dim")
        and hasattr(hf_config, "qk_nope_head_dim")
        and hasattr(hf_config, "v_head_dim")
    ):
        q_head_dim = hf_config.qk_rope_head_dim + hf_config.qk_nope_head_dim
        k_size = n_layers * n_attn_heads * q_head_dim
        v_size = n_layers * n_attn_heads * hf_config.v_head_dim
        
        score = (prefill_len * k_size + (output_len - 1) * k_size / 2 +
                prefill_len * v_size + (output_len - 1) * v_size / 2)
    else:
        kv_size = n_layers * n_attn_heads * d_head
        score = (prefill_len * kv_size + (output_len - 1) * kv_size / 2) * 2
    
    return score / 1e12


def _process_outputs(output_data, per_token_kv_size, attention_size_per_token, model_name, hf_config, n_layers, n_attn_heads, d_head):
    """Process outputs to calculate KV sizes and attention scores."""
    kvs = []
    true_kvs = []
    attn_scores = []
    
    for prefill_len, output_len in zip(output_data['prefill_lengths'], output_data['output_lengths']):
        # Calculate attention score
        attn_score = _calculate_attention_score(model_name, hf_config, prefill_len, output_len,
                                              n_layers, n_attn_heads, d_head)
        attn_scores.append(attn_score)
        
        # Calculate KV sizes
        kv_size = (prefill_len * per_token_kv_size + (output_len - 1) * per_token_kv_size / 2) / 1e12
        true_kv = (prefill_len * per_token_kv_size + output_len * per_token_kv_size) / 1e9
        
        kvs.append(kv_size)
        true_kvs.append(true_kv)
    
    return {
        'kv_size': sum(kvs),
        'true_kv_size': sum(true_kvs) * 1e3,
        'attention_score': sum(attn_scores)
    }


def _process_outputs_continuous(out, per_token_kv_size, attention_size_per_token, model_name, hf_config, n_layers, n_attn_heads, d_head):
    """Process outputs to calculate KV sizes and attention scores for continuous batching."""
    kvs = []
    true_kvs = []
    attn_scores = []
    
    # Calculate attention score
    ctx_len = out['seq_lens_sum']
    attn_score = _calculate_attention_score(model_name, hf_config, ctx_len, 1,
                                          n_layers, n_attn_heads, d_head)
    attn_scores.append(attn_score)
    
    # Calculate KV sizes
    kv_size = (ctx_len * per_token_kv_size) / 1e12
    true_kv = (ctx_len * per_token_kv_size + 1 * per_token_kv_size) / 1e9
    kvs.append(kv_size)
    true_kvs.append(true_kv)
    
    return {
        'kv_size': sum(kvs) / len(kvs) if kvs else 0,
        'true_kv_size': sum(true_kvs) * 1e3,
        'attention_score': sum(attn_scores) / len(attn_scores)
    }
