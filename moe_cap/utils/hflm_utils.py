from moe_cap.utils.basic_utils import (
    _get_hardware_specs,
    _calculate_kv_size,
    _calculate_attention_size,
    _calculate_expert_config,
    _calculate_decoding_metrics,
    _calculate_throughput_metrics
)

def _calculate_batch_metrics_hflm(output_len, context_prefill_size, decoding_tp, n_layers, d_model, n_attn_heads, d_head, n_kv_heads, n_experts_per_tok, d_ff, avg_activated_experts, hf_config, num_gpus, model_name, used_dtype, batch_size, precision):
    """A compact batch-metrics wrapper kept for compatibility with callers."""
    hardware_specs = _get_hardware_specs(used_dtype)

    per_token_kv_size = _calculate_kv_size(model_name, hf_config, n_layers, d_head, n_kv_heads)
    attention_size_per_token = _calculate_attention_size(model_name, hf_config, d_model, n_attn_heads, d_head, n_kv_heads)

    # Compute basic metrics used by decoding/prefill calculators
    metrics_data = {
        'kv_size': (context_prefill_size * per_token_kv_size + (output_len - 1) * per_token_kv_size / 2) / 1e12,
        'true_kv_size': (context_prefill_size * per_token_kv_size + output_len * per_token_kv_size) / 1e9,
        'attention_score': _calculate_throughput_metrics(batch_size, [context_prefill_size], 1)[0],
    }

    expert_config = _calculate_expert_config(model_name, hf_config, d_ff, d_model, n_layers)

    # Delegate to decoding calculator for final numbers (use default decoding activation placeholder)
    smbu, smfu = _calculate_decoding_metrics(model_name, n_layers, attention_size_per_token, expert_config, avg_activated_experts, metrics_data, hardware_specs, num_gpus, precision, batch_size, decoding_tp)

    return {
        'smbu': smbu,
        'smfu': smfu,
        'kv_size': metrics_data['true_kv_size'],
        'decoding_throughput': decoding_tp,
        'ttft': 0,
    }
