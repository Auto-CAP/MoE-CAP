from moe_cap.utils.basic_utils import (
    _get_hardware_specs,
    _extract_output_data,
    _calculate_kv_size,
    _calculate_attention_size,
    _calculate_expert_config,
    _process_outputs_continuous,
    _calculate_prefill_metrics,
    _calculate_decoding_metrics
)

def _calculate_continuous_metrics(n_layers, d_model, gpu_raw_type,
                                n_attn_heads, d_head, n_kv_heads, d_ff, hf_config, num_gpus, model_name, 
                                used_dtype, precision, output_data):
    """Calculate metrics for a batch of outputs"""
    # Initialize hardware specs and output lists
    hardware_specs = _get_hardware_specs(used_dtype, gpu_raw_type)
    
    # Calculate model-specific sizes
    per_token_kv_size = _calculate_kv_size(model_name, hf_config, n_layers, d_head, n_kv_heads)
    attention_size_per_token = _calculate_attention_size(model_name, hf_config, d_model, n_attn_heads, d_head, n_kv_heads)
    expert_config = _calculate_expert_config(model_name, hf_config, d_ff, d_model, n_layers)
    
    # Process outputs and calculate metrics
    ttfts = []
    tpots = []
    prefill_tps = []
    decoding_tps = []
    true_kvs = []
    prefill_smbus = []
    prefill_smfus = []
    decoding_smbus = []
    decoding_smfus = []

    for out in output_data:
        # Skip only if expert_activation is explicitly None or negative
        # Allow 0 to support non-MoE models or when expert tracking is not yet implemented
        if out.get('expert_activation') is None or out.get('expert_activation', 0) < 0:
            continue
        
        metrics_data = _process_outputs_continuous(out, per_token_kv_size, attention_size_per_token, 
                                    model_name, hf_config, n_layers, n_attn_heads, d_head)

        true_kvs.append(metrics_data['true_kv_size'])

        # Calculate throughput metrics
        if out['forward_mode'] == 'prefill':
            # Use expert_activation if available, otherwise default to 0
            prefill_activation = out.get('expert_activation', 0)
            ttft = out['latency']
            prefill_tp = out['seq_lens_sum'] / ttft
            ttfts.append(ttft)
            prefill_tps.append(prefill_tp)
            prefill_smbu, prefill_smfu = _calculate_prefill_metrics(model_name=model_name, n_layers=n_layers, attention_size_per_token=attention_size_per_token,
                                               expert_config=expert_config, hardware_specs=hardware_specs, num_gpus=num_gpus, precision=precision, ttft=ttft, 
                                               prefill_tp=prefill_tp, prefill_activation=prefill_activation, metrics_data=metrics_data)
            prefill_smbus.append(prefill_smbu)
            prefill_smfus.append(prefill_smfu)

        else:
            # Use expert_activation if available, otherwise default to 0
            decoding_activation = out.get('expert_activation', 0)
            tpot = out['latency']
            batch_size = out['batch_size']
            decoding_tp = batch_size / tpot
            tpots.append(tpot)
            decoding_tps.append(decoding_tp)

            decoding_smbu, decoding_smfu = _calculate_decoding_metrics(model_name=model_name, n_layers=n_layers, attention_size_per_token=attention_size_per_token,
                                               expert_config=expert_config, decode_steps_activation=decoding_activation, metrics_data=metrics_data,
                                               hardware_specs=hardware_specs, num_gpus=num_gpus, precision=precision, batch_size=batch_size, decoding_tp=decoding_tp, tpot=tpot)
            decoding_smbus.append(decoding_smbu)
            decoding_smfus.append(decoding_smfu)

    
    # Aggregate metrics
    prefill_smbu = sum(prefill_smbus) / len(prefill_smbus) if prefill_smbus else 0
    prefill_smfu = sum(prefill_smfus) / len(prefill_smfus) if prefill_smfus else 0
    decoding_smbu = sum(decoding_smbus) / len(decoding_smbus) if decoding_smbus else 0
    decoding_smfu = sum(decoding_smfus) / len(decoding_smfus) if decoding_smfus else 0
    decoding_tp = sum(decoding_tps) / len(decoding_tps) if decoding_tps else 0
    tpot = sum(tpots) / len(tpots) if tpots else 0
    ttft = sum(ttfts) / len(ttfts) if ttfts else 0
    prefill_tp = sum(prefill_tps) / len(prefill_tps) if prefill_tps else 0
    kv_size = sum(true_kvs) / len(true_kvs) if true_kvs else 0


    return {
        'prefill_smbu': prefill_smbu,
        'prefill_smfu': prefill_smfu,
        'decoding_smbu': decoding_smbu,
        'decoding_smfu': decoding_smfu,
        'kv_size': kv_size,
        'decoding_throughput': decoding_tp,
        'prefill_tp': prefill_tp,
        'ttft': ttft,
        'tpot': tpot,
        'gpu_raw_type': gpu_raw_type
    }
