def _get_qwen_expert_config(hf_config, d_model):
    """Get Qwen-specific expert configuration"""
    if (hasattr(hf_config, "moe_intermediate_size") and 
        hasattr(hf_config, "shared_expert_intermediate_size")):
        
        return {
            'expert_size': hf_config.moe_intermediate_size * 3 * d_model / 1e12,
            'shared_experts_size_total': hf_config.shared_expert_intermediate_size * 3 * d_model / 1e12
        }
    return {}


def _get_qwen3_expert_config(hf_config, d_model):
    """Get Qwen3-specific expert configuration"""
    if hasattr(hf_config, "moe_intermediate_size"):
        # Calculate number of MoE layers vs dense layers
        n_layers = hf_config.num_hidden_layers
        mlp_only_layers = set(getattr(hf_config, "mlp_only_layers", []))
        decoder_sparse_step = getattr(hf_config, "decoder_sparse_step", 1)
        
        # Count MoE layers: layers that are not in mlp_only_layers and satisfy the sparse_step condition
        num_moe_layers = 0
        for layer_idx in range(n_layers):
            if (layer_idx not in mlp_only_layers) and ((layer_idx + 1) % decoder_sparse_step == 0):
                num_moe_layers += 1
        
        num_dense_layers = n_layers - num_moe_layers
        
        return {
            'expert_size': hf_config.moe_intermediate_size * 3 * d_model / 1e12,
            'dense_ffn_size': hf_config.intermediate_size * 3 * d_model / 1e12,
            'num_moe_layers': num_moe_layers,
            'num_dense_layers': num_dense_layers
        }
    return {}


def _calculate_qwen_prefill(n_layers, attention_size_per_token, expert_config, prefill_activation,
                          hardware_specs, num_gpus, precision, ttft, prefill_tp, metrics_data):
    smbu_numerator = (n_layers * (prefill_activation * expert_config['expert_size'] + 
                                expert_config['shared_experts_size_total'] + 
                                attention_size_per_token) + metrics_data['kv_size']) * precision / ttft
    smbu = smbu_numerator / (num_gpus * hardware_specs['peak_bandwidth_tb'])
    
    smfu_numerator = (n_layers * (attention_size_per_token + expert_config['expert_size'] + 
                                expert_config['shared_experts_size_total']) + metrics_data['attention_score']) * 2 * prefill_tp
    smfu = smfu_numerator / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    
    return smbu, smfu


def _calculate_qwen3_prefill(n_layers, attention_size_per_token, expert_config, prefill_activation,
                           hardware_specs, num_gpus, precision, ttft, prefill_tp, metrics_data):
    # Qwen3 has both MoE layers and dense FFN layers
    num_moe_layers = expert_config.get('num_moe_layers', 0)
    num_dense_layers = expert_config.get('num_dense_layers', n_layers)
    
    # Bandwidth: MoE layers use activated experts, dense layers use full FFN
    moe_bandwidth = num_moe_layers * (prefill_activation * expert_config['expert_size'])
    dense_bandwidth = num_dense_layers * expert_config.get('dense_ffn_size', 0)
    attention_bandwidth = n_layers * attention_size_per_token
    
    smbu_numerator = (moe_bandwidth + dense_bandwidth + attention_bandwidth + 
                     metrics_data['kv_size']) * precision / ttft
    smbu = smbu_numerator / (num_gpus * hardware_specs['peak_bandwidth_tb'])
    
    # FLOPS: Similar calculation for compute
    moe_flops = num_moe_layers * expert_config['expert_size']
    dense_flops = num_dense_layers * expert_config.get('dense_ffn_size', 0)
    
    smfu_numerator = (moe_flops + dense_flops + attention_bandwidth + 
                     metrics_data['attention_score']) * 2 * prefill_tp
    smfu = smfu_numerator / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    
    return smbu, smfu


def _calculate_qwen_decoding(n_layers, attention_size_per_token, expert_config, activation,
                           metrics_data, hardware_specs, num_gpus, precision, batch_size=None, decoding_tp=None, tpot=None):

    if tpot is None:
        assert decoding_tp is not None and batch_size is not None, "Either tpot or decoding_tp and batch_size must be provided."
        tpot = batch_size / decoding_tp
    smbu_numerator = ((n_layers * (activation * expert_config['expert_size'] + 
                                 expert_config['shared_experts_size_total'] + 
                                 attention_size_per_token) + 
                      metrics_data['kv_size']) * precision / tpot)
    smbu = smbu_numerator / (num_gpus * hardware_specs['peak_bandwidth_tb'])
    
    smfu_numerator = ((n_layers * (attention_size_per_token + expert_config['expert_size'] + 
                                 expert_config['shared_experts_size_total']) + 
                      metrics_data['attention_score']) * 2 * decoding_tp)
    smfu = smfu_numerator / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    
    return smbu, smfu


def _calculate_qwen3_decoding(n_layers, attention_size_per_token, expert_config, activation,
                            metrics_data, hardware_specs, num_gpus, precision, batch_size, decoding_tp, tpot=None):
    if tpot is None:
        tpot = batch_size / decoding_tp
    
    # Qwen3 has both MoE layers and dense FFN layers
    num_moe_layers = expert_config.get('num_moe_layers', 0)
    num_dense_layers = expert_config.get('num_dense_layers', n_layers)
    
    # Bandwidth: MoE layers use activated experts, dense layers use full FFN
    moe_bandwidth = num_moe_layers * activation * expert_config['expert_size']
    dense_bandwidth = num_dense_layers * expert_config.get('dense_ffn_size', 0)
    attention_bandwidth = n_layers * attention_size_per_token

    smbu_numerator = (moe_bandwidth + dense_bandwidth + attention_bandwidth + 
                     metrics_data['kv_size']) * precision / tpot
    smbu = smbu_numerator / (num_gpus * hardware_specs['peak_bandwidth_tb'])
    
    # FLOPS: Similar calculation for compute
    moe_flops = num_moe_layers * expert_config['expert_size']
    dense_flops = num_dense_layers * expert_config.get('dense_ffn_size', 0)
    
    smfu_numerator = (moe_flops + dense_flops + attention_bandwidth + 
                     metrics_data['attention_score']) * 2 * decoding_tp
    smfu = smfu_numerator / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    
    return smbu, smfu
