def _calculate_deepseek_prefill(n_layers, attention_size_per_token, expert_config, prefill_activation,
                              hardware_specs, num_gpus, precision, ttft, prefill_tp, metrics_data):
    smbu_numerator = ((n_layers * attention_size_per_token + 
                      expert_config['deepseek_sparse_layer_num'] * 
                      (prefill_activation * expert_config['expert_size'] + 
                       expert_config['shared_experts_size_total']) + 
                      expert_config['deepseek_num_dense_layer'] * 
                      expert_config['deepseek_dense_ffn_size'] + metrics_data['kv_size']) * precision / ttft)
    smbu = smbu_numerator / (num_gpus * hardware_specs['peak_bandwidth_tb'])
    
    smfu_numerator = ((n_layers * attention_size_per_token + 
                      expert_config['deepseek_sparse_layer_num'] * 
                      (expert_config['expert_size'] + expert_config['shared_experts_size_total']) + 
                      expert_config['deepseek_num_dense_layer'] * 
                      expert_config['deepseek_dense_ffn_size'] + metrics_data['attention_score']) * 2 * prefill_tp)
    smfu = smfu_numerator / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    
    return smbu, smfu


def _calculate_deepseek_decoding(n_layers, attention_size_per_token, expert_config, activation,
                               metrics_data, hardware_specs, num_gpus, precision, batch_size, decoding_tp, tpot=None):
    if tpot is None:
        tpot = batch_size / decoding_tp

    smbu_numerator = ((n_layers * attention_size_per_token + 
                      expert_config['deepseek_sparse_layer_num'] * 
                      (activation * expert_config['expert_size'] + 
                       expert_config['shared_experts_size_total']) + 
                      expert_config['deepseek_num_dense_layer'] * 
                      expert_config['deepseek_dense_ffn_size'] + 
                      metrics_data['kv_size']) * precision / tpot)
    smbu = smbu_numerator / (num_gpus * hardware_specs['peak_bandwidth_tb'])
    
    smfu_numerator = ((n_layers * attention_size_per_token + 
                      expert_config['deepseek_sparse_layer_num'] * 
                      (expert_config['expert_size'] + expert_config['shared_experts_size_total']) + 
                      expert_config['deepseek_num_dense_layer'] * 
                      expert_config['deepseek_dense_ffn_size'] + 
                      metrics_data['attention_score']) * 2 * decoding_tp)
    smfu = smfu_numerator / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    
    return smbu, smfu


def _calculate_deepseek_attention_size(hf_config, d_model, n_attn_heads):
    """Calculate DeepSeek-specific attention size"""
    q_head_dim = hf_config.qk_rope_head_dim + hf_config.qk_nope_head_dim
    
    base_size = ((d_model * (hf_config.kv_lora_rank + hf_config.qk_rope_head_dim)) +
                (hf_config.kv_lora_rank * n_attn_heads * 
                 (q_head_dim - hf_config.qk_rope_head_dim + hf_config.v_head_dim)) +
                (hf_config.v_head_dim * n_attn_heads * d_model))
    
    if hasattr(hf_config, "q_lora_rank") and hf_config.q_lora_rank:
        q_size = (d_model * hf_config.q_lora_rank + 
                 hf_config.q_lora_rank * n_attn_heads * q_head_dim)
    else:
        q_size = d_model * n_attn_heads * q_head_dim
    
    return (base_size + q_size) / 1e12


def _get_deepseek_expert_config(hf_config, d_model, n_layers):
    """Get DeepSeek-specific expert configuration"""
    if (hasattr(hf_config, "moe_intermediate_size") and 
        hasattr(hf_config, "intermediate_size") and 
        hasattr(hf_config, "first_k_dense_replace")):
        
        deepseek_num_dense_layer = hf_config.first_k_dense_replace
        return {
            'expert_size': hf_config.moe_intermediate_size * 3 * d_model / 1e12,
            'shared_experts_size_total': hf_config.moe_intermediate_size * 3 * d_model * 2 / 1e12,
            'deepseek_dense_ffn_size': hf_config.intermediate_size * 3 * d_model / 1e12,
            'deepseek_sparse_layer_num': n_layers - deepseek_num_dense_layer,
            'deepseek_num_dense_layer': deepseek_num_dense_layer
        }
    return {}
