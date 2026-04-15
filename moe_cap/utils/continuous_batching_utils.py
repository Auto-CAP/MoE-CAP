from moe_cap.utils.basic_utils import (
    _get_hardware_specs,
    _extract_output_data,
    _calculate_kv_size,
    _calculate_attention_size,
    _calculate_expert_config,
    _process_outputs_continuous,
    _calculate_prefill_metrics,
    _calculate_decoding_metrics,
)


def _calculate_continuous_metrics(
    n_layers,
    d_model,
    gpu_raw_type,
    n_attn_heads,
    d_head,
    n_kv_heads,
    d_ff,
    hf_config,
    num_gpus,
    model_name,
    used_dtype,
    precision,
    output_data,
):
    """Calculate metrics for a batch of outputs"""
    # Initialize hardware specs and output lists
    hardware_specs = _get_hardware_specs(used_dtype, gpu_raw_type)

    # Calculate model-specific sizes
    per_token_kv_size = _calculate_kv_size(
        model_name, hf_config, n_layers, d_head, n_kv_heads
    )
    attention_size_per_token = _calculate_attention_size(
        model_name, hf_config, d_model, n_attn_heads, d_head, n_kv_heads
    )
    expert_config = _calculate_expert_config(
        model_name, hf_config, d_ff, d_model, n_layers
    )

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

    has_per_req_info = any(
        out.get("forward_mode") == "prefill" and out.get("per_req_info")
        for out in output_data
    )

    if has_per_req_info:
        req_prefill_accum = {}

        for out in output_data:
            if (
                out.get("expert_activation") is None
                or out.get("expert_activation", 0) < 0
            ):
                continue

            # Skip warmup/health-check probes (tiny prefills from server startup)
            if out["forward_mode"] == "prefill" and out.get("seq_lens_sum", 0) <= 10:
                continue

            if out["forward_mode"] != "prefill":
                metrics_data = _process_outputs_continuous(
                    out,
                    per_token_kv_size,
                    attention_size_per_token,
                    model_name,
                    hf_config,
                    n_layers,
                    n_attn_heads,
                    d_head,
                )
                true_kvs.append(metrics_data["true_kv_size"])

                decoding_activation = out.get("expert_activation", 0)
                tpot = out["latency"]
                batch_size = out["batch_size"]
                decoding_tp = batch_size / tpot
                tpots.append(tpot)
                decoding_tps.append(decoding_tp)

                decoding_smbu, decoding_smfu = _calculate_decoding_metrics(
                    model_name=model_name,
                    n_layers=n_layers,
                    attention_size_per_token=attention_size_per_token,
                    expert_config=expert_config,
                    decode_steps_activation=decoding_activation,
                    metrics_data=metrics_data,
                    hardware_specs=hardware_specs,
                    num_gpus=num_gpus,
                    precision=precision,
                    batch_size=batch_size,
                    decoding_tp=decoding_tp,
                    tpot=tpot,
                )
                decoding_smbus.append(decoding_smbu)
                decoding_smfus.append(decoding_smfu)
                continue

            per_req_info = out.get("per_req_info", [])
            if not per_req_info:
                prefill_activation = out.get("expert_activation", 0)
                ttft = out["latency"]
                prefill_tp = out["seq_lens_sum"] / ttft
                ttfts.append(ttft)
                prefill_tps.append(prefill_tp)

                metrics_data = _process_outputs_continuous(
                    out,
                    per_token_kv_size,
                    attention_size_per_token,
                    model_name,
                    hf_config,
                    n_layers,
                    n_attn_heads,
                    d_head,
                )
                true_kvs.append(metrics_data["true_kv_size"])

                prefill_smbu, prefill_smfu = _calculate_prefill_metrics(
                    model_name=model_name,
                    n_layers=n_layers,
                    attention_size_per_token=attention_size_per_token,
                    expert_config=expert_config,
                    hardware_specs=hardware_specs,
                    num_gpus=num_gpus,
                    precision=precision,
                    ttft=ttft,
                    prefill_tp=prefill_tp,
                    prefill_activation=prefill_activation,
                    metrics_data=metrics_data,
                )
                prefill_smbus.append(prefill_smbu)
                prefill_smfus.append(prefill_smfu)
                continue

            record_latency = out["latency"]
            for req_info in per_req_info:
                req_idx = req_info.get("req_id", req_info.get("req_pool_idx"))
                if req_idx not in req_prefill_accum:
                    req_prefill_accum[req_idx] = {
                        "latency_sum": 0.0,
                        "total_tokens": 0,
                        "seq_lens_sum": 0,
                        "last_expert_activation": 0,
                        "last_record": None,
                        "completed": False,
                    }

                accum = req_prefill_accum[req_idx]
                accum["latency_sum"] += record_latency
                accum["seq_lens_sum"] += req_info.get("extend_len", 0)
                if "total_len" in req_info:
                    accum["total_tokens"] = req_info["total_len"]
                accum["last_expert_activation"] = out.get("expert_activation", 0)
                accum["last_record"] = out
                if req_info.get("is_last_chunk"):
                    accum["completed"] = True

        for accum in req_prefill_accum.values():
            if not accum["completed"] or accum["last_record"] is None:
                continue

            ttft = accum["latency_sum"]
            total_tokens = (
                accum["total_tokens"]
                if accum["total_tokens"] > 0
                else accum["seq_lens_sum"]
            )
            prefill_tp = (total_tokens / ttft) if ttft > 0 else 0
            prefill_activation = accum["last_expert_activation"]

            ttfts.append(ttft)
            prefill_tps.append(prefill_tp)

            aggregated_out = dict(accum["last_record"])
            aggregated_out["seq_lens_sum"] = total_tokens
            metrics_data = _process_outputs_continuous(
                aggregated_out,
                per_token_kv_size,
                attention_size_per_token,
                model_name,
                hf_config,
                n_layers,
                n_attn_heads,
                d_head,
            )
            true_kvs.append(metrics_data["true_kv_size"])

            prefill_smbu, prefill_smfu = _calculate_prefill_metrics(
                model_name=model_name,
                n_layers=n_layers,
                attention_size_per_token=attention_size_per_token,
                expert_config=expert_config,
                hardware_specs=hardware_specs,
                num_gpus=num_gpus,
                precision=precision,
                ttft=ttft,
                prefill_tp=prefill_tp,
                prefill_activation=prefill_activation,
                metrics_data=metrics_data,
            )
            prefill_smbus.append(prefill_smbu)
            prefill_smfus.append(prefill_smfu)
    else:
        for out in output_data:
            if (
                out.get("expert_activation") is None
                or out.get("expert_activation", 0) < 0
            ):
                continue

            if out["forward_mode"] == "prefill" and out.get("seq_lens_sum", 0) <= 10:
                continue

            metrics_data = _process_outputs_continuous(
                out,
                per_token_kv_size,
                attention_size_per_token,
                model_name,
                hf_config,
                n_layers,
                n_attn_heads,
                d_head,
            )

            true_kvs.append(metrics_data["true_kv_size"])

            # Calculate throughput metrics
            if out["forward_mode"] == "prefill":
                # Use expert_activation if available, otherwise default to 0
                prefill_activation = out.get("expert_activation", 0)
                ttft = out["latency"]
                prefill_tp = out["seq_lens_sum"] / ttft
                ttfts.append(ttft)
                prefill_tps.append(prefill_tp)
                prefill_smbu, prefill_smfu = _calculate_prefill_metrics(
                    model_name=model_name,
                    n_layers=n_layers,
                    attention_size_per_token=attention_size_per_token,
                    expert_config=expert_config,
                    hardware_specs=hardware_specs,
                    num_gpus=num_gpus,
                    precision=precision,
                    ttft=ttft,
                    prefill_tp=prefill_tp,
                    prefill_activation=prefill_activation,
                    metrics_data=metrics_data,
                )
                prefill_smbus.append(prefill_smbu)
                prefill_smfus.append(prefill_smfu)

            else:
                # Use expert_activation if available, otherwise default to 0
                decoding_activation = out.get("expert_activation", 0)
                tpot = out["latency"]
                batch_size = out["batch_size"]
                decoding_tp = batch_size / tpot
                tpots.append(tpot)
                decoding_tps.append(decoding_tp)

                decoding_smbu, decoding_smfu = _calculate_decoding_metrics(
                    model_name=model_name,
                    n_layers=n_layers,
                    attention_size_per_token=attention_size_per_token,
                    expert_config=expert_config,
                    decode_steps_activation=decoding_activation,
                    metrics_data=metrics_data,
                    hardware_specs=hardware_specs,
                    num_gpus=num_gpus,
                    precision=precision,
                    batch_size=batch_size,
                    decoding_tp=decoding_tp,
                    tpot=tpot,
                )
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
        "prefill_smbu": prefill_smbu,
        "prefill_smfu": prefill_smfu,
        "decoding_smbu": decoding_smbu,
        "decoding_smfu": decoding_smfu,
        "kv_size": kv_size,
        "decoding_throughput": decoding_tp,
        "prefill_tp": prefill_tp,
        "ttft": ttft,
        "tpot": tpot,
        "gpu_raw_type": gpu_raw_type,
    }
