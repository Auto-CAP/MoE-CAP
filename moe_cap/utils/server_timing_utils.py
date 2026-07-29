"""Server-side timing helpers shared by profiler output and engine adapters.

``ttft`` is a request-level metric: for chunked prefill, every request
accumulates the CUDA duration of every prefill forward pass that contains it.
``prefill_pass_latency_s`` is deliberately separate and remains a per-forward
metric for bandwidth calculations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any


def _safe_int(value: Any, default: int | None = 0) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _valid_latency(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_server_timing(server_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate server records without mixing pass and request semantics."""

    prefill_pass_latencies: list[float] = []
    decode_pass_latencies: list[float] = []
    request_latency: dict[str, float] = {}
    completed_requests: set[str] = set()
    missing_per_req_info_count = 0

    for record in server_records:
        mode = record.get("forward_mode")
        latency = _valid_latency(record.get("latency"))
        if latency is None:
            continue

        if mode in ("decode", "decoding"):
            decode_pass_latencies.append(latency)
            continue
        if mode != "prefill":
            continue

        # Recorder warm-up probes are not benchmark requests.
        seq_lens_sum = _safe_int(record.get("seq_lens_sum"), default=None)
        if seq_lens_sum is not None and seq_lens_sum <= 10:
            continue

        prefill_pass_latencies.append(latency)
        per_req_info = record.get("per_req_info")
        if not isinstance(per_req_info, Sequence) or isinstance(
            per_req_info, (str, bytes)
        ) or not per_req_info:
            missing_per_req_info_count += 1
            continue

        valid_req_found = False
        for req_info in per_req_info:
            if not isinstance(req_info, Mapping):
                continue
            req_id = req_info.get("req_id", req_info.get("req_pool_idx"))
            if req_id is None:
                continue
            req_id = str(req_id)
            valid_req_found = True
            request_latency[req_id] = request_latency.get(req_id, 0.0) + latency
            if bool(req_info.get("is_last_chunk", False)):
                completed_requests.add(req_id)

        if not valid_req_found:
            missing_per_req_info_count += 1

    completed_ttfts = [
        request_latency[req_id]
        for req_id in completed_requests
        if req_id in request_latency
    ]
    # Any prefill pass without request provenance can hide chunks belonging to
    # an otherwise completed request. In that case, refusing to publish TTFT is
    # safer than silently under-counting it.
    ttft = _mean(completed_ttfts) if missing_per_req_info_count == 0 else None

    return {
        "ttft": ttft,
        "prefill_pass_latency_s": _mean(prefill_pass_latencies),
        "tpot": _mean(decode_pass_latencies),
        "ttft_request_count": len(completed_ttfts),
        "prefill_pass_count": len(prefill_pass_latencies),
        "decode_pass_count": len(decode_pass_latencies),
        "incomplete_request_count": len(set(request_latency) - completed_requests),
        "missing_per_req_info_count": missing_per_req_info_count,
    }


def _first_nonzero(*values: Any) -> Any:
    for value in values:
        if value is not None and value != 0:
            return value
    return 0


def resolve_performance_timings(
    server_records: Sequence[Mapping[str, Any]],
    *,
    res_dict: Mapping[str, Any],
    simple_ttft: float,
    simple_tpot: float,
) -> dict[str, Any]:
    """Resolve published timing fields and record their provenance."""

    timing = aggregate_server_timing(server_records)
    if server_records:
        ttft = timing["ttft"]
        ttft_source = (
            "server_request_aggregate"
            if ttft is not None
            else "unavailable_missing_per_req_info"
        )
        tpot = timing["tpot"]
        tpot_source = "server_decode_pass_mean" if tpot is not None else "unavailable"
    else:
        ttft = _first_nonzero(res_dict.get("ttft"), simple_ttft)
        tpot = _first_nonzero(res_dict.get("tpot"), simple_tpot)
        ttft_source = "client_or_continuous_fallback"
        tpot_source = "client_or_continuous_fallback"

    return {
        "ttft": ttft,
        "tpot": tpot,
        "ttft_source": ttft_source,
        "tpot_source": tpot_source,
        "prefill_pass_latency_s": timing["prefill_pass_latency_s"],
        "ttft_request_count": timing["ttft_request_count"],
        "prefill_pass_count": timing["prefill_pass_count"],
        "decode_pass_count": timing["decode_pass_count"],
        "incomplete_ttft_request_count": timing["incomplete_request_count"],
        "missing_prefill_per_req_info_count": timing[
            "missing_per_req_info_count"
        ],
    }


def _value_at(value: Any, index: int) -> Any:
    if value is None:
        return None
    try:
        return value[index]
    except (IndexError, KeyError, TypeError):
        return None


def _request_id(req: Any) -> str | None:
    req_id = getattr(req, "req_id", None)
    if req_id is None:
        req_id = getattr(req, "request_id", None)
    return str(req_id) if req_id is not None else None


def _prompt_length(req: Any) -> int | None:
    prompt_token_ids = getattr(req, "prompt_token_ids", None)
    if prompt_token_ids is not None:
        try:
            return len(prompt_token_ids)
        except TypeError:
            pass
    for name in (
        "total_prompt_tokens",
        "num_prompt_tokens",
        "prompt_len",
        "prompt_length",
    ):
        value = _safe_int(getattr(req, name, None), default=None)
        if value is not None:
            return value
    return None


def build_vllm_prefill_per_req_info(
    scheduler_output: Any,
    prefill_req_ids: set[str],
    prompt_lengths: MutableMapping[str, int],
) -> list[dict[str, Any]]:
    """Build reliable vLLM-v1 prefill metadata across new and cached chunks.

    Unknown prompt lengths are omitted rather than being mislabeled as final
    chunks. ``prompt_lengths`` is process-local state retained across scheduler
    steps and cleaned when a request enters decode.
    """

    scheduled_token_map = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not isinstance(scheduled_token_map, Mapping):
        return []
    scheduled_by_id = {str(req_id): value for req_id, value in scheduled_token_map.items()}
    target_ids = {str(req_id) for req_id in prefill_req_ids}
    computed_by_id: dict[str, int] = {}

    def capture_request(req: Any) -> None:
        req_id = _request_id(req)
        if req_id is None:
            return
        prompt_len = _prompt_length(req)
        if prompt_len is not None:
            prompt_lengths[req_id] = prompt_len
        computed = _safe_int(getattr(req, "num_computed_tokens", None), default=None)
        if computed is not None:
            computed_by_id[req_id] = computed

    for attr_name in ("scheduled_new_reqs", "scheduled_resumed_reqs"):
        requests = getattr(scheduler_output, attr_name, None) or []
        for req in requests:
            capture_request(req)

    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if cached is not None:
        cached_req_ids = getattr(cached, "req_ids", None)
        if cached_req_ids is not None:
            computed_tokens = getattr(cached, "num_computed_tokens", None)
            for index, req_id in enumerate(cached_req_ids):
                req_id = str(req_id)
                computed = _safe_int(_value_at(computed_tokens, index), default=None)
                if computed is not None:
                    computed_by_id[req_id] = computed
        else:
            try:
                for req in cached:
                    capture_request(req)
            except TypeError:
                pass

    per_req_info: list[dict[str, Any]] = []
    for req_id, scheduled_value in scheduled_by_id.items():
        if req_id not in target_ids:
            continue
        prompt_len = prompt_lengths.get(req_id)
        computed = computed_by_id.get(req_id)
        scheduled = _safe_int(scheduled_value, default=None)
        if prompt_len is None or computed is None or scheduled is None or scheduled <= 0:
            continue
        per_req_info.append(
            {
                "req_id": req_id,
                "extend_len": scheduled,
                "total_len": prompt_len,
                "is_last_chunk": computed + scheduled >= prompt_len,
            }
        )

    return per_req_info
