"""Regression tests for server-side TTFT semantics."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from moe_cap.utils.server_timing_utils import (
    aggregate_server_timing,
    build_vllm_prefill_per_req_info,
    resolve_performance_timings,
)


def _prefill(latency, seq_lens_sum, per_req_info=None):
    record = {
        "forward_mode": "prefill",
        "latency": latency,
        "seq_lens_sum": seq_lens_sum,
    }
    if per_req_info is not None:
        record["per_req_info"] = per_req_info
    return record


def _req(req_id, *, last):
    return {
        "req_id": req_id,
        "extend_len": 10,
        "total_len": 20,
        "is_last_chunk": last,
    }


def test_request_ttft_is_not_prefill_pass_mean():
    records = [
        # Recorder warm-up probes must not affect either timing.
        _prefill(99.0, 6),
        _prefill(0.1, 20, [_req("A", last=False), _req("B", last=False)]),
        _prefill(0.2, 20, [_req("A", last=True)]),
        _prefill(0.3, 20, [_req("B", last=True)]),
        {"forward_mode": "decode", "latency": 0.01, "seq_lens_sum": 2},
        {"forward_mode": "decode", "latency": 0.03, "seq_lens_sum": 2},
    ]

    timing = aggregate_server_timing(records)

    # A: 0.1 + 0.2 = 0.3; B: 0.1 + 0.3 = 0.4.
    assert timing["ttft"] == pytest.approx(0.35)
    assert timing["prefill_pass_latency_s"] == pytest.approx(0.2)
    assert timing["tpot"] == pytest.approx(0.02)
    assert timing["ttft_request_count"] == 2
    assert timing["prefill_pass_count"] == 3


def test_serializer_uses_request_ttft_and_keeps_pass_latency_separate():
    records = [
        _prefill(0.1, 20, [_req("A", last=False), _req("B", last=False)]),
        _prefill(0.2, 20, [_req("A", last=True)]),
        _prefill(0.3, 20, [_req("B", last=True)]),
    ]

    performance = resolve_performance_timings(
        records,
        res_dict={"ttft": 123.0, "tpot": 456.0},
        simple_ttft=789.0,
        simple_tpot=987.0,
    )

    assert performance["ttft"] == pytest.approx(0.35)
    assert performance["ttft_source"] == "server_request_aggregate"
    assert performance["prefill_pass_latency_s"] == pytest.approx(0.2)


def test_server_records_without_request_provenance_are_not_called_ttft():
    performance = resolve_performance_timings(
        [_prefill(0.2, 2048)],
        res_dict={"ttft": 0.2},
        simple_ttft=0.5,
        simple_tpot=0.01,
    )

    assert performance["ttft"] is None
    assert performance["ttft_source"] == "unavailable_missing_per_req_info"
    assert performance["prefill_pass_latency_s"] == pytest.approx(0.2)


def test_no_server_records_uses_explicit_client_fallback():
    performance = resolve_performance_timings(
        [],
        res_dict={"ttft": 0.7, "tpot": 0.02},
        simple_ttft=0.8,
        simple_tpot=0.03,
    )

    assert performance["ttft"] == pytest.approx(0.7)
    assert performance["tpot"] == pytest.approx(0.02)
    assert performance["ttft_source"] == "client_or_continuous_fallback"
    assert performance["prefill_pass_latency_s"] is None


def test_vllm_cached_chunks_use_persistent_prompt_length():
    prompt_lengths = {}
    first_step = SimpleNamespace(
        num_scheduled_tokens={"r1": 2048},
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="r1",
                prompt_token_ids=list(range(4096)),
                num_computed_tokens=0,
            )
        ],
        scheduled_resumed_reqs=[],
        scheduled_cached_reqs=None,
    )

    first = build_vllm_prefill_per_req_info(
        first_step, {"r1"}, prompt_lengths
    )
    assert first == [
        {
            "req_id": "r1",
            "extend_len": 2048,
            "total_len": 4096,
            "is_last_chunk": False,
        }
    ]

    second_step = SimpleNamespace(
        num_scheduled_tokens={"r1": 2048},
        scheduled_new_reqs=[],
        scheduled_resumed_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r1"],
            num_computed_tokens=[2048],
            num_output_tokens=[0],
        ),
    )
    second = build_vllm_prefill_per_req_info(
        second_step, {"r1"}, prompt_lengths
    )
    assert second == [
        {
            "req_id": "r1",
            "extend_len": 2048,
            "total_len": 4096,
            "is_last_chunk": True,
        }
    ]


def test_vllm_unknown_prompt_length_does_not_default_to_last_chunk():
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"unknown": 2048},
        scheduled_new_reqs=[],
        scheduled_resumed_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["unknown"],
            num_computed_tokens=[2048],
            num_output_tokens=[0],
        ),
    )

    assert build_vllm_prefill_per_req_info(
        scheduler_output, {"unknown"}, {}
    ) == []


def test_profiler_serializer_cannot_restore_prefill_pass_mean_as_ttft():
    source = (
        Path(__file__).parents[1]
        / "moe_cap"
        / "runner"
        / "openai_api_profile.py"
    ).read_text(encoding="utf-8")

    assert "resolve_performance_timings(" in source
    assert '"ttft": (sum(prefill_latencies)' not in source
    assert "**performance_timings" in source
