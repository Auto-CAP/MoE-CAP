"""Tests for warm-up exclusion in the client prefill pass filter.

The warm-up preamble is not always a tiny probe. Traces from the 2026-07-29
h200x8 campaign carry THREE non-client prefill passes — an engine probe
(seq_lens_sum 7-21), a seq-1 artifact, and a dataset-sample pass (seq 34-111)
that no size threshold can distinguish from a real prompt. The old per-record
``seq_lens_sum <= 10`` filter under-excluded, diluting prefill_avg_batch_size,
prefill_pass_latency_s and prefill expert activation (measured: published 36.86
vs true 51.2 on kimi h200x8). Client passes are anchored on the client request
count instead; the fixtures below are the measured real pass structures.
"""

import ast
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional


SOURCE_PATH = (
    Path(__file__).parents[1] / "moe_cap" / "runner" / "openai_api_profile.py"
)


def _load_helper():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_client_prefill_records"
    )
    namespace: Dict[str, Any] = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
    }
    exec(  # noqa: S102 - executing our own source under test
        compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE_PATH), "exec"),
        namespace,
    )
    return namespace["_client_prefill_records"]


def _passes(pairs):
    return [
        {"forward_mode": "prefill", "batch_size": b, "seq_lens_sum": s}
        for b, s in pairs
    ]


class ClientPrefillRecordsTest(unittest.TestCase):
    def setUp(self):
        self.helper = _load_helper()

    def test_three_pass_warmup_preamble_is_excluded(self):
        # kimi-k2.5 / arena-hard / h200x8 / 20260729-153647, as measured: probe
        # (1,21), artifact (1,1), dataset-sample warm-up (1,34), then the
        # client burst whose batch sizes sum to exactly 256.
        pairs = [(1, 21), (1, 1), (1, 34), (61, 7320), (71, 9186), (52, 5706), (55, 7617), (17, 2894)]
        out = self.helper(_passes(pairs), 256)
        got = [r["batch_size"] for r in out]
        self.assertEqual(got, [61, 71, 52, 55, 17])
        self.assertAlmostEqual(sum(got) / len(got), 51.2)
        # The reproduction: a threshold-only filter keeps the seq-21 probe and
        # the seq-34 warm-up and publishes the historical wrong mean.
        naive = [b for b, s in pairs if s > 10]
        self.assertAlmostEqual(sum(naive) / len(naive), 36.857142857142854)

    def test_dataset_sample_warmup_with_promptlike_length(self):
        # deepseek-r1 / gsm8k / h200x8 / 20260729-102819: the warm-up carries a
        # 105-token dataset sample — inside the real gsm8k prompt range.
        pairs = [(1, 7), (1, 1), (1, 105), (86, 8278), (99, 10161), (57, 5594), (14, 1372)]
        out = self.helper(_passes(pairs), 256)
        self.assertEqual([r["batch_size"] for r in out], [86, 99, 57, 14])

    def test_leading_singleton_client_pass_is_kept_when_anchored(self):
        # gpt-oss / gsm8k / b200x1 / 20260730-0727: the (1,173) pass IS a client
        # request — without it the big passes sum to 255, not 256.
        pairs = [(1, 6), (1, 1), (1, 173), (50, 8192), (129, 21928), (56, 9288), (20, 3499)]
        out = self.helper(_passes(pairs), 256)
        self.assertEqual([r["batch_size"] for r in out], [1, 50, 129, 56, 20])

    def test_chunked_prefill_is_never_trimmed(self):
        # LongBench shape: leading batch-1 passes are real chunks of thousands
        # of tokens; rows exceed requests by construction. Only the tiny probe
        # drops; the anchor cannot land, so nothing else is touched.
        pairs = [(1, 7), (1, 15839), (2, 9834), (1, 16384), (2, 21971)]
        out = self.helper(_passes(pairs), 4)
        self.assertEqual([r["batch_size"] for r in out], [1, 2, 1, 2])

    def test_retry_polluted_trace_falls_back_unchanged(self):
        # deepseek-r1 / arena-hard / h200x8 / 20260729-103030: two extra rows
        # hide inside the big passes, the anchor cannot land exactly, and the
        # helper must not guess.
        pairs = [(1, 27), (60, 7000), (70, 9000), (60, 6000), (48, 5000), (20, 2000)]
        out = self.helper(_passes(pairs), 256)
        self.assertEqual([r["batch_size"] for r in out], [1, 60, 70, 60, 48, 20])

    def test_client_failures_leave_sum_short_and_untouched(self):
        # 13 client-side failures (the D55 qwen3-4b shape): fewer prefilled
        # rows than prompts sent; no trim may fire.
        pairs = [(1, 9), (100, 8000), (100, 9000), (43, 4000)]
        out = self.helper(_passes(pairs), 256)
        self.assertEqual([r["batch_size"] for r in out], [100, 100, 43])

    def test_no_probe_vllm_shape_unchanged(self):
        pairs = [(128, 130000), (128, 131000)]
        out = self.helper(_passes(pairs), 256)
        self.assertEqual([r["batch_size"] for r in out], [128, 128])

    def test_per_rank_duplicated_bs1_trace_cannot_anchor(self):
        # Historical per-rank vLLM recorders repeat every pass once per TP
        # rank. At bs1 with short prompts every pass is (1, seq<1000), so an
        # uncapped trim would reach the anchor by discarding the first
        # (TP-1)*n REAL passes and bias every latency/activation mean toward
        # the tail. The trim cap must force the fallback instead.
        tp, n = 4, 8
        pairs = [(1, 150)] * (tp * n)
        out = self.helper(_passes(pairs), n)
        self.assertEqual(len(out), tp * n)

    def test_warmup_with_batch_above_one_never_trims(self):
        # Not an observed shape; documents the failure mode — the anchor
        # cannot land, dilution persists, and the helper must fall back
        # rather than guess.
        pairs = [(2, 40), (100, 8000), (100, 9000), (56, 5000)]
        out = self.helper(_passes(pairs), 256)
        self.assertEqual([r["batch_size"] for r in out], [2, 100, 100, 56])

    def test_zero_client_count_returns_threshold_filtered_list(self):
        pairs = [(1, 9), (1, 40), (10, 1000)]
        out = self.helper(_passes(pairs), 0)
        self.assertEqual([r["batch_size"] for r in out], [1, 10])

    def test_records_missing_fields_are_kept_and_fallback_safe(self):
        recs = [
            {"forward_mode": "prefill"},  # no batch_size, no seq_lens_sum
            {"forward_mode": "prefill", "batch_size": 100, "seq_lens_sum": 8000},
        ]
        out = self.helper(recs, 256)
        self.assertEqual(len(out), 2)


class AggregationWiringTest(unittest.TestCase):
    """The three prefill aggregates must be fed from the helper's output.

    Reverting the aggregation loop while keeping the helper would pass every
    fixture above; this pins the wiring at the AST level.
    """

    def test_prefill_lists_are_appended_inside_the_helper_loop(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper_loops = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "_client_prefill_records"
        ]
        self.assertEqual(len(helper_loops), 1)
        appended = {
            node.func.value.id
            for node in ast.walk(helper_loops[0])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
        }
        self.assertEqual(
            appended,
            {"prefill_activations", "prefill_latencies", "prefill_batch_sizes"},
        )
        # And nothing appends to those lists outside the helper loop.
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id.startswith("prefill_")
                and not any(node in ast.walk(loop) for loop in helper_loops)
            ):
                self.fail(
                    f"append to {node.func.value.id} outside the "
                    f"_client_prefill_records loop (line {node.lineno})"
                )


if __name__ == "__main__":
    unittest.main()
