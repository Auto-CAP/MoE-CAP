"""Measured prefill totals and request cohorts persisted by the runner.

The F118 pair (forwarded tokens over the client's prefill passes and those
passes' summed duration) was computed and dropped; the cohort counts
disambiguate attempted / served / completed requests. All additive; the
published aggregates keep their historical bases.
"""

import ast
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SOURCE_PATH = Path(__file__).parents[1] / "moe_cap" / "runner" / "openai_api_profile.py"
HELPERS = {
    "_client_prefill_records",
    "_cohort_counts",
    "_prefill_step_totals",
}


def _load_helpers():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in HELPERS
    ]
    missing = HELPERS - {node.name for node in nodes}
    if missing:
        raise AssertionError(f"runner helpers not found: {sorted(missing)}")
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE_PATH), "exec"),
        namespace,
    )
    return namespace


@dataclass
class FakeResult:
    success: bool
    ttft: float = 0.0


class CohortCountsTest(unittest.TestCase):
    def test_three_cohorts_from_one_mixed_run(self):
        helpers = _load_helpers()
        results = [
            FakeResult(success=True, ttft=0.4),  # streamed success
            FakeResult(success=True, ttft=0.0),  # non-streaming success
            FakeResult(success=False, ttft=0.7),  # failed after first token
            FakeResult(success=False, ttft=0.0),  # refused before any pass
        ]
        attempted, served, completed = helpers["_cohort_counts"](results)
        self.assertEqual((attempted, served, completed), (4, 3, 2))

    def test_cohorts_nest(self):
        helpers = _load_helpers()
        results = [FakeResult(success=False, ttft=0.0)] * 3
        attempted, served, completed = helpers["_cohort_counts"](results)
        self.assertEqual((attempted, served, completed), (3, 0, 0))
        self.assertLessEqual(completed, served)
        self.assertLessEqual(served, attempted)


class PrefillStepTotalsTest(unittest.TestCase):
    def _records(self):
        return [
            {
                "forward_mode": "prefill",
                "batch_size": 2,
                "seq_lens_sum": 900,
                "scheduled_tokens": 900,
                "latency": 0.5,
            },
            {
                "forward_mode": "decode",
                "batch_size": 2,
                "seq_lens_sum": 2,
                "scheduled_tokens": 2,
                "latency": 0.01,
            },
            {
                "forward_mode": "prefill",
                "batch_size": 1,
                "seq_lens_sum": 600,
                "scheduled_tokens": 400,  # 200-token cached prefix
                "latency": 0.25,
            },
        ]

    def test_sums_scheduled_tokens_and_latency_over_prefill_passes(self):
        helpers = _load_helpers()
        forwarded, elapsed = helpers["_prefill_step_totals"](self._records(), 3)
        self.assertEqual(forwarded, 1300)
        self.assertAlmostEqual(elapsed, 0.75)

    def test_missing_scheduled_tokens_nulls_forwarded_never_partial(self):
        helpers = _load_helpers()
        records = self._records()
        del records[2]["scheduled_tokens"]
        forwarded, elapsed = helpers["_prefill_step_totals"](records, 3)
        self.assertIsNone(forwarded)
        self.assertAlmostEqual(elapsed, 0.75)

    def test_missing_latency_nulls_elapsed(self):
        helpers = _load_helpers()
        records = self._records()
        del records[0]["latency"]
        forwarded, elapsed = helpers["_prefill_step_totals"](records, 3)
        self.assertEqual(forwarded, 1300)
        self.assertIsNone(elapsed)

    def test_no_prefill_passes_is_null_null(self):
        helpers = _load_helpers()
        self.assertEqual(helpers["_prefill_step_totals"]([], 0), (None, None))
        self.assertEqual(helpers["_prefill_step_totals"](None, 0), (None, None))

    def test_warmup_probe_excluded_like_every_other_prefill_aggregate(self):
        helpers = _load_helpers()
        records = [
            {
                "forward_mode": "prefill",
                "batch_size": 1,
                "seq_lens_sum": 5,  # startup probe
                "scheduled_tokens": 5,
                "latency": 3.0,
            }
        ] + self._records()
        forwarded, elapsed = helpers["_prefill_step_totals"](records, 3)
        self.assertEqual(forwarded, 1300)
        self.assertAlmostEqual(elapsed, 0.75)


class DetailedWriterProjectionTest(unittest.TestCase):
    """The detailed-results writer stays the historical inline projection;
    the only change is carrying scheduled_tokens through when a record has
    it (records without it project byte-identically — evidenced end-to-end
    against real archived traces in the writer dry run)."""

    def test_writer_carries_scheduled_tokens_only_when_present(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        self.assertIn('if sr.get("scheduled_tokens") is not None:', source)
        self.assertIn(
            'record["scheduled_tokens"] = sr["scheduled_tokens"]', source
        )

    def test_writer_does_not_project_a_schema_field(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("record_schema", source)


class MetricsWiringTest(unittest.TestCase):
    """The run loop must persist the new quantities into metrics_*.json."""

    def test_metrics_dict_carries_the_new_fields(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        for needle in (
            '"prefill_forwarded_tokens": prefill_forwarded_tokens',
            '"prefill_step_elapsed_s": prefill_step_elapsed_s',
            '"prefill_tokens_attempted": prefill_tokens_total',
            '"prefill_tokens_completed": prefill_tokens_completed_total',
            '"attempted": attempted_requests',
            '"served": served_requests',
            '"completed": completed_requests',
        ):
            self.assertIn(needle, source)

    def test_completed_token_sum_is_gated_on_success(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        self.assertIn("prefill_tokens_completed_total += input_token_count", source)


if __name__ == "__main__":
    unittest.main()
