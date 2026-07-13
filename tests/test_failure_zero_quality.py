"""Regression tests for the failure-as-zero benchmark quality policy.

Every failed or missing model-generation request must count as a zero-quality
prediction, and the reported quality total must stay the full requested /
evaluable sample count. Failed requests are never silently excluded.

The general-accuracy helpers live in ``moe_cap.runner.openai_api_profile``,
whose module top-level pulls in torch/transformers and is expensive (and, in
some environments, unimportable). To keep these tests focused and cheap we load
just the two pure helper functions out of the source file via ``ast`` instead of
importing the whole runner. The Arena-Hard aggregation helper lives in the
lightweight ``moe_cap.utils.arena_hard_judge`` module and is imported directly.
"""

import ast
import os
import unittest
from dataclasses import dataclass
from typing import Any, List, Tuple

from moe_cap.utils.arena_hard_judge import merge_failures_as_zero
from moe_cap.utils.acc_metrics import compute_accuracy_metrics


# --- Lightweight loader: extract runner helpers without importing the runner ---

_RUNNER_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "moe_cap",
    "runner",
    "openai_api_profile.py",
)

_WANTED_HELPERS = {
    "build_failure_zero_predictions",
    "arena_hard_success_failure_indices",
}


@dataclass
class _FakeResult:
    """Stand-in for RequestFuncOutput (only the fields the helpers touch)."""

    generated_text: str = ""
    success: bool = False
    error: str = ""


def _load_runner_helpers():
    """Compile only the pure helper functions out of the runner source.

    Avoids executing the module's expensive top-level imports (torch,
    transformers, ...) which are unnecessary — and sometimes unavailable — for
    exercising the pure alignment logic.
    """
    with open(_RUNNER_SRC, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=_RUNNER_SRC)

    wanted_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _WANTED_HELPERS
    ]
    missing = _WANTED_HELPERS - {n.name for n in wanted_nodes}
    if missing:
        raise AssertionError(f"runner helpers not found in source: {sorted(missing)}")

    module = ast.Module(body=wanted_nodes, type_ignores=[])
    code = compile(module, filename=_RUNNER_SRC, mode="exec")
    # Provide the names referenced by the helpers' annotations/bodies.
    namespace = {
        "List": List,
        "Any": Any,
        "Tuple": Tuple,
        "RequestFuncOutput": _FakeResult,
    }
    exec(code, namespace)
    return namespace


_HELPERS = _load_runner_helpers()
build_failure_zero_predictions = _HELPERS["build_failure_zero_predictions"]
arena_hard_success_failure_indices = _HELPERS["arena_hard_success_failure_indices"]


class TestBuildFailureZeroPredictions(unittest.TestCase):
    def test_middle_failures_score_zero_and_keep_total(self):
        # Perfect echoing model, but some requests fail in the middle.
        n = 20
        ground_truth = [str(i) for i in range(n)]
        failure_indices = {3, 7, 12, 15}
        results = [
            _FakeResult(generated_text=str(i), success=i not in failure_indices)
            for i in range(n)
        ]

        predictions, targets = build_failure_zero_predictions(results, ground_truth)

        # Total stays the full ground-truth length; nothing is dropped.
        self.assertEqual(len(predictions), n)
        self.assertEqual(len(targets), n)
        self.assertEqual(targets, ground_truth)

        # Failures become empty predictions at their ORIGINAL index; successes
        # keep their generated text aligned to the matching target.
        for i in range(n):
            if i in failure_indices:
                self.assertEqual(predictions[i], "")
            else:
                self.assertEqual(predictions[i], str(i))

        # Scoring: successes correct, failures score zero, total == n.
        metrics = compute_accuracy_metrics(
            predictions=predictions,
            targets=targets,
            dataset_name="gsm8k",
            extract_answers=True,
        )
        self.assertEqual(metrics["total"], n)
        self.assertEqual(metrics["correct"], n - len(failure_indices))
        self.assertAlmostEqual(
            metrics["exact_match"], (n - len(failure_indices)) / n
        )

    def test_missing_tail_results_count_zero(self):
        # Results object is shorter than ground truth (missing tail requests).
        n = 10
        ground_truth = [str(i) for i in range(n)]
        results = [_FakeResult(generated_text=str(i), success=True) for i in range(6)]

        predictions, targets = build_failure_zero_predictions(results, ground_truth)

        self.assertEqual(len(predictions), n)
        self.assertEqual(len(targets), n)
        # Present+successful indices carry text; missing tail is empty (zero).
        self.assertEqual(predictions[:6], [str(i) for i in range(6)])
        self.assertEqual(predictions[6:], ["", "", "", ""])

        metrics = compute_accuracy_metrics(
            predictions=predictions,
            targets=targets,
            dataset_name="gsm8k",
            extract_answers=True,
        )
        self.assertEqual(metrics["total"], n)
        self.assertEqual(metrics["correct"], 6)

    def test_failure_as_zero_matches_scaled_success_only_rate(self):
        # Mirrors the confirmed LongBench evidence relationship:
        # failure-as-zero acc == success-only acc * (successes / total).
        n = 256
        n_fail = 13
        ground_truth = [str(i) for i in range(n)]
        failure_indices = set(range(n_fail))  # first 13 fail
        results = [
            _FakeResult(generated_text=str(i), success=i not in failure_indices)
            for i in range(n)
        ]

        predictions, targets = build_failure_zero_predictions(results, ground_truth)
        metrics = compute_accuracy_metrics(
            predictions=predictions,
            targets=targets,
            dataset_name="gsm8k",
            extract_answers=True,
        )
        success_only_rate = 1.0  # perfect echo on the 243 successes
        self.assertEqual(metrics["total"], n)
        self.assertAlmostEqual(
            metrics["exact_match"], success_only_rate * (n - n_fail) / n
        )


class TestArenaHardSuccessFailureIndices(unittest.TestCase):
    def test_split_preserves_original_indices(self):
        n_eval = 12
        failure_indices = {2, 5, 9}
        results = [
            _FakeResult(success=i not in failure_indices) for i in range(n_eval)
        ]
        success_indices, failed_indices = arena_hard_success_failure_indices(
            results, n_eval
        )
        self.assertEqual(success_indices, [i for i in range(n_eval) if i not in failure_indices])
        self.assertEqual(failed_indices, sorted(failure_indices))
        # Disjoint and complete coverage of the evaluable population.
        self.assertEqual(len(success_indices) + len(failed_indices), n_eval)
        self.assertEqual(set(success_indices) & set(failed_indices), set())

    def test_missing_result_objects_are_failures(self):
        # Fewer result objects than evaluable samples -> tail counts as failed.
        n_eval = 8
        results = [_FakeResult(success=True) for _ in range(5)]
        success_indices, failed_indices = arena_hard_success_failure_indices(
            results, n_eval
        )
        self.assertEqual(success_indices, [0, 1, 2, 3, 4])
        self.assertEqual(failed_indices, [5, 6, 7])


class TestMergeFailuresAsZero(unittest.TestCase):
    def _judge_result(self, per_question, errors=0):
        return {
            "arena_hard_win_rate": 0.0,  # placeholder, recomputed by merge
            "arena_hard_wins": 0,
            "arena_hard_losses": 0,
            "arena_hard_ties": 0,
            "arena_hard_errors": errors,
            "arena_hard_total": len(per_question),
            "arena_hard_judge_model": "gpt-4.1",
            "arena_hard_per_question": per_question,
        }

    def test_failures_merge_as_losses_over_full_population(self):
        # 5 evaluable samples; indices 1 and 3 failed generation.
        # Judged (successful) scores at original indices 0,2,4.
        success_indices = [0, 2, 4]
        judged = [
            {"index": 0, "score": 1.0},  # win
            {"index": 1, "score": 0.5},  # tie (local index, remapped to orig 2)
            {"index": 2, "score": 1.0},  # win (remapped to orig 4)
        ]
        judge_result = self._judge_result(judged, errors=0)
        failed_records = [
            {"index": 1, "question": "q1", "uid": "u1"},
            {"index": 3, "question": "q3", "uid": "u3"},
        ]

        merged = merge_failures_as_zero(
            judge_result=judge_result,
            success_indices=success_indices,
            failed_records=failed_records,
            total=5,
        )

        # Total is the full evaluable population, not the successful subset.
        self.assertEqual(merged["arena_hard_total"], 5)
        self.assertEqual(merged["arena_hard_failed_generations"], 2)

        pq = merged["arena_hard_per_question"]
        self.assertEqual(len(pq), 5)
        # Ordered by original index and remapped correctly.
        self.assertEqual([r["index"] for r in pq], [0, 1, 2, 3, 4])
        # Failed generations score 0 and are flagged.
        self.assertEqual(pq[1]["score"], 0.0)
        self.assertTrue(pq[1]["generation_failed"])
        self.assertEqual(pq[3]["score"], 0.0)
        self.assertTrue(pq[3]["generation_failed"])
        # Successful generations keep their judged scores and are not flagged.
        self.assertEqual(pq[0]["score"], 1.0)
        self.assertFalse(pq[0]["generation_failed"])
        self.assertEqual(pq[2]["score"], 0.5)
        self.assertEqual(pq[4]["score"], 1.0)

        # Aggregate: wins are the two 1.0 scores; the two failures are losses;
        # the 0.5 is a tie. win_rate = (1+0+0.5+0+1)/5 * 100 = 50.0.
        self.assertEqual(merged["arena_hard_wins"], 2)
        self.assertEqual(merged["arena_hard_losses"], 2)
        self.assertEqual(merged["arena_hard_ties"], 1)
        self.assertEqual(merged["arena_hard_win_rate"], 50.0)
        # Judge/API error count is carried through unchanged.
        self.assertEqual(merged["arena_hard_errors"], 0)

    def test_all_failed_gives_zero_win_rate_full_total(self):
        judge_result = self._judge_result([], errors=0)
        failed_records = [{"index": i} for i in range(4)]
        merged = merge_failures_as_zero(
            judge_result=judge_result,
            success_indices=[],
            failed_records=failed_records,
            total=4,
        )
        self.assertEqual(merged["arena_hard_total"], 4)
        self.assertEqual(merged["arena_hard_failed_generations"], 4)
        self.assertEqual(merged["arena_hard_losses"], 4)
        self.assertEqual(merged["arena_hard_win_rate"], 0.0)
        self.assertEqual([r["index"] for r in merged["arena_hard_per_question"]], [0, 1, 2, 3])
        self.assertTrue(all(r["score"] == 0.0 for r in merged["arena_hard_per_question"]))

    def test_judge_errors_carry_through(self):
        # Judge/API failures (scored 0.5 ties) are a separate concern and their
        # count must survive the merge untouched.
        success_indices = [0, 1]
        judged = [
            {"index": 0, "score": 0.5, "error": "judge timeout"},
            {"index": 1, "score": 1.0},
        ]
        merged = merge_failures_as_zero(
            judge_result=self._judge_result(judged, errors=1),
            success_indices=success_indices,
            failed_records=[{"index": 2}],
            total=3,
        )
        self.assertEqual(merged["arena_hard_errors"], 1)
        self.assertEqual(merged["arena_hard_total"], 3)
        self.assertEqual(merged["arena_hard_ties"], 1)  # the judge-error 0.5
        self.assertEqual(merged["arena_hard_losses"], 1)  # the failed generation

    def test_mismatched_judged_count_fails_loudly(self):
        # judged records must match the success indices exactly.
        judge_result = self._judge_result([{"index": 0, "score": 1.0}])
        with self.assertRaises(ValueError):
            merge_failures_as_zero(
                judge_result=judge_result,
                success_indices=[0, 1],  # two successes but only one judged record
                failed_records=[],
                total=2,
            )

    def test_incomplete_coverage_fails_loudly(self):
        # success + failed must cover exactly `total` distinct indices.
        judge_result = self._judge_result([{"index": 0, "score": 1.0}])
        with self.assertRaises(ValueError):
            merge_failures_as_zero(
                judge_result=judge_result,
                success_indices=[0],
                failed_records=[{"index": 1}],
                total=5,  # claims 5 evaluable but only 2 covered
            )


if __name__ == "__main__":
    unittest.main()
