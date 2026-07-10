"""Regression test: successful-only accuracy must stay index-aligned.

Failures in the middle of a run must NOT shift ground-truth targets. The old
code paired ``[r for r in results if r.success]`` against
``ground_truth[:len(predictions)]``, which slid every target after the first
failure by one slot and destroyed accuracy. This test reproduces the fixed
pairing logic from ``OpenAIAPIMoEProfiler.run_async`` and proves alignment.
"""

import unittest
from dataclasses import dataclass

from moe_cap.utils.acc_metrics import compute_accuracy_metrics


@dataclass
class _FakeResult:
    generated_text: str
    success: bool


def _paired_predictions_targets(results, ground_truth):
    """Fixed pairing: successful-only, aligned to each result's ORIGINAL index."""
    paired = [
        (r.generated_text, ground_truth[i])
        for i, r in enumerate(results)
        if r.success and i < len(ground_truth)
    ]
    predictions = [p for p, _ in paired]
    targets = [t for _, t in paired]
    return predictions, targets


def _old_buggy_predictions_targets(results, ground_truth):
    """Original shifting logic, kept only to demonstrate the regression."""
    predictions = [r.generated_text for r in results if r.success]
    targets = ground_truth[: len(predictions)]
    return predictions, targets


class TestAccuracyIndexAlignment(unittest.TestCase):
    def test_mid_run_failures_do_not_shift_targets(self):
        # Model that always echoes the correct answer for its index.
        n = 20
        ground_truth = [str(i) for i in range(n)]
        failure_indices = {3, 7, 12, 15}  # failures scattered in the middle
        results = [
            _FakeResult(generated_text=str(i), success=i not in failure_indices)
            for i in range(n)
        ]

        predictions, targets = _paired_predictions_targets(results, ground_truth)

        # Every surviving prediction is paired with the target at its own index.
        expected = [str(i) for i in range(n) if i not in failure_indices]
        self.assertEqual(predictions, expected)
        self.assertEqual(targets, expected)
        self.assertEqual(len(predictions), n - len(failure_indices))

        # A perfect echoing model must score 1.0 once alignment is correct.
        metrics = compute_accuracy_metrics(
            predictions=predictions,
            targets=targets,
            dataset_name="gsm8k",
            extract_answers=True,
        )
        self.assertEqual(metrics["exact_match"], 1.0)
        self.assertEqual(metrics["correct"], metrics["total"])

    def test_old_logic_would_have_misaligned(self):
        # Guard that the scenario actually exercises the bug: the buggy pairing
        # mismatches targets, so the same perfect model scores below 1.0.
        n = 20
        ground_truth = [str(i) for i in range(n)]
        failure_indices = {3, 7, 12, 15}
        results = [
            _FakeResult(generated_text=str(i), success=i not in failure_indices)
            for i in range(n)
        ]

        old_preds, old_targets = _old_buggy_predictions_targets(results, ground_truth)
        # Targets are just the first len(preds) of ground_truth -> shifted.
        self.assertNotEqual(old_targets, old_preds)
        old_metrics = compute_accuracy_metrics(
            predictions=old_preds,
            targets=old_targets,
            dataset_name="gsm8k",
            extract_answers=True,
        )
        self.assertLess(old_metrics["exact_match"], 1.0)


def _arena_hard_alignment(results, all_input_raw, all_uids, baseline_dict):
    """Fixed Arena-Hard pairing from ``run_async`` (UID path).

    Returns (predictions, questions, baseline_answers), each aligned to the
    ORIGINAL result index of every successful request.
    """
    success_indices = [i for i, r in enumerate(results) if r.success]
    predictions = [results[i].generated_text for i in success_indices]
    questions = [all_input_raw[i] for i in success_indices if i < len(all_input_raw)]
    uids = [all_uids[i] for i in success_indices if i < len(all_uids)]
    baseline_answers = [baseline_dict.get(uid, "") for uid in uids]
    return predictions, questions, baseline_answers


def _arena_hard_positional(results, baseline_values):
    """Fixed Arena-Hard positional fallback (no UIDs): index-safe or raise."""
    success_indices = [i for i, r in enumerate(results) if r.success]
    out_of_range = [i for i in success_indices if i >= len(baseline_values)]
    if out_of_range:
        raise ValueError("cannot align baselines without UIDs")
    return [baseline_values[i] for i in success_indices]


class TestArenaHardIndexAlignment(unittest.TestCase):
    def test_uid_path_stays_aligned(self):
        n = 12
        all_input_raw = [f"q{i}" for i in range(n)]
        all_uids = [f"uid{i}" for i in range(n)]
        # Baseline keyed by uid; deliberately unordered to prove UID lookup wins.
        baseline_dict = {f"uid{i}": f"base{i}" for i in reversed(range(n))}
        failure_indices = {2, 5, 9}
        results = [
            _FakeResult(generated_text=f"ans{i}", success=i not in failure_indices)
            for i in range(n)
        ]

        preds, questions, baselines = _arena_hard_alignment(
            results, all_input_raw, all_uids, baseline_dict
        )

        kept = [i for i in range(n) if i not in failure_indices]
        self.assertEqual(preds, [f"ans{i}" for i in kept])
        self.assertEqual(questions, [f"q{i}" for i in kept])
        # Each baseline matches the question at the same original index.
        self.assertEqual(baselines, [f"base{i}" for i in kept])
        self.assertEqual(len(preds), len(questions))
        self.assertEqual(len(preds), len(baselines))

    def test_positional_fallback_stays_aligned(self):
        n = 8
        baseline_values = [f"base{i}" for i in range(n)]
        failure_indices = {1, 4}
        results = [
            _FakeResult(generated_text=f"ans{i}", success=i not in failure_indices)
            for i in range(n)
        ]

        baselines = _arena_hard_positional(results, baseline_values)
        kept = [i for i in range(n) if i not in failure_indices]
        # Positional baseline follows original indices, not a shifted prefix.
        self.assertEqual(baselines, [f"base{i}" for i in kept])

    def test_positional_fallback_fails_loudly_when_short(self):
        n = 6
        # Baseline shorter than the dataset -> a successful high index can't align.
        baseline_values = [f"base{i}" for i in range(4)]
        results = [
            _FakeResult(generated_text=f"ans{i}", success=True) for i in range(n)
        ]
        with self.assertRaises(ValueError):
            _arena_hard_positional(results, baseline_values)


if __name__ == "__main__":
    unittest.main()
