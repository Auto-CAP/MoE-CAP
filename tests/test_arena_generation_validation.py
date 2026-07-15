from pathlib import Path

from moe_cap.utils.arena_hard_judge import is_degenerate_generation


ROOT = Path(__file__).resolve().parents[1]


def test_arena_marks_empty_and_whitespace_only_generations_degenerate():
    assert is_degenerate_generation("")
    assert is_degenerate_generation(" \n\t" * 1024)


def test_arena_marks_long_repeated_punctuation_tail_degenerate():
    assert is_degenerate_generation("valid prefix" + "!" * 512)
    assert is_degenerate_generation("reasoning" + "?" * 2000 + "  \n")


def test_arena_keeps_short_or_normal_answers():
    assert not is_degenerate_generation("A normal Arena answer!")
    assert not is_degenerate_generation("YES")
    assert not is_degenerate_generation("0")
    assert not is_degenerate_generation("!" * 511)


def test_posthoc_verifier_filters_degenerate_generations_before_judging():
    source = (ROOT / "verify_amd_moe_arena.py").read_text(encoding="utf-8")
    assert "and not is_degenerate_generation" in source
    assert '"empty or degenerate generation"' in source
    assert "merge_failures_as_zero" in source
