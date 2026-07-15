import pytest

from moe_cap.utils.acc_metrics import extract_answer


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Josh made a profit of $70,000.", "70000"),
        ("So the profit is $70,000.", "70000"),
        ("Answer: Marie ordered **2 boxes of pizza**.", "2"),
        ("### ✅ Answer: **20 cups** of feed in the final meal.", "20"),
        ("Answer: **$57,500**", "57500"),
        (r"The percentage is \\boxed{60\\%}.", "60"),
        ("#### 42", "42"),
        ("#### 106.12", "106"),
        ("Answer: $26.00", "26"),
        ("She makes $2 * 16 = $32 per day.", "32"),
    ],
)
def test_gsm8k_extracts_natural_final_answer_formats(text, expected):
    assert extract_answer(text, "gsm8k") == expected


def test_gsm8k_prefers_latest_explicit_final_answer_over_intermediate_values():
    text = """Total cost = $130,000.
Selling price = $200,000.
Josh made a profit of **$70,000**."""
    assert extract_answer(text, "gsm8k") == "70000"


def test_gsm8k_conclusion_uses_last_number_not_first_number_on_final_line():
    text = """2200 - 1980 = \\boxed{220}
So, Janeth's remaining balance after 12 months is $220."""
    assert extract_answer(text, "gsm8k") == "220"


def test_gsm8k_supports_terminal_answer_label_followed_by_one_answer_line():
    assert extract_answer("Final Answer:\nCharlie has 17 stickers left.", "gsm8k") == "17"
    assert extract_answer("Final Answer:\n**\\$140**", "gsm8k") == "140"


def test_gsm8k_does_not_use_arbitrary_last_number_without_a_final_answer_signal():
    text = "We have 3 apples and buy 4 more, but the response stops before answering."
    assert extract_answer(text, "gsm8k") == ""


def test_gsm8k_plain_word_answer_does_not_override_explicit_boxed_result():
    text = r"The final result is \\boxed{42}. To answer the follow-up, 5 is prime."
    assert extract_answer(text, "gsm8k") == "42"
