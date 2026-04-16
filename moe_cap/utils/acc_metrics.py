"""
Accuracy metrics utilities for evaluating model predictions.

This module provides functions to compute various accuracy metrics
such as exact match (EM), used across different profilers and evaluation tasks.
"""

import re
import string
import importlib
from collections import Counter
from typing import List, Dict, Any, Optional


def extract_answer(text: str, dataset_name: str) -> str:
    """
    Extract the final answer from generated text based on dataset type.

    Args:
        text: The generated text containing the answer
        dataset_name: Name of the dataset to determine extraction strategy

    Returns:
        Extracted answer string, or empty string if no valid answer found

    Examples:
        >>> extract_answer("Step 1... #### 42", "gsm8k")
        '42'
        >>> extract_answer("The answer is: 123", "math")
        '123'
        >>> extract_answer("\\boxed{42}", "math")
        '42'
    """
    if not text or not text.strip():
        return ""

    if dataset_name.lower() in ["gsm8k", "math", "numinamath"]:
        # Priority 1: Look for explicit answer markers (high confidence patterns)
        # These are the primary patterns that indicate a deliberate final answer
        high_confidence_patterns = [
            r"####\s*(-?\d[\d,]*)",  # GSM8K style: #### 42
            r"\\boxed\{(-?\d[\d,]*)\}",  # LaTeX boxed: \boxed{42}
            r"\*\*Final Answer:\*\*\s*\$?(-?\d[\d,]*)",  # **Final Answer:** 42 or **Final Answer:** $42
            r"\*\*Answer:\*\*\s*\$?(-?\d[\d,]*)",  # **Answer:** 42 or **Answer:** $42
            r"\*\*\$?(-?\d[\d,]*)\$?\*\*\s*\.?\s*$",  # **42** or **$42** at end of text
            r"(?:final\s+)?answer\s*(?:is|:)\s*\$?(-?\d[\d,]*)\$?(?:\s*\.)?$",  # "answer is 42" at end
            r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*\$?(-?\d[\d,]*)\$?",  # "the final answer is 42"
            r"=\s*\*?\*?\$?(-?\d[\d,]*)\$?\*?\*?\s*\.?\s*$",  # "= 42" or "= **42**" at the very end
        ]

        for pattern in high_confidence_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).replace(",", "").strip()

        # Priority 2: Look for boxed content (common in math)
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
        if boxed_match:
            content = boxed_match.group(1).strip()
            # Extract number from boxed content
            num_match = re.search(r"(-?\d[\d,]*)", content)
            if num_match:
                return num_match.group(1).replace(",", "").strip()

        # Priority 3: Look for bold numbers anywhere (** **) - common in markdown output
        bold_patterns = [
            r"\*\*\$?(-?\d[\d,]*)\$?\*\*",  # **42** or **$42**
        ]
        for pattern in bold_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Return the last bold number (most likely the final answer)
                return matches[-1].replace(",", "").strip()

        # Priority 4: Check last line for a standalone number or answer pattern
        lines = text.strip().split("\n")
        for line in reversed(lines[-3:]):  # Check last 3 lines
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Look for "So the answer is X" or similar concluding statements
            conclude_match = re.search(
                r"(?:so|thus|therefore|hence)[\s,]+(?:the\s+)?(?:final\s+)?answer\s+(?:is|=)\s*\$?(-?\d[\d,]*)",
                line,
                re.IGNORECASE,
            )
            if conclude_match:
                return conclude_match.group(1).replace(",", "").strip()
            # Check if the line is just a number (possibly with $ signs for formatting)
            standalone_match = re.match(r"^\$?(-?\d[\d,]*)\$?\.?$", line)
            if standalone_match:
                return standalone_match.group(1).replace(",", "").strip()

        # NOTE: We do NOT use fallback "last number in text" as it's too unreliable
        # and leads to false positives when the model hasn't actually answered
        return ""

    if dataset_name.lower() in ["longbench_v2"]:
        # Official LongBench v2 extraction: match "The correct answer is (X)" format
        text_clean = text.replace("*", "")
        match = re.search(
            r"The correct answer is \(([A-D])\)", text_clean, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()
        match = re.search(r"The correct answer is ([A-D])", text_clean, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # Fallback: find any standalone A-D letter
        match = re.search(r"\b([A-D])\b", text_clean)
        if match:
            return match.group(1).upper()
        return ""

    if dataset_name.lower() in ["mmlu-pro", "mmlu_pro"]:
        text_clean = text.replace("*", "")
        # Match "The answer is (X)" or "The answer is X"
        match = re.search(r"[Tt]he answer is \(?([A-J])\)?", text_clean)
        if match:
            return match.group(1).upper()
        match = re.search(
            r"(?:answer|option)\s*(?:is|:)\s*\(?([A-J])\)?", text_clean, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()
        # Fallback: any standalone A-J letter
        match = re.search(r"\b([A-J])\b", text_clean)
        if match:
            return match.group(1).upper()
        return ""

    # Strip thinking tokens — models may use various formats:
    # gpt-oss: "analysis...assistantfinal<answer>"
    # Qwen3: "<think>...</think><answer>"
    # DeepSeek: "<reasoning>...</reasoning><answer>"
    if "assistantfinal" in text:
        text = text.split("assistantfinal")[-1]
    elif "</think>" in text:
        text = text.split("</think>")[-1]
    elif "</reasoning>" in text:
        text = text.split("</reasoning>")[-1]
    elif text.startswith("analysis") and "assistantfinal" not in text:
        text = ""

    text = text.strip()
    if text.lower() in ["unanswerable", "not answerable", "cannot be answered"]:
        return "None"

    return text


def compute_exact_match(predictions: List[str], targets: List[str]) -> Dict[str, Any]:
    """
    Compute exact match (EM) accuracy between predictions and targets.

    Args:
        predictions: List of predicted answers
        targets: List of ground truth answers

    Returns:
        Dictionary containing:
            - exact_match: EM score (0.0 to 1.0)
            - correct: Number of correct predictions
            - total: Total number of predictions
            - no_answer: Number of empty/missing predictions

    Raises:
        ValueError: If predictions and targets have different lengths

    Examples:
        >>> compute_exact_match(['42', '100'], ['42', '99'])
        {'exact_match': 0.5, 'correct': 1, 'total': 2, 'no_answer': 0}
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Predictions ({len(predictions)}) and targets ({len(targets)}) "
            "must have the same length"
        )

    correct = 0
    no_answer = 0
    for pred, target in zip(predictions, targets):
        # Normalize both strings for comparison
        pred_norm = str(pred).strip().lower() if pred else ""
        target_norm = str(target).strip().lower()

        # Count empty/no predictions separately
        if not pred_norm:
            no_answer += 1
            continue

        if pred_norm == target_norm:
            correct += 1

    em_score = correct / len(predictions) if len(predictions) > 0 else 0.0

    return {
        "exact_match": em_score,
        "correct": correct,
        "total": len(predictions),
        "no_answer": no_answer,
    }


def _normalize_answer(s: str) -> str:
    """Official LongBench normalization for English QA."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _f1_score(prediction_tokens, ground_truth_tokens):
    """Token-level F1 using Counter (official implementation)."""
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def _qa_f1_score(prediction, ground_truth):
    """QA F1 with normalization (official)."""
    norm_pred = _normalize_answer(prediction)
    norm_gt = _normalize_answer(ground_truth)
    pred_tokens = norm_pred.split()
    gt_tokens = norm_gt.split()
    return _f1_score(pred_tokens, gt_tokens)


def _rouge_score(prediction, ground_truth):
    """ROUGE-L F1 score."""
    try:
        rouge_module = importlib.import_module("rouge")
        rouge = rouge_module.Rouge()
        scores = rouge.get_scores(
            prediction if prediction.strip() else "empty", ground_truth
        )
        return scores[0]["rouge-l"]["f"]
    except Exception:
        return 0.0


def _classification_score(prediction, ground_truth, all_classes=None):
    """Official classification metric."""
    if not all_classes:
        return float(ground_truth.lower() in prediction.lower())
    em_match_list = []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def _count_score(prediction, ground_truth):
    """Count matching score (for passage_count)."""
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if n == ground_truth)
    return min(right_num, 1)


def _retrieval_score(prediction, ground_truth):
    """Retrieval score (for passage_retrieval)."""
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    right_num = sum(1 for m in matches if m in prediction)
    return right_num / len(matches) if matches else 0.0


def _code_sim_score(prediction, ground_truth):
    """Code similarity using fuzz ratio."""
    try:
        fuzz = importlib.import_module("fuzzywuzzy.fuzz")
        return fuzz.ratio(prediction, ground_truth) / 100.0
    except Exception:
        # Fallback: simple character overlap
        common = sum(1 for a, b in zip(prediction, ground_truth) if a == b)
        total = max(len(prediction), len(ground_truth), 1)
        return common / total


_LONGBENCH_V1_METRICS = {
    "narrativeqa": _qa_f1_score,
    "qasper": _qa_f1_score,
    "multifieldqa_en": _qa_f1_score,
    "hotpotqa": _qa_f1_score,
    "2wikimqa": _qa_f1_score,
    "musique": _qa_f1_score,
    "triviaqa": _qa_f1_score,
    "gov_report": _rouge_score,
    "qmsum": _rouge_score,
    "multi_news": _rouge_score,
    "samsum": _rouge_score,
    "trec": _classification_score,
    "passage_retrieval_en": _retrieval_score,
    "passage_count": _count_score,
    "lcc": _code_sim_score,
    "repobench-p": _code_sim_score,
}

# Datasets where prediction should be truncated to first line
_TRUNCATE_FIRST_LINE = {"trec", "triviaqa", "samsum"}


def compute_accuracy_metrics(
    predictions: List[str],
    targets: List[str],
    dataset_name: str,
    extract_answers: bool = True,
) -> Dict[str, Any]:
    """
    Compute accuracy metrics with optional answer extraction.

    This is a convenience function that combines answer extraction and
    exact match computation in a single call.

    Args:
        predictions: List of predicted texts (raw or extracted)
        targets: List of ground truth answers
        dataset_name: Name of the dataset (used for answer extraction)
        extract_answers: Whether to extract answers from predictions

    Returns:
        Dictionary containing accuracy metrics

    Examples:
        >>> compute_accuracy_metrics(
        ...     ['The answer is 42', 'Result: 100'],
        ...     ['42', '100'],
        ...     'gsm8k',
        ...     extract_answers=True
        ... )
        {'exact_match': 1.0, 'correct': 2, 'total': 2}
    """
    if extract_answers:
        extracted_predictions = [
            extract_answer(pred, dataset_name) for pred in predictions
        ]
    else:
        extracted_predictions = predictions
    result = compute_exact_match(extracted_predictions, targets)

    if dataset_name.lower() in ["longbench_v1"]:
        scores = []
        for pred, tgt_info in zip(predictions, targets):
            if isinstance(tgt_info, dict):
                answers = tgt_info.get("answers", [])
                subset = tgt_info.get("dataset", "")
                all_classes = tgt_info.get("all_classes")
            else:
                answers = [tgt_info]
                subset = ""
                all_classes = None

            metric_fn = _LONGBENCH_V1_METRICS.get(subset, _qa_f1_score)

            # Truncate to first line for certain datasets
            if subset in _TRUNCATE_FIRST_LINE:
                pred = pred.lstrip("\n").split("\n")[0]

            # Score against all ground truths, take max
            best_score = 0.0
            for answer in answers:
                if metric_fn == _classification_score:
                    score = metric_fn(pred, answer, all_classes=all_classes)
                else:
                    score = metric_fn(pred, answer)
                best_score = max(best_score, score)
            scores.append(best_score)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        result["exact_match"] = avg_score
        result["correct"] = sum(1 for s in scores if s > 0.5)
        result["total"] = len(scores)

    return result


def format_accuracy_summary(metrics: Dict[str, Any]) -> str:
    """
    Format accuracy metrics into a human-readable string.

    Args:
        metrics: Dictionary containing accuracy metrics

    Returns:
        Formatted string summary

    Examples:
        >>> format_accuracy_summary({'exact_match': 0.85, 'correct': 850, 'total': 1000})
        'EM = 0.8500 (850/1000)'
    """
    em = metrics.get("exact_match", 0.0)
    correct = metrics.get("correct", 0)
    total = metrics.get("total", 0)
    no_answer = metrics.get("no_answer", 0)

    if no_answer > 0:
        return f"EM = {em:.4f} ({correct}/{total}, {no_answer} no answer)"
    return f"EM = {em:.4f} ({correct}/{total})"
