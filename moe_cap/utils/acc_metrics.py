"""
Accuracy metrics utilities for evaluating model predictions.

This module provides functions to compute various accuracy metrics
such as exact match (EM), used across different profilers and evaluation tasks.
"""

import re
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
            r'####\s*(-?\d[\d,]*)',  # GSM8K style: #### 42
            r'\\boxed\{(-?\d[\d,]*)\}',  # LaTeX boxed: \boxed{42}
            r'\*\*Final Answer:\*\*\s*\$?(-?\d[\d,]*)',  # **Final Answer:** 42 or **Final Answer:** $42
            r'\*\*Answer:\*\*\s*\$?(-?\d[\d,]*)',  # **Answer:** 42 or **Answer:** $42
            r'\*\*\$?(-?\d[\d,]*)\$?\*\*\s*\.?\s*$',  # **42** or **$42** at end of text
            r'(?:final\s+)?answer\s*(?:is|:)\s*\$?(-?\d[\d,]*)\$?(?:\s*\.)?$',  # "answer is 42" at end
            r'(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*\$?(-?\d[\d,]*)\$?',  # "the final answer is 42"
            r'=\s*\*?\*?\$?(-?\d[\d,]*)\$?\*?\*?\s*\.?\s*$',  # "= 42" or "= **42**" at the very end
        ]
        
        for pattern in high_confidence_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).replace(',', '').strip()
        
        # Priority 2: Look for boxed content (common in math)
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            content = boxed_match.group(1).strip()
            # Extract number from boxed content
            num_match = re.search(r'(-?\d[\d,]*)', content)
            if num_match:
                return num_match.group(1).replace(',', '').strip()
        
        # Priority 3: Look for bold numbers anywhere (** **) - common in markdown output
        bold_patterns = [
            r'\*\*\$?(-?\d[\d,]*)\$?\*\*',  # **42** or **$42**
        ]
        for pattern in bold_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Return the last bold number (most likely the final answer)
                return matches[-1].replace(',', '').strip()
        
        # Priority 4: Check last line for a standalone number or answer pattern
        lines = text.strip().split('\n')
        for line in reversed(lines[-3:]):  # Check last 3 lines
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Look for "So the answer is X" or similar concluding statements
            conclude_match = re.search(
                r'(?:so|thus|therefore|hence)[\s,]+(?:the\s+)?(?:final\s+)?answer\s+(?:is|=)\s*\$?(-?\d[\d,]*)',
                line, re.IGNORECASE
            )
            if conclude_match:
                return conclude_match.group(1).replace(',', '').strip()
            # Check if the line is just a number (possibly with $ signs for formatting)
            standalone_match = re.match(r'^\$?(-?\d[\d,]*)\$?\.?$', line)
            if standalone_match:
                return standalone_match.group(1).replace(',', '').strip()
        
        # NOTE: We do NOT use fallback "last number in text" as it's too unreliable
        # and leads to false positives when the model hasn't actually answered
        return ""
    
    # For other datasets, return the full text stripped
    return text.strip()


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
        "no_answer": no_answer
    }


def compute_accuracy_metrics(
    predictions: List[str],
    targets: List[str],
    dataset_name: str,
    extract_answers: bool = True
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
    
    return compute_exact_match(extracted_predictions, targets)


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
    em = metrics.get('exact_match', 0.0)
    correct = metrics.get('correct', 0)
    total = metrics.get('total', 0)
    no_answer = metrics.get('no_answer', 0)
    
    if no_answer > 0:
        return f"EM = {em:.4f} ({correct}/{total}, {no_answer} no answer)"
    return f"EM = {em:.4f} ({correct}/{total})"
