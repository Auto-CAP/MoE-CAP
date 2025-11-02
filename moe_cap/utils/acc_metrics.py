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
        Extracted answer string
        
    Examples:
        >>> extract_answer("Step 1... #### 42", "gsm8k")
        '42'
        >>> extract_answer("The answer is: 123", "math")
        '123'
        >>> extract_answer("Result is 3.14 but final answer is 42", "gsm8k")
        '42'
    """
    if dataset_name.lower() in ["gsm8k", "math", "numinamath"]:
        # Try to extract integer from the last line or after common answer indicators
        # Pattern matches integers only (no decimals)
        patterns = [
            r'####\s*(-?\d[\d,]*)',  # GSM8K style (integer only)
            r'answer is:?\s*(-?\d[\d,]*)',  # Common pattern (integer only)
            r'(-?\d[\d,]*)\s*$',  # Last integer at the end of text
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).replace(',', '').strip()
        
        # Fallback: extract last integer in text (no decimals)
        numbers = re.findall(r'-?\d[\d,]*', text)
        if numbers:
            return numbers[-1].replace(',', '').strip()
    
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
            
    Raises:
        ValueError: If predictions and targets have different lengths
        
    Examples:
        >>> compute_exact_match(['42', '100'], ['42', '99'])
        {'exact_match': 0.5, 'correct': 1, 'total': 2}
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Predictions ({len(predictions)}) and targets ({len(targets)}) "
            "must have the same length"
        )
    
    correct = 0
    for pred, target in zip(predictions, targets):
        # Normalize both strings for comparison
        pred_norm = str(pred).strip().lower()
        target_norm = str(target).strip().lower()
        
        if pred_norm == target_norm:
            correct += 1
    
    em_score = correct / len(predictions) if len(predictions) > 0 else 0.0
    
    return {
        "exact_match": em_score,
        "correct": correct,
        "total": len(predictions)
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
    
    return f"EM = {em:.4f} ({correct}/{total})"
