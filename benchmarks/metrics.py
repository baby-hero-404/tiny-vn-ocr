"""Metrics calculation for OCR benchmarking."""

from typing import Dict, Tuple
from rapidfuzz import distance

def calculate_cer(truth: str, pred: str) -> float:
    """Calculate Character Error Rate (Levenshtein distance / max length)."""
    if not truth and not pred:
        return 0.0
    if not truth:
        return 1.0
    dist = distance.Levenshtein.distance(truth, pred)
    return min(1.0, dist / max(len(truth), 1))

def evaluate_predictions(truth_fields: Dict[str, str], pred_fields: Dict[str, str], fields_to_check: list[str]) -> Tuple[int, int, float]:
    """
    Evaluate predicted fields against truth fields.
    
    Returns:
        Tuple of (total_valid_fields, exact_matches, total_cer_sum)
    """
    total_fields = 0
    exact_matches = 0
    total_cer = 0.0
    
    for field in fields_to_check:
        t_val = str(truth_fields.get(field, "")).strip()
        p_val = str(pred_fields.get(field, "")).strip()
        
        if not t_val:  # Skip empty ground truth fields
            continue
            
        total_fields += 1
        
        if t_val == p_val:
            exact_matches += 1
            
        cer = calculate_cer(t_val, p_val)
        total_cer += cer
        
    return total_fields, exact_matches, total_cer
