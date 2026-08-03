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


def evaluate_field_wise(truth_fields: Dict[str, str], pred_fields: Dict[str, str], fields_to_check: list[str]) -> Dict[str, Dict[str, float]]:
    """Per-field evaluation: for each field return exact_match (0/1) and CER.
    
    Returns:
        Dict mapping field_name -> {"exact_match": 0|1, "cer": float, "present": bool}
        Only includes fields that have a non-empty ground truth value.
    """
    result = {}
    for field in fields_to_check:
        t_val = str(truth_fields.get(field, "")).strip()
        if not t_val:
            continue
        p_val = str(pred_fields.get(field, "")).strip()
        cer = calculate_cer(t_val, p_val)
        result[field] = {
            "exact_match": 1 if t_val == p_val else 0,
            "cer": cer,
            "present": bool(p_val),
        }
    return result

def evaluate_critical_fields(truth_fields: Dict[str, str], pred_fields: Dict[str, str], critical_fields: list[str]) -> bool:
    """Check if all critical fields match exactly (Option B strict match)."""
    for field in critical_fields:
        t_val = str(truth_fields.get(field, "")).strip()
        if not t_val:
            continue
        p_val = str(pred_fields.get(field, "")).strip()
        if t_val != p_val:
            return False
    return True
