"""Method using best_practice_vietocr baseline but with Validation Layer applied."""

import numpy as np
from typing import Dict

from benchmarks.registry import register_method
from benchmarks.methods.best_practice_vietocr import method_best_practice_vietocr
from src.postprocessing.validator import FieldValidator

@register_method("validated_best_practice_vietocr")
def method_validated_best_practice_vietocr(image: np.ndarray, doc_type: str) -> Dict[str, str]:
    """Extract text using best_practice_vietocr, then apply the FieldValidator."""
    
    # 1. Extraction (Baseline)
    fields = method_best_practice_vietocr(image, doc_type)
    
    # 2. Validation
    validated_fields = FieldValidator.validate_all(fields)
    
    return validated_fields
