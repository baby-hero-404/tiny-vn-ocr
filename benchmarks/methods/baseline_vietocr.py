"""Baseline method using RapidOCR for detection + VietOCR for recognition."""

import numpy as np
from typing import Dict

from benchmarks.registry import register_method
from src.ocr.engine import OCREngine
from src.postprocessing.parser import parse_document

# Singleton engine
_ENGINE = OCREngine()

@register_method("baseline_vietocr")
def method_vietocr(image: np.ndarray, doc_type: str) -> Dict[str, str]:
    """Extract text using RapidOCR (detection) + VietOCR (recognition), then parse."""
    lines = _ENGINE.recognize_lines_vietocr(image)
    fields = parse_document(lines, doc_type)
    return fields
