"""Method testing custom preprocessing before OCR."""

import cv2
import numpy as np
from typing import Dict

from benchmarks.registry import register_method
from src.ocr.engine import OCREngine
from src.postprocessing.parser import parse_document

_ENGINE = OCREngine()

def custom_preprocessing(image: np.ndarray) -> np.ndarray:
    """Apply heavy CLAHE and unsharp masking, specifically tuned for VietOCR."""
    processed = image.copy()
    
    # Resize if too large
    max_side = 1600
    h, w = processed.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        processed = cv2.resize(processed, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Grayscale
    if len(processed.shape) == 3:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        gray = processed

    # Binarization using adaptive thresholding directly on gray
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    
    # Morphological opening to remove small noise (like CCCD background patterns)
    kernel = np.ones((2, 2), np.uint8)
    clean_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Convert back to BGR so RapidOCR doesn't complain about channels
    bgr = cv2.cvtColor(clean_binary, cv2.COLOR_GRAY2BGR)
    
    return bgr

@register_method("enhanced_preprocessing_vietocr")
def method_enhanced_vietocr(image: np.ndarray, doc_type: str) -> Dict[str, str]:
    """Apply custom heavy binarization, then 2-stage VietOCR."""
    enhanced = custom_preprocessing(image)
    # We use VietOCR here since binarization might strip colors but keeps shapes
    lines = _ENGINE.recognize_lines_vietocr(enhanced)
    fields = parse_document(lines, doc_type)
    return fields


