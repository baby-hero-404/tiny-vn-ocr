"""Best Practice pipeline combining Regex, Address Normalization, Business Rules, and Name Correction."""

import numpy as np
from typing import Dict, Any

from benchmarks.registry import register_method
from src.postprocessing.address_norm import normalize_gender_nationality, normalize_address
from src.postprocessing.cccd_rules import validate_and_fix_expiry
from src.postprocessing.vn_name_dict import correct_full_name
from src.postprocessing.layout_parser import layout_parse

# We need access to the raw OCR lines/bboxes
from src.ocr.engine import OCREngine

_ENGINE = OCREngine()

@register_method("best_practice_vietocr")
def method_best_practice_vietocr(image: np.ndarray, doc_type: str) -> Dict[str, str]:
    """
    Full pipeline (Layout-Aware Document Understanding):
    1. OCR with bboxes
    2. Layout Graph Parser (spatial extraction)
    3. Vietnamese surname correction
    4. CCCD business rules (expiry date cross-validation)
    5. Gender/Nationality normalization
    """
    h, w = image.shape[:2]
    if max(h, w) > 1600:
        scale = 1600.0 / max(h, w)
        import cv2
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 1. OCR with bboxes
    elements = _ENGINE.recognize_lines_vietocr_with_bboxes(image)
    
    # 2. Layout Graph Parser (spatial extraction)
    fields = layout_parse(elements, doc_type, image=image, ocr_engine=_ENGINE)
        
    # 3. Vietnamese surname auto-correction
    if "full_name" in fields:
        fields["full_name"] = correct_full_name(fields["full_name"])
        
    # 4. Apply CCCD Business Rules (expiry date cross-validation)
    if doc_type in ("cccd", "cccd_front"):
        fields = validate_and_fix_expiry(fields)
        
    # 5. Gender/Nationality normalization
    fields = normalize_gender_nationality(fields)
    
    # 6. Address normalization using Hierarchical DB
    if "place_of_origin" in fields:
        fields["place_of_origin"] = normalize_address(fields["place_of_origin"])
    if "place_of_residence" in fields:
        fields["place_of_residence"] = normalize_address(fields["place_of_residence"])
    if "place_of_birth" in fields:
        fields["place_of_birth"] = normalize_address(fields["place_of_birth"])
        
    return fields


_PIPELINE = None

@register_method("production_pipeline")
def method_production_pipeline(image: np.ndarray, doc_type: str) -> Dict[str, str]:
    """Production DocumentOCRPipeline with automatic side detection, normalization and validation."""
    global _PIPELINE
    if _PIPELINE is None:
        from src.pipeline import DocumentOCRPipeline
        _PIPELINE = DocumentOCRPipeline(ocr_engine=_ENGINE)
    resp = _PIPELINE.process(image, doc_type)
    return {k: str(v) for k, v in resp.fields.items() if v is not None}
