"""ROI-based method using document contour alignment and template cropping."""

import numpy as np
from typing import Dict

from benchmarks.registry import register_method
from src.preprocessing.alignment import crop_document_contour
from src.roi.extractor import ROIExtractor
from src.ocr.engine import OCREngine

_ENGINE = OCREngine()
_EXTRACTOR = ROIExtractor()

@register_method("roi_based_vietocr")
def method_roi_vietocr(image: np.ndarray, doc_type: str) -> Dict[str, str]:
    """Align image, extract ROIs, and run VietOCR on each crop."""
    # 1. Align document (crop to standard CCCD size)
    aligned = crop_document_contour(image, target_w=856, target_h=540)
    
    # 2. Extract ROIs
    rois = _EXTRACTOR.extract_rois(aligned, doc_type)
    
    fields = {}
    # 3. Process each crop
    for field_name, crop in rois.items():
        if crop is None or crop.size == 0:
            continue
            
        # We can use recognize() which tries RapidOCR, but we want VietOCR explicitly for Vietnamese text
        # Since VietOCR handles diacritics better.
        # But OCREngine doesn't have a recognize_vietocr for a single crop that isn't line-based.
        # However, recognize_lines_vietocr takes an image and returns a list of lines.
        lines = _ENGINE.recognize_lines_vietocr(crop)
        text = " ".join(lines).strip()
        
        # Minor cleanups for specific fields (could use normalizer, but let's keep it simple)
        if field_name == "id_number":
            text = text.replace(" ", "")
            
        fields[field_name] = text
        
    return fields


