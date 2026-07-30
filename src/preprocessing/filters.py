"""Image enhancement and filtering operations for OCR optimization."""

from typing import Optional, Dict, Any
import cv2
import numpy as np


def enhance_for_ocr(
    crop: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Enhance image ROI crop for OCR processing.

    Applies grayscale conversion, scaling, denoising, and adaptive thresholding.
    """
    if crop is None or crop.size == 0:
        return crop

    cfg = config or {}
    scale_factor = cfg.get("scale_factor", 2.0)
    apply_denoise = cfg.get("denoise", True)
    apply_threshold = cfg.get("adaptive_threshold", True)

    processed = crop.copy()

    # 1. Grayscale conversion
    if len(processed.shape) == 3:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # 2. Resize by scale factor
    if scale_factor > 1.0:
        h, w = processed.shape[:2]
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 3. Denoising
    if apply_denoise:
        processed = cv2.fastNlMeansDenoising(processed, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # 4. Adaptive Thresholding
    if apply_threshold:
        processed = cv2.adaptiveThreshold(
            processed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

    return processed
