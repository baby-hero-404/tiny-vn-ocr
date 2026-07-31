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


def enhance_image_for_detection(image: np.ndarray) -> np.ndarray:
    """Enhance the full image BEFORE passing to OCR Detection (RapidOCR).
    
    Best practices for OCR preprocessing:
    1. Resize down if too large (to prevent small text from vanishing in RapidOCR's internal scaling).
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance faint text.
    3. Unsharp Masking to make edges crisp for VietOCR.
    """
    if image is None or image.size == 0:
        return image
        
    processed = image.copy()
    
    # 1. Resize if image is extremely high-res (e.g. > 2000px on any side)
    max_side = 2000
    h, w = processed.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        processed = cv2.resize(processed, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 2. CLAHE for Contrast Enhancement (Operates on L channel of LAB color space)
    if len(processed.shape) == 3:
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        processed = clahe.apply(processed)
        
    # 3. Unsharp Masking (Sharpening)
    gaussian = cv2.GaussianBlur(processed, (0, 0), 2.0)
    processed = cv2.addWeighted(processed, 1.5, gaussian, -0.5, 0)
    
    return processed
