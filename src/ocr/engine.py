"""Unified OCR Engine combining RapidOCR primary and Tesseract fallback."""

from typing import Tuple, Optional, Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Try importing RapidOCR
try:
    from rapidocr_onnxruntime import RapidOCR

    HAS_RAPIDOCR = True
except Exception:
    HAS_RAPIDOCR = False

# Try importing pytesseract
try:
    import pytesseract

    HAS_PYTESSERACT = True
except Exception:
    HAS_PYTESSERACT = False


class OCREngine:
    """Unified OCR Engine with primary ONNX RapidOCR and Tesseract fallback."""

    def __init__(self, rapidocr_kwargs: Optional[Dict[str, Any]] = None):
        self._rapidocr_kwargs = rapidocr_kwargs or {}
        self._rapidocr = None
        self._rapidocr_initialized = False
        self._tesseract_available = False

        if HAS_PYTESSERACT:
            try:
                pytesseract.get_tesseract_version()
                self._tesseract_available = True
            except Exception:
                self._tesseract_available = False

    @property
    def rapidocr_engine(self):
        """Lazy-load RapidOCR engine instance."""
        if self._rapidocr is not None:
            return self._rapidocr

        if not self._rapidocr_initialized:
            if HAS_RAPIDOCR:
                try:
                    self._rapidocr = RapidOCR(**self._rapidocr_kwargs)
                except Exception as e:
                    logger.warning(f"Failed to initialize RapidOCR: {e}")
                    self._rapidocr = None
            self._rapidocr_initialized = True
        return self._rapidocr

    @property
    def available_engines(self) -> Dict[str, bool]:
        return {
            "rapidocr": HAS_RAPIDOCR and (self.rapidocr_engine is not None or self._rapidocr is not None),
            "tesseract": self._tesseract_available,
        }

    def recognize_rapidocr(self, crop: np.ndarray) -> Tuple[str, float]:
        """Recognize text using RapidOCR engine."""
        engine = self.rapidocr_engine
        if engine is None:
            return "", 0.0

        try:
            result, _ = engine(crop)
            if not result:
                return "", 0.0

            texts = []
            scores = []
            for item in result:
                if len(item) >= 3:
                    text = str(item[1]).strip()
                    score = float(item[2])
                    if text:
                        texts.append(text)
                        scores.append(score)

            if not texts:
                return "", 0.0

            combined_text = " ".join(texts)
            avg_score = sum(scores) / len(scores) if scores else 0.0
            return combined_text, avg_score
        except Exception as e:
            logger.debug(f"RapidOCR recognition failed: {e}")
            return "", 0.0

    def recognize_tesseract(
        self,
        crop: np.ndarray,
        field_type: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Recognize text using Tesseract fallback engine."""
        if not self._tesseract_available:
            return "", 0.0

        try:
            config = "--psm 7"
            if field_type in ("id_number", "license_number", "date_of_birth", "date_of_issue", "date_of_expiry"):
                config += " -c tessedit_char_whitelist=0123456789/"

            text = pytesseract.image_to_string(crop, config=config, lang="vie+eng").strip()
            if not text:
                text = pytesseract.image_to_string(crop, config="--psm 7").strip()

            confidence = 0.7 if text else 0.0
            return text, confidence
        except Exception as e:
            logger.debug(f"Tesseract recognition failed: {e}")
            return "", 0.0

    def recognize(
        self,
        crop: np.ndarray,
        field_type: Optional[str] = None,
        min_confidence: float = 0.5,
    ) -> Tuple[str, float, str]:
        """Recognize text from image crop using dual OCR engines.

        Returns: (recognized_text, confidence_score, engine_name_used)
        """
        if crop is None or crop.size == 0:
            return "", 0.0, "none"

        # 1. Try RapidOCR primary
        text, score = self.recognize_rapidocr(crop)
        if text and score >= min_confidence:
            return text, score, "rapidocr"

        # 2. Try Tesseract fallback
        fallback_text, fallback_score = self.recognize_tesseract(crop, field_type)
        if fallback_text:
            return fallback_text, fallback_score, "tesseract"

        if text:
            return text, score, "rapidocr"

        return "", 0.0, "none"
