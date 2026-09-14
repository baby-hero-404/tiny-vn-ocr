"""Unified OCR Engine combining RapidOCR primary and Tesseract fallback."""

from typing import Tuple, Optional, Dict, Any, List
import logging
import re
import unicodedata
import numpy as np
import cv2

from src.preprocessing.filters import enhance_image_for_detection

logger = logging.getLogger(__name__)

_FIXED_BANNERS = (
    "cong hoa xa hoi chu nghia viet nam",
    "doc lap tu do hanh phuc",
    "can cuoc cong dan",
    "can cuoc",
    "citizen identity card",
    "identity card",
    "dac diem nhan dang",
    "ngon tro trai",
    "ngon tro phai",
    "dau vet rieng",
)

_PURE_LABELS = (
    "ho va ten", "full name", "ho ten",
    "ngay sinh", "date of birth",
    "gioi tinh", "sex",
    "quoc tich", "nationality",
    "que quan", "place of origin",
    "noi thuong tru", "place of residence", "thuong tru",
    "noi cu tru", "cu tru",
    "noi dang ky khai sinh", "khai sinh", "place of birth",
    "co gia tri den", "date of expiry",
    "ngay, thang, nam", "date, month, year",
    "cuc truong", "giam doc", "bo cong an"
)

def _strip_accents_fast(s: str) -> str:
    nfkd = unicodedata.normalize('NFD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

def _should_refine_with_vietocr(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) <= 2:
        return False
    # Pure numbers, dates, or symbols
    clean_no_digits = re.sub(r'[\d\s/.\-:,_#*<>I|]', '', stripped)
    if not clean_no_digits:
        return False
    norm = _strip_accents_fast(stripped).lower()
    # MRZ or chip noise
    if "<" in norm or "idvnm" in norm:
        return False
    # Fixed national banners
    if any(b in norm for b in _FIXED_BANNERS):
        return False
    # If it has a colon with substantive letters after it (inline value) -> refine!
    if ":" in stripped:
        val = stripped.split(":", 1)[1].strip()
        if len(re.sub(r'[\d\s/.\-]', '', val)) >= 3:
            return True
        return False
    # Pure labels
    for lbl in _PURE_LABELS:
        if norm == lbl or norm.startswith(lbl + " /") or norm.endswith("/ " + lbl):
            return False
    # Variable text (names, places, mixed address) -> refine with VietOCR!
    return True

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

_SHARED_VIETOCR_PREDICTOR = None
_SHARED_RAPIDOCR_ENGINE = None

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
        """Lazy-load RapidOCR engine instance as a singleton."""
        global _SHARED_RAPIDOCR_ENGINE
        if _SHARED_RAPIDOCR_ENGINE is not None:
            return _SHARED_RAPIDOCR_ENGINE

        if not self._rapidocr_initialized:
            if HAS_RAPIDOCR:
                try:
                    _SHARED_RAPIDOCR_ENGINE = RapidOCR(**self._rapidocr_kwargs)
                except Exception as e:
                    logger.warning(f"Failed to initialize RapidOCR: {e}")
                    _SHARED_RAPIDOCR_ENGINE = None
            self._rapidocr_initialized = True
        return _SHARED_RAPIDOCR_ENGINE

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

    def recognize_lines(self, image: np.ndarray) -> List[str]:
        """Extract full document text block using RapidOCR line detection."""
        if not HAS_RAPIDOCR or not self.rapidocr_engine:
            return []
            
        try:
            result, _ = self.rapidocr_engine(image)
            if not result:
                return []

            texts = []
            for item in result:
                if len(item) >= 3:
                    text = str(item[1]).strip()
                    if text:
                        texts.append(text)

            return texts
        except Exception as e:
            logger.debug(f"RapidOCR recognize_lines failed: {e}")
            return []

    def recognize_lines_vietocr(self, image: np.ndarray) -> List[str]:
        """
        2-Stage OCR Pipeline:
        1. RapidOCR detects text bounding boxes (fast & accurate).
        2. VietOCR (VGG_Seq2Seq) recognizes text from cropped boxes (handles diacritics).
        Falls back to RapidOCR lines if VietOCR is unavailable.
        """
        if not HAS_RAPIDOCR or not self.rapidocr_engine:
            return []

        has_vietocr = False
        try:
            import torch
            global _SHARED_VIETOCR_PREDICTOR
            if _SHARED_VIETOCR_PREDICTOR is None:
                from src.ocr.onnx_predictor import ONNXPredictor
                _SHARED_VIETOCR_PREDICTOR = ONNXPredictor('vgg_seq2seq')
            self.vietocr_predictor = _SHARED_VIETOCR_PREDICTOR
            has_vietocr = True
        except Exception as e:
            logger.debug(f"VietOCR unavailable ({e}), using RapidOCR lines.")

        if not has_vietocr:
            return self.recognize_lines(image)

        try:
            # Stage 1: Detection
            result, _ = self.rapidocr_engine(image)
            if not result:
                return []

            from PIL import Image

            texts = []
            for item in result:
                if len(item) >= 2:
                    rapid_text = str(item[1]).strip() if item[1] is not None else ""
                    text = rapid_text

                    box = item[0]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    x_coords = [p[0] for p in box]
                    y_coords = [p[1] for p in box]

                    padding_y_top = 6
                    padding_y_bottom = 4
                    padding_x = 3

                    xmin = max(0, int(min(x_coords)) - padding_x)
                    xmax = min(image.shape[1], int(max(x_coords)) + padding_x)
                    ymin = max(0, int(min(y_coords)) - padding_y_top)
                    ymax = min(image.shape[0], int(max(y_coords)) + padding_y_bottom)

                    if ymax > ymin and xmax > xmin and _should_refine_with_vietocr(rapid_text):
                        crop = image[ymin:ymax, xmin:xmax]
                        crop_enhanced = enhance_image_for_detection(crop)
                        crop_rgb = cv2.cvtColor(crop_enhanced, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(crop_rgb)

                        refined = self.vietocr_predictor.predict(pil_img)
                        if refined and len(refined.strip()) > 0:
                            text = refined.strip()

                    if text:
                        texts.append(text)

            return texts
        except Exception as e:
            logger.warning(f"VietOCR 2-stage recognition failed: {e}, falling back to RapidOCR")
            return self.recognize_lines(image)

    def recognize_lines_vietocr_with_bboxes(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        2-Stage OCR Pipeline that returns text along with bounding boxes.
        Falls back to RapidOCR with bboxes if VietOCR / Torch is unavailable.
        """
        if not HAS_RAPIDOCR or not self.rapidocr_engine:
            return []

        has_vietocr = False
        try:
            import torch
            global _SHARED_VIETOCR_PREDICTOR
            if _SHARED_VIETOCR_PREDICTOR is None:
                from src.ocr.onnx_predictor import ONNXPredictor
                _SHARED_VIETOCR_PREDICTOR = ONNXPredictor('vgg_seq2seq')
            self.vietocr_predictor = _SHARED_VIETOCR_PREDICTOR
            has_vietocr = True
        except Exception as e:
            logger.debug(f"VietOCR unavailable ({e}), using RapidOCR with bboxes.")

        if not has_vietocr:
            return self.recognize_lines_rapidocr_with_bboxes(image)

        try:
            # Stage 1: Detection
            result, _ = self.rapidocr_engine(image)
            if not result:
                return []

            from PIL import Image
            elements = []
            for item in result:
                if len(item) >= 2:
                    rapid_text = str(item[1]).strip() if item[1] is not None else ""
                    text = rapid_text

                    box = item[0]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    x_coords = [p[0] for p in box]
                    y_coords = [p[1] for p in box]

                    padding_y_top = 6
                    padding_y_bottom = 4
                    padding_x = 3

                    xmin = max(0, int(min(x_coords)) - padding_x)
                    xmax = min(image.shape[1], int(max(x_coords)) + padding_x)
                    ymin = max(0, int(min(y_coords)) - padding_y_top)
                    ymax = min(image.shape[0], int(max(y_coords)) + padding_y_bottom)

                    crop = None
                    if ymax > ymin and xmax > xmin:
                        crop = image[ymin:ymax, xmin:xmax]

                        if _should_refine_with_vietocr(rapid_text):
                            crop_enhanced = enhance_image_for_detection(crop)
                            crop_rgb = cv2.cvtColor(crop_enhanced, cv2.COLOR_BGR2RGB)
                            pil_img = Image.fromarray(crop_rgb)

                            refined = self.vietocr_predictor.predict(pil_img)
                            if refined and len(refined.strip()) > 0:
                                text = refined.strip()

                    if text:
                        tight_xmin = min(x_coords)
                        tight_xmax = max(x_coords)
                        tight_ymin = min(y_coords)
                        tight_ymax = max(y_coords)

                        elements.append({
                            "text": text,
                            "bbox": [tight_xmin, tight_ymin, tight_xmax, tight_ymax],
                            "center": [
                                (tight_xmin + tight_xmax) / 2,
                                (tight_ymin + tight_ymax) / 2
                            ],
                            "height": tight_ymax - tight_ymin,
                            "crop": crop
                        })

            return elements
        except Exception as e:
            logger.warning(f"VietOCR 2-stage recognition with bboxes failed: {e}, falling back to RapidOCR")
            return self.recognize_lines_rapidocr_with_bboxes(image)

    def recognize_lines_rapidocr_with_bboxes(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs ONLY RapidOCR to detect boxes and read text. Fast and memory efficient.
        Returns a list of dicts: {"text": str, "bbox": [xmin, ymin, xmax, ymax], "center": [cx, cy], "crop": crop}
        """
        if not HAS_RAPIDOCR or not self.rapidocr_engine:
            return []

        try:
            result, _ = self.rapidocr_engine(image)
            if not result:
                return []

            elements = []
            for item in result:
                if len(item) >= 3:
                    box = item[0]
                    text = item[1]
                    score = float(item[2])

                    x_coords = [p[0] for p in box]
                    y_coords = [p[1] for p in box]

                    tight_xmin = min(x_coords)
                    tight_xmax = max(x_coords)
                    tight_ymin = min(y_coords)
                    tight_ymax = max(y_coords)

                    # Store padded crop
                    padding_y_top = 6
                    padding_y_bottom = 4
                    padding_x = 3

                    xmin = max(0, int(tight_xmin) - padding_x)
                    xmax = min(image.shape[1], int(tight_xmax) + padding_x)
                    ymin = max(0, int(tight_ymin) - padding_y_top)
                    ymax = min(image.shape[0], int(tight_ymax) + padding_y_bottom)

                    crop = None
                    if ymax > ymin and xmax > xmin:
                        crop = image[ymin:ymax, xmin:xmax]

                    elements.append({
                        "text": text,
                        "score": score,
                        "bbox": [tight_xmin, tight_ymin, tight_xmax, tight_ymax],
                        "center": [
                            (tight_xmin + tight_xmax) / 2,
                            (tight_ymin + tight_ymax) / 2
                        ],
                        "height": tight_ymax - tight_ymin,
                        "crop": crop
                    })
            return elements
        except Exception as e:
            logger.error(f"RapidOCR recognize_lines failed: {e}")
            return []

    def recognize_crop_vietocr(self, crop: np.ndarray) -> str:
        """Recognize text from a single crop image using VietOCR ONNX model."""
        if crop is None or crop.size == 0:
            return ""
        try:
            import torch
            global _SHARED_VIETOCR_PREDICTOR
            if _SHARED_VIETOCR_PREDICTOR is None:
                from src.ocr.onnx_predictor import ONNXPredictor
                _SHARED_VIETOCR_PREDICTOR = ONNXPredictor('vgg_seq2seq')

            self.vietocr_predictor = _SHARED_VIETOCR_PREDICTOR

            from PIL import Image
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)

            text = self.vietocr_predictor.predict(pil_img)
            return text.strip()
        except Exception as e:
            logger.debug(f"VietOCR ONNX recognize_crop fallback to RapidOCR: {e}")
            text, _ = self.recognize_rapidocr(crop)
            return text.strip()

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
