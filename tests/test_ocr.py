"""Unit tests for OCR engine."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from src.ocr.engine import OCREngine


def test_ocr_engine_rapidocr_primary():
    engine = OCREngine()
    engine._rapidocr_available = True
    mock_rapid = MagicMock()
    mock_rapid.return_value = (
        [[[0, 0], "079123456789", 0.95]],
        10.0,
    )
    engine._rapidocr = mock_rapid

    crop = np.zeros((30, 100, 3), dtype=np.uint8)
    text, score, engine_name = engine.recognize(crop, field_type="id_number")

    assert text == "079123456789"
    assert score == 0.95
    assert engine_name == "rapidocr"


def test_ocr_engine_tesseract_fallback():
    engine = OCREngine()
    engine._rapidocr_available = False
    engine._tesseract_available = True

    crop = np.zeros((30, 100, 3), dtype=np.uint8)

    with patch("pytesseract.image_to_string", return_value="NGUYEN VAN A"):
        text, score, engine_name = engine.recognize(crop, field_type="full_name")
        assert text == "NGUYEN VAN A"
        assert engine_name == "tesseract"


def test_ocr_engine_empty_crop():
    engine = OCREngine()
    text, score, engine_name = engine.recognize(np.array([]))
    assert text == ""
    assert score == 0.0
    assert engine_name == "none"
