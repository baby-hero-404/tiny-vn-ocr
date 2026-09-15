"""Integration tests for Document OCR Pipeline."""

from unittest.mock import MagicMock
import numpy as np
import pytest
from src.pipeline import DocumentOCRPipeline
from src.schemas.document import DocumentResponse


def test_pipeline_process_cccd_mocked_ocr():
    mock_ocr = MagicMock()
    mock_ocr.recognize.side_effect = lambda crop, field_type=None: {
        "id_number": ("O79123456789", 0.95, "rapidocr"),
        "full_name": ("NGUYEN VAN A", 0.90, "rapidocr"),
        "date_of_birth": ("01.01.1995", 0.88, "rapidocr"),
        "gender": ("NAM", 0.92, "rapidocr"),
    }.get(field_type, ("", 0.0, "none"))

    pipeline = DocumentOCRPipeline(ocr_engine=mock_ocr)
    dummy_img = np.zeros((600, 900, 3), dtype=np.uint8)

    response = pipeline.process(dummy_img, "cccd")

    assert isinstance(response, DocumentResponse)
    assert response.document_type == "cccd"
    assert response.fields["id_number"] == "079123456789"
    assert response.fields["full_name"] in ("NGUYEN VAN A", "NGUYỄN VĂN A")
    assert response.fields["date_of_birth"] == "01/01/1995"
    assert response.fields["gender"] in ("NAM", "Nam")
    assert response.confidence > 0.8
    assert response.metadata.ocr_engine == "rapidocr"
    assert response.metadata.processing_time_ms >= 0
