"""Unit tests for ROI extraction module."""

import numpy as np
import pytest
from src.roi.extractor import ROIExtractor, extract_rois, DOCUMENT_TEMPLATES


def test_extract_rois_cccd():
    image = np.zeros((600, 900, 3), dtype=np.uint8)
    rois = extract_rois(image, "cccd")

    assert "id_number" in rois
    assert "full_name" in rois
    assert "date_of_birth" in rois
    assert rois["id_number"].shape[0] > 0
    assert rois["id_number"].shape[1] > 0


def test_extract_rois_driving_license():
    image = np.zeros((600, 900, 3), dtype=np.uint8)
    rois = extract_rois(image, "driving_license")

    assert "license_number" in rois
    assert "full_name" in rois
    assert "license_class" in rois


def test_extract_rois_unsupported_type():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        extract_rois(image, "unknown_doc")
