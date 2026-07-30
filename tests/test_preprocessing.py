"""Unit tests for preprocessing and alignment modules."""

import numpy as np
import cv2
from src.preprocessing.alignment import align_document
from src.preprocessing.filters import enhance_for_ocr


def test_align_document_with_none_template():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    aligned = align_document(image, template=None)
    assert aligned.shape == image.shape


def test_align_document_low_features_fallback():
    # Blank images with no keypoints
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    template = np.zeros((100, 100, 3), dtype=np.uint8)
    aligned = align_document(image, template=template)
    assert aligned.shape == image.shape


def test_align_document_valid_homography():
    # Create image with textured pattern
    template = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(template, (20, 20), (80, 80), (255, 255, 255), -1)
    cv2.circle(template, (150, 150), 30, (255, 255, 255), -1)

    # Shifted image
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (90, 90), (255, 255, 255), -1)
    cv2.circle(image, (160, 160), 30, (255, 255, 255), -1)

    aligned = align_document(image, template)
    assert aligned is not None
    assert aligned.shape[:2] == template.shape[:2]


def test_enhance_for_ocr():
    crop = np.ones((50, 100, 3), dtype=np.uint8) * 128
    enhanced = enhance_for_ocr(crop, config={"scale_factor": 2.0, "denoise": True, "adaptive_threshold": True})

    assert len(enhanced.shape) == 2  # Grayscale
    assert enhanced.shape[0] == 100  # 50 * 2
    assert enhanced.shape[1] == 200  # 100 * 2
