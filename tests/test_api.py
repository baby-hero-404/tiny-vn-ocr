"""Integration tests for FastAPI REST API endpoints."""

from unittest.mock import patch
import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "engines" in data


def test_ocr_endpoint_success():
    img = np.zeros((300, 450, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = io.BytesIO(img_encoded.tobytes())

    with patch("src.ocr.engine.OCREngine.recognize", return_value=("079123456789", 0.95, "rapidocr")):
        response = client.post(
            "/api/v1/ocr?document_type=cccd",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["document_type"] == "cccd"
    assert "fields" in data
    assert "metadata" in data


def test_ocr_endpoint_invalid_file_type():
    response = client.post(
        "/api/v1/ocr?document_type=cccd",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_ocr_endpoint_invalid_document_type():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = io.BytesIO(img_encoded.tobytes())

    response = client.post(
        "/api/v1/ocr?document_type=invalid_doc",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
