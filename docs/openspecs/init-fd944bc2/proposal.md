# Proposal: Tiny VN OCR Implementation

## Why this change is needed

The Tiny VN OCR repository currently contains architecture documentation (`docs/architecture.md`) and a placeholder `README.md`, but lacks the underlying implementation. 
Building a lightweight, high-accuracy Vietnamese identity document (CCCD, GPLX, Vehicle Registration) OCR system optimized for resource-constrained environments (RAM ~2GB, CPU-only) requires implementing an end-to-end pipeline. 

The proposal establishes a template-based ROI extraction pipeline using OpenCV feature alignment, a dual OCR engine (RapidOCR primary, Tesseract fallback), rule-based post-processing/normalization, Pydantic unified schema validation, a FastAPI web service, and a comprehensive test suite.

## What changes at a high level

- **Dependencies & Setup**: Define project requirements in `requirements.txt` and `pyproject.toml` including `opencv-python-headless`, `rapidocr-onnxruntime`, `pytesseract`, `pydantic`, `fastapi`, `uvicorn`, `rapidfuzz`, and `pytest`.
- **Unified Data Schemas (`src/schemas/document.py`)**: Implement Pydantic data models for CCCD, GPLX (driving license), and vehicle registration documents, along with the unified JSON response format.
- **Image Preprocessing & Alignment (`src/preprocessing/`)**: 
  - `alignment.py`: Implement ORB feature matching, RANSAC homography estimation, and perspective warping for template alignment with fallback mechanisms.
  - `filters.py`: Implement image enhancement filters (grayscale conversion, scaling, denoising, adaptive thresholding).
- **ROI Extraction (`src/roi/extractor.py`)**: Define coordinate templates for supported document types and implement ROI crop extraction routines.
- **Unified OCR Engine (`src/ocr/engine.py`)**: Create a unified OCR interface wrapping RapidOCR (ONNX runtime) as primary engine and Tesseract as fallback, supporting PSM and character whitelists.
- **Post-Processing & Validation (`src/postprocessing/normalizer.py`)**: Implement character substitution rules (e.g. `O` -> `0`, `S` -> `5`), regex validation for ID numbers and dates, and fuzzy matching for Vietnamese strings.
- **Pipeline Orchestration (`src/pipeline.py`)**: Build `DocumentOCRPipeline` to execute alignment, ROI cropping, OCR recognition, text normalization, and schema validation.
- **REST API (`src/api/app.py`)**: Expose FastAPI HTTP endpoints (`/api/v1/ocr` and `/health`) for document image submission and health checks.
- **Testing Suite (`tests/`)**: Add unit and integration tests verifying alignment, ROI extraction, OCR engines, normalization, schema validation, and API routes.
- **Documentation (`README.md`)**: Update README with installation instructions, local running steps, API usage, and architecture overview.
