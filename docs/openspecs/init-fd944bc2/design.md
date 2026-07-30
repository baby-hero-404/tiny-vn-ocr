# Technical Design: Tiny VN OCR Implementation

## Architectural Overview

Tiny VN OCR is designed as a modular, lightweight Python-based OCR solution for Vietnamese identity documents (CCCD, GPLX, Vehicle Registration). It operates under strict resource constraints (~2GB RAM, CPU-only).

Rather than performing full-image OCR using heavy Vision/Transformer models, Tiny VN OCR follows a template-driven approach:
`Input Image` -> `Quality Check & Alignment (ORB Homography)` -> `ROI Extraction` -> `Dual OCR Engine (RapidOCR / Tesseract)` -> `Post-Processing & Validation` -> `Unified Pydantic Schema JSON`.

```
                        +----------------------+
                        |     Input Image      |
                        +----------+-----------+
                                   |
                                   v
                        +----------------------+
                        | OpenCV Preprocessing |
                        |  & Homography Align  |
                        +----------+-----------+
                                   |
                                   v
                        +----------------------+
                        |   ROI Extractor      |
                        | (CCCD, GPLX, Cavet)  |
                        +----------+-----------+
                                   |
                                   v
                        +----------------------+
                        |  Dual OCR Engine     |
                        | RapidOCR (Primary)   |
                        | Tesseract (Fallback) |
                        +----------+-----------+
                                   |
                                   v
                        +----------------------+
                        | Post-processing &    |
                        | Regex/Fuzzy Validate |
                        +----------+-----------+
                                   |
                                   v
                        +----------------------+
                        | Unified JSON Schema  |
                        +----------------------+
```

## Core Modules and File Design

### 1. Data Schemas (`src/schemas/document.py`)
- Define Pydantic V2 models for specific fields:
  - `CCCDFields`: `id_number`, `full_name`, `date_of_birth`, `gender`, `nationality`, `place_of_origin`, `place_of_residence`, `date_of_issue`.
  - `GPLXFields`: `license_number`, `full_name`, `date_of_birth`, `nationality`, `license_class`, `date_of_issue`, `date_of_expiry`.
  - `VehicleRegistrationFields`: `license_plate`, `owner_name`, `owner_address`, `brand`, `model`, `color`, `engine_number`, `chassis_number`, `vehicle_type`, `date_of_issue`.
- Define unified response schemas:
  - `IdentityFields`, `VehicleFields`, `LicenseFields`
  - `DocumentResponse`: `document_type`, `confidence`, `fields`, `metadata`.

### 2. Preprocessing & Alignment (`src/preprocessing/`)
- `src/preprocessing/alignment.py`:
  - `align_document(image: np.ndarray, template: np.ndarray) -> np.ndarray`: Uses OpenCV `ORB_create()`, `BFMatcher(NORM_HAMMING, crossCheck=True)`, `cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)`, and `cv2.warpPerspective`. Returns aligned image or fallback on failure (< 4 matches).
- `src/preprocessing/filters.py`:
  - `enhance_for_ocr(crop: np.ndarray, config: dict) -> np.ndarray`: Applies grayscale conversion, `cv2.resize` (scale factor ~2.0), `cv2.fastNlMeansDenoising` or Gaussian blur, and `cv2.adaptiveThreshold` for ROI OCR optimization.

### 3. ROI Extractor (`src/roi/extractor.py`)
- Pre-defined template bounding boxes normalized `(ymin, xmin, ymax, xmax)` for `cccd_front`, `gplx_front`, `vehicle_registration_front`.
- `ROIExtractor`: Methods to crop field sub-images using NumPy slicing from aligned document images.

### 4. OCR Engine (`src/ocr/engine.py`)
- `OCREngine`: Unified interface.
  - Primary: `RapidOCR` initialized via `rapidocr_onnxruntime`. Uses CPU ONNX runtime.
  - Fallback: `pytesseract.image_to_string` with custom `config='--psm 7'` or whitelist parameters.
  - `recognize(crop: np.ndarray, field_type: str) -> Tuple[str, float]`: Tries RapidOCR first; if confidence < threshold or empty, falls back to Tesseract.

### 5. Post-Processing & Validation (`src/postprocessing/normalizer.py`)
- `normalize_text(text: str, field_type: str) -> str`:
  - Strips unwanted whitespace, noise symbols.
  - Digit confusion replacement map: `{'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8'}` when field requires numeric output (e.g. ID number).
- `validate_field(text: str, field_type: str) -> bool`:
  - CCCD ID regex: `^\d{12}$`
  - Date regex: `^\d{2}/\d{2}/\d{4}$`
  - GPLX Class enum check: `A1, A, B1, B2, C, D, E, F`
- `fuzzy_correct(text: str, dictionary: list) -> str`: Uses `rapidfuzz` for dictionary distance correction on common Vietnamese text values.

### 6. Pipeline Orchestration (`src/pipeline.py`)
- `DocumentOCRPipeline`:
  - `process(image: np.ndarray, document_type: str) -> DocumentResponse`
  - Controls timing metadata, alignment execution, ROI iteration, field OCR, normalization, validation, and Pydantic schema construction.

### 7. REST API (`src/api/app.py`)
- FastAPI app instance with Uvicorn.
- Endpoints:
  - `POST /api/v1/ocr`: Accepts `file: UploadFile` and `document_type: str`.
  - `GET /health`: Returns system status and available OCR engines.

### 8. Test Suite (`tests/`)
- `test_schemas.py`: Unit tests for Pydantic models and serialization.
- `test_preprocessing.py`: Unit tests for alignment functions and image filters.
- `test_roi.py`: Tests for template loading and region cropping.
- `test_ocr.py`: Mocked and functional tests for OCR engines.
- `test_normalizer.py`: Unit tests for text cleanup, character mapping, regex, and fuzzy matching.
- `test_pipeline.py`: Integration tests for end-to-end processing.
- `test_api.py`: FastAPI `TestClient` tests for REST API endpoints.

## Trade-offs & Resource Management

- **Memory Efficiency**: RapidOCR ONNX models are initialized once at service startup to prevent repeated model loading overhead. Memory peak is constrained under ~700MB.
- **CPU Performance**: Processing ROIs instead of full images dramatically reduces OCR inference time (<500ms total processing time).
- **Alignment Fallback**: If keypoint alignment fails (e.g. severe tilt/blur), the pipeline proceeds with unaligned ROI cropping or defaults gracefully rather than throwing a server 500 error.
