# Specifications: Tiny VN OCR Implementation

## Requirement: Unified Document Data Schemas

### Scenario: Valid CCCD Document Field Parsing
- **WHEN** raw OCR text dictionary for CCCD is passed to `CCCDFields` Pydantic model
- **THEN** it validates the 12-digit ID number, date format (`DD/MM/YYYY`), gender enum (`NAM` / `NU`), and constructs a valid `DocumentResponse` with `document_type="cccd"`.

### Scenario: Valid Driving License (GPLX) Field Parsing
- **WHEN** raw OCR text dictionary for GPLX is passed to `GPLXFields` Pydantic model
- **THEN** it validates the license number, license class (e.g. `A1`, `B2`), date fields, and returns a valid `DocumentResponse` with `document_type="driving_license"`.

### Scenario: Valid Vehicle Registration Field Parsing
- **WHEN** raw OCR text dictionary for Vehicle Registration is passed to `VehicleRegistrationFields` Pydantic model
- **THEN** it validates license plate, engine number, chassis number, and returns a valid `DocumentResponse` with `document_type="vehicle_registration"`.

---

## Requirement: Image Preprocessing & Alignment

### Scenario: Template Alignment using ORB Homography
- **WHEN** a document image and template reference image are processed by `align_document`
- **THEN** it detects ORB keypoints, matches features, calculates RANSAC homography matrix, and warps the input image to match the template coordinates.

### Scenario: Graceful Fallback on Low-Feature Image Alignment
- **WHEN** ORB feature matching yields insufficient valid matches (< 4 keypoints)
- **THEN** the alignment engine falls back to original input geometry or contour-based bounds without raising unhandled exceptions.

### Scenario: ROI Image Filtering for OCR Optimization
- **WHEN** an ROI image crop is processed by `enhance_for_ocr`
- **THEN** it converts the crop to grayscale, resizes by 2x, applies denoising, and adaptive thresholding to maximize OCR recognition accuracy.

---

## Requirement: ROI Region Extraction

### Scenario: Multi-Field ROI Extraction
- **WHEN** an aligned image and document template definition (CCCD, GPLX, Vehicle Registration) are passed to `extract_rois`
- **THEN** it extracts separate cropped image arrays for each specified document field (e.g. `id_number`, `full_name`, `date_of_birth`).

---

## Requirement: Dual OCR Engine Execution

### Scenario: Primary OCR Recognition with RapidOCR
- **WHEN** an image crop is passed to `OCREngine.recognize` with primary engine settings
- **THEN** RapidOCR processes the image crop using ONNX runtime and returns recognized text along with a confidence score.

### Scenario: Fallback OCR Recognition with Tesseract
- **WHEN** RapidOCR engine is unavailable or returns text with confidence below threshold
- **THEN** `OCREngine` automatically invokes Tesseract OCR configured with line-mode PSM (e.g. PSM 7) and target whitelists.

---

## Requirement: Text Post-Processing & Normalization

### Scenario: Character Substitution Normalization
- **WHEN** raw OCR text for numeric fields (e.g. ID number) contains common character confusions like `O` for `0` or `S` for `5`
- **THEN** `normalize_field` applies substitution mapping (`O->0`, `I->1`, `S->5`) to produce clean digit sequences.

### Scenario: Regex Pattern Validation & Fuzzy Matching
- **WHEN** dirty or misread text for names, dates, or ID numbers is normalized
- **THEN** regex checks validate string format and fuzzy dictionary matching resolves diacritic/spelling errors against standard Vietnamese vocabularies.

---

## Requirement: End-to-End Pipeline Execution

### Scenario: Full Document Pipeline Processing
- **WHEN** an image file bytes and document type parameter are submitted to `DocumentOCRPipeline.process`
- **THEN** it performs alignment, ROI cropping, OCR recognition, text post-processing, and schema formatting within ~700MB peak memory constraints on CPU.

---

## Requirement: REST API Endpoints

### Scenario: POST /api/v1/ocr Endpoint Success
- **WHEN** a multipart POST request with a valid document image and `document_type` query parameter is sent to `/api/v1/ocr`
- **THEN** the API returns HTTP 200 with the unified JSON response payload containing fields, confidence score, and execution metadata.

### Scenario: GET /health Endpoint Check
- **WHEN** a GET request is sent to `/health`
- **THEN** the server returns HTTP 200 OK with `{"status": "healthy"}`.
