# Rejected OCR Approaches & Methods

This document tracks OCR methods and parsing pipelines that were implemented, benchmarked, and subsequently rejected due to low accuracy or poor scalability. Keeping a record of these prevents us from attempting the same failed ideas in the future.

## 1. ROI-based Extraction (`roi_based.py`)
- **Concept**: Use fixed bounding boxes (Regions of Interest) based on template matching or hardcoded coordinates to extract specific fields (ID, Name, DOB).
- **Result**: `0.0% Exact Match Accuracy` | `~97% CER`
- **Reason for Rejection**: Highly sensitive to document alignment, rotation, and slight variations in card layouts. Cropping coordinates almost always missed the text or cropped it partially.

## 2. Enhanced Preprocessing (`enhanced_preprocessing.py`)
- **Concept**: Apply aggressive image preprocessing (adaptive thresholding, morphological operations, contrast enhancement) before sending to OCR to improve recognition.
- **Result**: `~24% Exact Match Accuracy` | `~56% CER`
- **Reason for Rejection**: VietOCR (and most modern deep learning OCR models) works best on natural RGB images. Heavy binarization and morphological transformations destroyed anti-aliasing features that the CNN expected, significantly degrading accuracy rather than helping.

## 3. Layout-Aware Parser (`layout_aware.py`)
- **Concept**: Try to use simple spatial clustering based on Y-coordinates to group text lines into logical blocks (e.g. finding the 'Address block').
- **Result**: `~31% Exact Match Accuracy` | `~54% CER`
- **Reason for Rejection**: Fails completely on multi-line text (like `place_of_origin` and `place_of_residence` overlapping in Y-space due to skew). The clustering logic was too brittle and lacked contextual understanding of labels vs values.

## 4. Graph-Based Parser V1 (`graph_based.py`)
- **Concept**: Build a directed graph of all OCR nodes and use spatial proximity to link labels to values.
- **Result**: `~40% Exact Match Accuracy` | `~43% CER`
- **Reason for Rejection**: Replaced by the superior `best_practice_vietocr.py` (which uses `layout_graph_parser.py`). The initial `graph_based.py` relied on simplistic closest-node heuristics that were often tricked by dense text blocks.

---
**Key Takeaway (Rule of Thumb for this project):**
- **DO NOT** attempt heuristic-based parsers with complex if/else rules or regex regex ("rule explosion").
- **DO NOT** use classical CV preprocessing (binarization/thresholding) before VietOCR.
- **DO** rely on the Validation Layer (`validator.py`) for post-OCR correction using Dictionary Matching (RapidFuzz).
- **DO** focus on improving the base OCR Engine's recognition capability (Fine-tuning) rather than masking errors with parser logic.
