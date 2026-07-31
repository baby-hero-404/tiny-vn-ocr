"""Document OCR pipeline orchestrator for Tiny VN OCR."""

from typing import Optional, Dict, Any
import time
import logging
import numpy as np

from src.preprocessing.alignment import align_document
from src.preprocessing.filters import enhance_for_ocr
from src.roi.extractor import ROIExtractor
from src.ocr.engine import OCREngine
from src.postprocessing.normalizer import normalize_field, validate_field
from src.schemas.document import (
    DocumentResponse,
    ProcessingMetadata,
    CCCDFields,
    GPLXFields,
    VehicleRegistrationFields,
)

logger = logging.getLogger(__name__)


class DocumentOCRPipeline:
    """End-to-end OCR pipeline for Vietnamese identity documents."""

    def __init__(self, ocr_engine: Optional[OCREngine] = None, roi_extractor: Optional[ROIExtractor] = None):
        self.ocr_engine = ocr_engine or OCREngine()
        self.roi_extractor = roi_extractor or ROIExtractor()

    def process(
        self,
        image: np.ndarray,
        document_type: str,
        template_image: Optional[np.ndarray] = None,
    ) -> DocumentResponse:
        """Process document image end-to-end and return unified JSON response."""
        start_time = time.perf_counter()

        if image is None or image.size == 0:
            raise ValueError("Invalid or empty input image provided.")

        doc_key = document_type.lower()
        if doc_key == "cccd_front":
            doc_key = "cccd"
        elif doc_key == "gplx_front":
            doc_key = "driving_license"
        elif doc_key == "vehicle_registration_front":
            doc_key = "vehicle_registration"

        # Check if we should use full-image heuristic parsing for CCCD
        if doc_key in ("cccd", "cccd_back"):
            from src.postprocessing.parser import parse_document
            
            # 1. Full Image OCR to extract raw lines
            # Sử dụng 2-stage OCR: RapidOCR (Detection) + VietOCR (Recognition) để giữ nguyên dấu Tiếng Việt
            lines = self.ocr_engine.recognize_lines_vietocr(image)
            engines_used = {"rapidocr"}
            
            extracted_fields = parse_document(lines, doc_key)
            field_confidences = [0.9] * len(extracted_fields)  # dummy confidence for full image OCR
        else:
            # Fallback to old ROI extraction for other documents
            # 1. Image alignment (ORB Homography)
            aligned_image = align_document(image, template_image)

            # 2. ROI Region Extraction
            rois = self.roi_extractor.extract_rois(aligned_image, doc_key)

            extracted_fields: Dict[str, Any] = {}
            field_confidences = []
            engines_used = set()

            # 3. Process each field ROI crop
            for field_name, crop in rois.items():
                if crop is None or crop.size == 0:
                    continue

                # Preprocessing filter for OCR optimization
                enhanced_crop = enhance_for_ocr(crop)

                # Dual OCR recognition
                raw_text, conf, engine_name = self.ocr_engine.recognize(enhanced_crop, field_type=field_name)

                if engine_name != "none":
                    engines_used.add(engine_name)

                # Post-processing normalization
                extracted_fields[field_name] = raw_text

                if conf > 0:
                    field_confidences.append(conf)

        # Normalize extracted fields
        validated_fields = {}
        for k, v in extracted_fields.items():
            validated_fields[k] = normalize_field(v, field_type=k)

        # Calculate overall confidence score
        overall_confidence = (
            sum(field_confidences) / len(field_confidences) if field_confidences else (0.8 if extracted_fields else 0.0)
        )
        overall_confidence = round(min(max(overall_confidence, 0.0), 1.0), 2)

        # Construct primary engine metadata tag
        primary_engine_tag = "rapidocr" if "rapidocr" in engines_used else ("tesseract" if "tesseract" in engines_used else "rapidocr")

        processing_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return DocumentResponse(
            document_type=doc_key,
            confidence=overall_confidence,
            fields=validated_fields,
            metadata=ProcessingMetadata(
                ocr_engine=primary_engine_tag,
                processing_time_ms=processing_time_ms,
            ),
        )
