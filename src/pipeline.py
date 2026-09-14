"""Document OCR pipeline orchestrator for Tiny VN OCR."""

from typing import Optional, Dict, Any, List
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

    @staticmethod
    def _get_cccd_side_affinity(elements: List[Dict[str, Any]]) -> int:
        """Calculate front vs back side affinity score. Higher score = more likely front side."""
        from src.postprocessing.parser import _strip_accents
        all_text = " ".join(el.get("text", "") for el in elements).lower()
        norm = _strip_accents(all_text)

        front_kw = ["ho va ten", "full name", "ho ten", "ngay sinh", "date of birth", "que quan", "place of origin", "quoc tich", "nationality", "gioi tinh", "sex", "so / no", "so:"]
        back_kw = ["khai sinh", "noi dang ky khai sinh", "cu tru", "noi cu tru", "cuc truong", "giam doc", "canh sat", "bo cong an", "ministry", "ngon tro", "dac diem"]

        front_score = sum(1 for kw in front_kw if kw in norm)
        back_score = sum(1 for kw in back_kw if kw in norm)
        return front_score - back_score

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

        # Check if we should use layout-aware best-practice extraction for CCCD / ID cards
        if doc_key in ("cccd", "cccd_front", "cccd_back", "id_card", "identity", "cccd_auto"):
            try:
                from src.postprocessing.layout_parser import layout_parse
                from src.postprocessing.vn_name_dict import correct_full_name
                from src.postprocessing.cccd_rules import validate_and_fix_expiry
                from src.postprocessing.address_norm import normalize_gender_nationality, normalize_address

                # 1. OCR with bboxes
                elements = self.ocr_engine.recognize_lines_vietocr_with_bboxes(image)
                
                # Auto-detect side if generic cccd or identity
                actual_side = doc_key
                if doc_key in ("cccd", "id_card", "identity", "cccd_auto"):
                    affinity = self._get_cccd_side_affinity(elements)
                    actual_side = "cccd_front" if affinity >= 0 else "cccd_back"

                # 2. Layout Graph Parser (spatial extraction)
                extracted_fields = layout_parse(elements, actual_side, image=image, ocr_engine=self.ocr_engine)
                
                # 3. Post-processing based on side
                if actual_side in ("cccd", "cccd_front"):
                    if "full_name" in extracted_fields:
                        extracted_fields["full_name"] = correct_full_name(extracted_fields["full_name"])
                    extracted_fields = validate_and_fix_expiry(extracted_fields)
                    extracted_fields = normalize_gender_nationality(extracted_fields)
                    if "place_of_origin" in extracted_fields:
                        extracted_fields["place_of_origin"] = normalize_address(extracted_fields["place_of_origin"])
                    if "place_of_residence" in extracted_fields:
                        extracted_fields["place_of_residence"] = normalize_address(extracted_fields["place_of_residence"])
                elif actual_side == "cccd_back":
                    if "place_of_birth" in extracted_fields:
                        extracted_fields["place_of_birth"] = normalize_address(extracted_fields["place_of_birth"])
                    if "place_of_residence" in extracted_fields:
                        extracted_fields["place_of_residence"] = normalize_address(extracted_fields["place_of_residence"])

                engines_used = {"rapidocr", "vietocr"}
                field_confidences = [0.95] * len(extracted_fields)
            except Exception as e:
                logger.warning(f"Layout graph parser failed, falling back to heuristic parser: {e}")
                from src.postprocessing.parser import parse_document
                lines = self.ocr_engine.recognize_lines_vietocr(image)
                engines_used = {"rapidocr"}
                extracted_fields = parse_document(lines, doc_key)
                field_confidences = [0.85] * len(extracted_fields)
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

    def process_dual(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        document_type: str = "cccd",
    ) -> DocumentResponse:
        """Process dual-sided document images (front and back) and return merged JSON response."""
        start_time = time.perf_counter()

        if image_a is None or image_a.size == 0 or image_b is None or image_b.size == 0:
            raise ValueError("Both images must be valid and non-empty for dual-image processing.")

        doc_key = document_type.lower()
        if doc_key in ("cccd_front", "cccd_back"):
            doc_key = "cccd"

        try:
            # 1. OCR with bboxes on both images
            elements_a = self.ocr_engine.recognize_lines_vietocr_with_bboxes(image_a)
            elements_b = self.ocr_engine.recognize_lines_vietocr_with_bboxes(image_b)

            # Auto-detect which image is Front and which is Back based on affinity
            affinity_a = self._get_cccd_side_affinity(elements_a)
            affinity_b = self._get_cccd_side_affinity(elements_b)

            if affinity_a >= affinity_b:
                front_img, front_elems = image_a, elements_a
                back_img, back_elems = image_b, elements_b
            else:
                front_img, front_elems = image_b, elements_b
                back_img, back_elems = image_a, elements_a

            from src.postprocessing.layout_parser import layout_parse
            from src.postprocessing.vn_name_dict import correct_full_name
            from src.postprocessing.cccd_rules import validate_and_fix_expiry
            from src.postprocessing.address_norm import normalize_gender_nationality, normalize_address

            # 2. Extract Front Side
            front_fields = layout_parse(front_elems, "cccd_front", image=front_img, ocr_engine=self.ocr_engine)
            if "full_name" in front_fields:
                front_fields["full_name"] = correct_full_name(front_fields["full_name"])
            front_fields = validate_and_fix_expiry(front_fields)
            front_fields = normalize_gender_nationality(front_fields)
            if "place_of_origin" in front_fields:
                front_fields["place_of_origin"] = normalize_address(front_fields["place_of_origin"])
            if "place_of_residence" in front_fields:
                front_fields["place_of_residence"] = normalize_address(front_fields["place_of_residence"])

            # 3. Extract Back Side
            back_fields = layout_parse(back_elems, "cccd_back", image=back_img, ocr_engine=self.ocr_engine)
            if "place_of_birth" in back_fields:
                back_fields["place_of_birth"] = normalize_address(back_fields["place_of_birth"])
            if "place_of_residence" in back_fields:
                back_fields["place_of_residence"] = normalize_address(back_fields["place_of_residence"])

            # 4. Merge fields: Back fields first, Front fields override
            merged_fields = {}
            for k, v in back_fields.items():
                if v:
                    merged_fields[k] = v

            for k, v in front_fields.items():
                if v:
                    merged_fields[k] = v

            # Fallback for residence: if front didn't have it but back does (new CCCD format)
            if not merged_fields.get("place_of_residence") and back_fields.get("place_of_residence"):
                merged_fields["place_of_residence"] = back_fields["place_of_residence"]

            # Normalize all fields
            validated_fields = {}
            for k, v in merged_fields.items():
                validated_fields[k] = normalize_field(v, field_type=k)

            overall_confidence = 0.95 if validated_fields else 0.0
            primary_engine_tag = "rapidocr"
        except Exception as e:
            logger.warning(f"Dual-image CCCD layout parsing failed, attempting fallback: {e}")
            res_a = self.process(image_a, "cccd")
            res_b = self.process(image_b, "cccd_back")
            validated_fields = {**res_b.fields, **res_a.fields}
            overall_confidence = round((res_a.confidence + res_b.confidence) / 2.0, 2)
            primary_engine_tag = res_a.metadata.ocr_engine

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
