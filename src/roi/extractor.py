"""ROI extraction and template definitions for Vietnamese identity documents."""

from typing import Dict, Any, Tuple, Optional
import numpy as np


# Normalized ROI coordinates (ymin, xmin, ymax, xmax) for supported document templates
DOCUMENT_TEMPLATES: Dict[str, Dict[str, Tuple[float, float, float, float]]] = {
    "cccd": {
        "id_number": (0.32, 0.35, 0.44, 0.85),
        "full_name": (0.44, 0.35, 0.54, 0.90),
        "date_of_birth": (0.54, 0.45, 0.62, 0.75),
        "gender": (0.62, 0.35, 0.70, 0.50),
        "nationality": (0.62, 0.70, 0.70, 0.95),
        "place_of_origin": (0.70, 0.35, 0.78, 0.95),
        "place_of_residence": (0.78, 0.35, 0.88, 0.95),
        "date_of_issue": (0.88, 0.45, 0.96, 0.80),
    },
    "driving_license": {
        "license_number": (0.22, 0.48, 0.32, 0.92),
        "full_name": (0.34, 0.35, 0.45, 0.92),
        "date_of_birth": (0.45, 0.38, 0.53, 0.75),
        "nationality": (0.53, 0.38, 0.61, 0.75),
        "license_class": (0.61, 0.75, 0.75, 0.92),
        "date_of_issue": (0.75, 0.38, 0.83, 0.75),
        "date_of_expiry": (0.83, 0.38, 0.92, 0.75),
    },
    "vehicle_registration": {
        "license_plate": (0.15, 0.30, 0.27, 0.85),
        "owner_name": (0.27, 0.30, 0.37, 0.90),
        "owner_address": (0.37, 0.20, 0.48, 0.95),
        "brand": (0.48, 0.25, 0.56, 0.55),
        "model": (0.48, 0.65, 0.56, 0.95),
        "color": (0.56, 0.25, 0.64, 0.55),
        "engine_number": (0.64, 0.30, 0.72, 0.90),
        "chassis_number": (0.72, 0.30, 0.80, 0.90),
        "vehicle_type": (0.80, 0.30, 0.88, 0.75),
        "date_of_issue": (0.88, 0.45, 0.96, 0.85),
    },
}

# Add aliases for template names
DOCUMENT_TEMPLATES["cccd_front"] = DOCUMENT_TEMPLATES["cccd"]
DOCUMENT_TEMPLATES["gplx_front"] = DOCUMENT_TEMPLATES["driving_license"]
DOCUMENT_TEMPLATES["vehicle_registration_front"] = DOCUMENT_TEMPLATES["vehicle_registration"]


class ROIExtractor:
    """Extractor for Region of Interest (ROI) crops based on document templates."""

    def __init__(self, templates: Optional[Dict[str, Dict[str, Tuple[float, float, float, float]]]] = None):
        self.templates = templates or DOCUMENT_TEMPLATES

    def extract_rois(self, image: np.ndarray, document_type: str) -> Dict[str, np.ndarray]:
        """Extract ROI image crops for each defined field in the given document type."""
        if image is None or image.size == 0:
            return {}

        doc_key = document_type.lower()
        if doc_key not in self.templates:
            raise ValueError(f"Unsupported document type: {document_type}. Supported: {list(self.templates.keys())}")

        template_fields = self.templates[doc_key]
        h, w = image.shape[:2]
        rois: Dict[str, np.ndarray] = {}

        for field_name, (ymin_rel, xmin_rel, ymax_rel, xmax_rel) in template_fields.items():
            ymin = max(0, int(ymin_rel * h))
            xmin = max(0, int(xmin_rel * w))
            ymax = min(h, int(ymax_rel * h))
            xmax = min(w, int(xmax_rel * w))

            if ymax > ymin and xmax > xmin:
                rois[field_name] = image[ymin:ymax, xmin:xmax].copy()

        return rois


def extract_rois(image: np.ndarray, document_type: str) -> Dict[str, np.ndarray]:
    """Helper function to extract ROIs using default ROIExtractor."""
    extractor = ROIExtractor()
    return extractor.extract_rois(image, document_type)
