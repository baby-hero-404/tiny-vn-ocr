"""QR Code Scanner and Parser for Vietnamese Chip CCCD.

Extracts ground-truth data from the top-right QR code on chip-based CCCD front cards.
The standard format is a pipe-delimited string:
    Số CCCD|Số CMND cũ|Họ và tên|Ngày sinh (DDMMYYYY)|Giới tính|Địa chỉ thường trú|Ngày cấp (DDMMYYYY)
"""

from typing import Optional, Dict, Any
import re
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

_QR_DETECTOR = None


def get_qr_detector() -> cv2.QRCodeDetector:
    global _QR_DETECTOR
    if _QR_DETECTOR is None:
        _QR_DETECTOR = cv2.QRCodeDetector()
    return _QR_DETECTOR


def _format_date(date_str: str) -> str:
    """Format DDMMYYYY to DD/MM/YYYY."""
    s = date_str.strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:2]}/{s[2:4]}/{s[4:]}"
    return s


def parse_qr_payload(payload: str) -> Optional[Dict[str, str]]:
    """Parse pipe-delimited CCCD QR code payload into structured fields."""
    if not payload or "|" not in payload:
        return None

    parts = payload.strip().split("|")
    if len(parts) < 5:
        return None

    fields: Dict[str, str] = {}

    # Part 0: CCCD number (12 digits)
    id_candidate = parts[0].strip()
    if len(id_candidate) == 12 and id_candidate.isdigit():
        fields["id_number"] = id_candidate

    # Part 1: Old CMND number (9 digits, optional)
    old_id = parts[1].strip()
    if old_id and len(old_id) == 9 and old_id.isdigit():
        fields["old_id_number"] = old_id

    # Part 2: Full name
    name = parts[2].strip()
    if name:
        fields["full_name"] = name.upper()

    # Part 3: Date of birth (DDMMYYYY)
    dob = parts[3].strip()
    if dob:
        fields["date_of_birth"] = _format_date(dob)

    # Part 4: Gender (Nam / Nữ)
    gender = parts[4].strip()
    if gender:
        fields["gender"] = gender

    # Part 5: Place of residence (if available)
    if len(parts) > 5:
        residence = parts[5].strip()
        if residence:
            fields["place_of_residence"] = residence

    # Part 6: Date of issue (DDMMYYYY, if available)
    if len(parts) > 6:
        issue = parts[6].strip()
        if issue:
            fields["date_of_issue"] = _format_date(issue)

    return fields if "id_number" in fields or "full_name" in fields else None


def scan_and_parse_cccd_qr(image: np.ndarray) -> Optional[Dict[str, str]]:
    """Scan and parse QR code from a CCCD image.
    
    Tries top-right quadrant first (standard chip card QR location), then full image.
    Applies adaptive scaling and contrast enhancements to decode challenging scans.
    """
    if image is None or image.size == 0:
        return None

    detector = get_qr_detector()
    h, w = image.shape[:2]

    # Candidate regions to inspect
    regions = [
        # Top-right quadrant (standard CCCD QR location)
        image[0:int(h * 0.45), int(w * 0.55):w],
        # Full image
        image,
    ]

    for roi in regions:
        if roi is None or roi.size == 0:
            continue

        # 1. Try raw detection
        try:
            val, pts, _ = detector.detectAndDecode(roi)
            if val:
                parsed = parse_qr_payload(val)
                if parsed:
                    return parsed
        except Exception:
            pass

        # 2. Try with grayscale + resizing + CLAHE
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            for scale in [1.5, 2.0, 0.75]:
                scaled = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                val, pts, _ = detector.detectAndDecode(scaled)
                if val:
                    parsed = parse_qr_payload(val)
                    if parsed:
                        return parsed
        except Exception:
            pass

    return None
