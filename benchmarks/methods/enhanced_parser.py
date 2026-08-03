"""Method using baseline VietOCR but with an enhanced heuristic parser."""

import re
import unicodedata
import numpy as np
from typing import Dict, List

from rapidfuzz import process, fuzz

from benchmarks.registry import register_method
from benchmarks.methods.cccd_rules import validate_and_fix_expiry
from benchmarks.methods.vn_name_dict import correct_surname
from src.ocr.engine import OCREngine
from src.postprocessing.parser import _strip_accents, detect_cccd_side, KNOWN_ISSUERS


def _normalize_gender_nationality(fields: Dict[str, str]) -> Dict[str, str]:
    """Normalize gender and nationality to canonical Vietnamese format."""
    if "gender" in fields:
        raw_g = fields["gender"].upper()
        if "NAM" in raw_g or "NÀM" in raw_g:
            fields["gender"] = "Nam"
        elif "NỮ" in raw_g or "NU" in raw_g or "NO" in raw_g:
            fields["gender"] = "Nữ"

    if "nationality" in fields:
        raw_n = fields["nationality"].upper()
        if "VIET" in raw_n or "VIỆT" in raw_n:
            fields["nationality"] = "Việt Nam"

    return fields

_ENGINE = OCREngine()

# ============================================================
# Label-stripping helpers (ported from layout_graph_parser.py)
# ============================================================

_LABEL_STRIP_PATTERNS = [
    re.compile(r'(?i)place\s+of\s*(origin|residence|birth|ongin)'),
    re.compile(r'(?i)noi\s+thuong\s+tru'),
    re.compile(r'(?i)que\s+quan'),
    re.compile(r'(?i)noi\s+(dang\s+ky\s+)?khai\s+sinh'),
    re.compile(r'(?i)noi\s+cu\s+tru'),
    re.compile(r'(?i)ho\s+va\s+ten'),
    re.compile(r'(?i)full\s+name'),
    re.compile(r'(?i)ngay\s+sinh'),
    re.compile(r'(?i)date\s+of\s*(birth|expiry|issue|issae)'),
    re.compile(r'(?i)co\s+gia\s+tri\s+den'),
]

_LABEL_FRAG_TOKENS = {
    "place", "of", "origin", "birth", "residence", "ongin", "ace", "ce",
    "esidence", "irth", "oforigin", "ofresidence", "ofbirth",
    "noi", "thuong", "tru", "que", "quan", "khai", "sinh", "cu",
    "ho", "va", "ten", "ngay", "nam", "gioi", "tinh", "quoc", "tich",
    "full", "name", "date", "sex", "nationality", "identity", "card",
    "ofexpiry", "ofissue", "ofissae",
}


def _strip_all_labels(text: str) -> str:
    """Aggressively remove all label keywords from text."""
    if not text:
        return text

    for p in _LABEL_STRIP_PATTERNS:
        text = p.sub('', text)

    # Strip individual leftover garbage tokens from the START
    words = text.split()
    while words and _strip_accents(words[0]).lower() in _LABEL_FRAG_TOKENS:
        words.pop(0)

    result = " ".join(words)
    # Clean punctuation and stray digits left behind (runs AFTER token strip
    # so it catches artifacts like "1 :" exposed by removing "Quê quán")
    result = re.sub(r'^[\s/:;\-I|]+', '', result)
    result = re.sub(r'^\d{1,2}\s*[:\s]\s*', '', result)
    result = re.sub(r'^[,;:\s/]+', '', result).strip()
    return result


def _is_garbage_line(line: str) -> bool:
    """True if the line is a date, long ID, or expiry/authority/label fragment."""
    norm = _strip_accents(line).lower()
    if re.search(r'\d{2}[/.\-]\d{2}[/.\-]\d{4}', line):
        return True
    if re.search(r'\b\d{4,5}[/.\-]\d{4}\b|\b\d{2}[/.\-]\d{6,7}\b', line):
        return True
    if re.search(r'\b\d{10,}\b', line):
        return True
    garbage_phrases = [
        "co gia tri", "expiry", "giam doc", "cuc truong",
        "date of", "ofexpiry", "ofissue",
    ]
    return any(p in norm for p in garbage_phrases)


def extract_date(text: str) -> str:
    """Extract standard date DD/MM/YYYY from text."""
    match = re.search(r'\d{2}[/.\-]\d{2}[/.\-]\d{4}', text)
    if match:
        return match.group(0).replace(".", "/").replace("-", "/")
    return ""


def enhanced_parse_cccd_front(lines: List[str]) -> Dict[str, str]:
    fields = {}

    # Clean lines
    clean_lines = [l.strip() for l in lines if l.strip()]
    full_text = " ".join(clean_lines)

    # 1. ID Number — search without \b boundaries since spaces are stripped
    id_text = full_text.replace(" ", "")
    id_match = re.search(r'\d{12}', id_text)
    if id_match:
        fields["id_number"] = id_match.group(0)

    # 2. Extract ALL dates
    dates = []
    def normalize_mangled_date(d_str: str) -> str:
        d_clean = re.sub(r'[^\d]', '', d_str)
        if len(d_clean) >= 8:
            return f"{d_clean[:2]}/{d_clean[2:4]}/{d_clean[-4:]}"
        return d_str.replace(".", "/").replace("-", "/")

    for line in clean_lines:
        for match in re.finditer(r'\d{2}[/.\-]\d{2}[/.\-]\d{4}', line):
            d = match.group(0).replace(".", "/").replace("-", "/")
            if d not in dates:
                dates.append(d)
        for match in re.finditer(r'\b\d{4,5}[/.\-]\d{4}\b|\b\d{2}[/.\-]\d{6,7}\b', line):
            norm = normalize_mangled_date(match.group(0))
            if norm not in dates:
                dates.append(norm)

    if len(dates) >= 2:
        if "date_of_birth" not in fields:
            fields["date_of_birth"] = dates[0]
        fields["date_of_expiry"] = dates[-1]
    elif len(dates) == 1:
        if "date_of_birth" not in fields:
            fields["date_of_birth"] = dates[0]
        elif "date_of_expiry" not in fields:
            fields["date_of_expiry"] = dates[0]

    # 3. Gender
    combined_lower = _strip_accents(full_text.lower())
    if "nam" in combined_lower and "nu" not in combined_lower:
        fields["gender"] = "Nam"
    elif "nu" in combined_lower:
        fields["gender"] = "Nữ"

    # 4. Nationality
    if "viet nam" in combined_lower:
        fields["nationality"] = "Việt Nam"

    # 5. Find line indices using fuzzy matching for labels
    name_idx, dob_idx, origin_idx, residence_idx = -1, -1, -1, -1

    for i, line in enumerate(clean_lines):
        norm_line = _strip_accents(line.lower())

        if name_idx == -1 and ("ho va ten" in norm_line or "full name" in norm_line):
            name_idx = i
        if dob_idx == -1 and ("ngay sinh" in norm_line or "date of birth" in norm_line):
            dob_idx = i
        if origin_idx == -1 and ("que quan" in norm_line or "place of origin" in norm_line):
            origin_idx = i
        if residence_idx == -1 and ("thuong tru" in norm_line or "place of residence" in norm_line):
            residence_idx = i

    # Fallback to simple "ten" "sinh" if fuzzy fails
    if name_idx == -1:
        for i, line in enumerate(clean_lines):
            norm = _strip_accents(line.lower())
            if "ten:" in norm or "name:" in norm or "tên:" in line.lower():
                name_idx = i; break

    # Extract Full Name — with boundary detection to stop at date/label lines
    if name_idx != -1:
        line = clean_lines[name_idx]
        if ":" in line:
            name = line.split(":", 1)[1].strip()
            # Strip trailing label/date fragments from inline name value
            name = _clean_name_value(name)
            if len(name) > 3:
                fields["full_name"] = name
        else:
            name_parts = []
            max_idx = dob_idx if dob_idx != -1 else min(name_idx + 3, len(clean_lines))
            for i in range(name_idx + 1, max_idx):
                part = clean_lines[i]
                # Stop if we hit a date label or date value
                norm_part = _strip_accents(part.lower())
                if _is_date_label(norm_part):
                    break
                if re.search(r'\d{2}/\d{2}/\d{4}', part):
                    break
                if not re.search(r'\d', part) and len(part) > 2:
                    name_parts.append(part)
            if name_parts:
                name = " ".join(name_parts)
                name = _clean_name_value(name)
                fields["full_name"] = name

    # Extract Origin & Residence — with aggressive label stripping
    def extract_address_block(start_idx: int, end_idx: int, labels_to_remove: List[str]) -> str:
        if start_idx == -1:
            return ""

        end = end_idx if end_idx != -1 else len(clean_lines)
        if end <= start_idx:
            end = len(clean_lines)

        block = list(clean_lines[start_idx:end])

        # Clean all lines in the block through label stripping
        cleaned_block = []
        for i, l in enumerate(block):
            # Strip labels from every line, not just the first
            stripped = _strip_all_labels(l)
            if not stripped:
                continue
            if _is_garbage_line(stripped):
                continue
            cleaned_block.append(stripped)

        result = " ".join(cleaned_block).replace(" ,", ",").strip()

        # Final cleanup: remove leading/trailing punctuation artifacts
        result = re.sub(r'^[\s,;:/\-]+', '', result)
        result = re.sub(r'[\s,;:/\-]+$', '', result)
        return result

    fields["place_of_origin"] = extract_address_block(origin_idx, residence_idx, ["quê quán", "que quan", "place of origin"])
    fields["place_of_residence"] = extract_address_block(residence_idx, len(clean_lines), ["nơi thường trú", "thuong tru", "place of residence", "noi th"])

    return fields


def _is_date_label(norm_text: str) -> bool:
    """Check if normalized text contains a date-related label."""
    markers = ["ngay", "date", "sinh", "birth", "thang", "nam sinh"]
    return any(m in norm_text for m in markers)


def _clean_name_value(name: str) -> str:
    """Remove trailing label/date garbage from a name string."""
    # Cut at common trailing noise patterns
    cut_patterns = [
        r'\s*Ngày.*$',
        r'\s*ngày.*$',
        r'\s*Date\s+of.*$',
        r'\s*date\s+of.*$',
        r'\s*/Date.*$',
        r'\s*\d{2}/\d{2}/\d{4}.*$',
    ]
    for p in cut_patterns:
        name = re.sub(p, '', name)

    # Remove non-letter characters that shouldn't be in a name
    name = re.sub(r'[0-9\._\+\*\?\!]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def enhanced_parse_cccd_back(lines: List[str]) -> Dict[str, str]:
    fields = {}
    clean_lines = [l.strip() for l in lines if l.strip()]

    # Extract ALL dates
    dates = []
    for line in clean_lines:
        for match in re.finditer(r'\d{2}[/.\-]\d{2}[/.\-]\d{4}', line):
            d = match.group(0).replace(".", "/").replace("-", "/")
            if d not in dates:
                dates.append(d)

    # Also extract date in format "Ngày 23 tháng 06 năm 2023"
    for line in clean_lines:
        norm = _strip_accents(line.lower())
        if "ngay" in norm and "thang" in norm and "nam" in norm:
            nums = re.findall(r'\d+', line)
            if len(nums) >= 3:
                d = f"{nums[-3].zfill(2)}/{nums[-2].zfill(2)}/{nums[-1]}"
                if d not in dates:
                    dates.append(d)

    if len(dates) == 1:
        fields["date_of_issue"] = dates[0]
    elif len(dates) >= 2:
        fields["date_of_issue"] = dates[0]
        fields["date_of_expiry"] = dates[-1]

    # Find indices
    residence_idx, birth_idx = -1, -1
    for i, line in enumerate(clean_lines):
        norm = _strip_accents(line.lower())
        if residence_idx == -1 and "cu tru" in norm:
            residence_idx = i
        if birth_idx == -1 and "khai sinh" in norm:
            birth_idx = i

    def extract_address_block(start_idx: int, end_idx: int, labels: List[str]) -> str:
        if start_idx == -1: return ""
        end = end_idx if (end_idx != -1 and end_idx > start_idx) else len(clean_lines)

        block = list(clean_lines[start_idx:end])

        # Clean all lines through label stripping
        cleaned_block = []
        for l in block:
            stripped = _strip_all_labels(l)
            if not stripped:
                continue
            if re.search(r'\d{2}/\d{2}/\d{4}', stripped):
                break
            if "bo cong an" in _strip_accents(stripped.lower()):
                break
            if _is_garbage_line(stripped):
                continue
            cleaned_block.append(stripped)

        result = " ".join(cleaned_block).replace(" ,", ",").strip()
        result = re.sub(r'^[\s,;:/\-]+', '', result)
        result = re.sub(r'[\s,;:/\-]+$', '', result)
        return result

    if residence_idx != -1:
        fields["place_of_residence"] = extract_address_block(residence_idx, birth_idx, ["nơi cư trú", "noi cu tru", "place of residence"])
    if birth_idx != -1:
        fields["place_of_birth"] = extract_address_block(birth_idx, len(clean_lines), ["nơi đăng ký khai sinh", "noi dang ky khai sinh", "place of birth"])

    # Extract issuing authority
    for i, line in enumerate(clean_lines):
        norm = _strip_accents(line.lower())
        if "giam doc" in norm or "cuc truong" in norm or "canh sat" in norm or "bo cong an" in norm:
            place = line
            if i + 1 < len(clean_lines):
                next_line = clean_lines[i+1]
                if not re.search(r'\d', next_line) and "director" not in next_line.lower():
                    place += " " + next_line

            place = re.sub(r'(?i)(CỤC TRƯỞNG|GIÁM ĐỐC|CỤC TRƯỜNG)', '', place).strip()

            # Fuzzy correct
            match_result = process.extractOne(place, KNOWN_ISSUERS, scorer=fuzz.WRatio, score_cutoff=55.0)
            if match_result:
                place = match_result[0]

            fields["place_of_issue"] = place
            break

    return fields


def enhanced_parse(lines: List[str], document_type: str) -> Dict[str, str]:
    if document_type == "cccd_auto":
        document_type = detect_cccd_side(lines)

    if document_type == "cccd":
        fields = enhanced_parse_cccd_front(lines)

        # Post-processing: surname correction
        if "full_name" in fields:
            fields["full_name"] = correct_surname(fields["full_name"])

        # Post-processing: CCCD expiry date cross-validation
        fields = validate_and_fix_expiry(fields)

        # Post-processing: gender/nationality normalization
        fields = _normalize_gender_nationality(fields)

        return fields
    elif document_type == "cccd_back":
        fields = enhanced_parse_cccd_back(lines)
        fields = _normalize_gender_nationality(fields)
        return fields
    return {}


@register_method("enhanced_parser_vietocr")
def method_enhanced_parser_vietocr(image: np.ndarray, doc_type: str) -> Dict[str, str]:
    """Run VietOCR and parse using the improved heuristic parser."""
    lines = _ENGINE.recognize_lines_vietocr(image)
    fields = enhanced_parse(lines, doc_type)
    return fields
