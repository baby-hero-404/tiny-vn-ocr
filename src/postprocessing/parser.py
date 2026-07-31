"""Heuristic text parser for full-image OCR results (flat text, no bboxes).

This parser works with plain text lines from the OCR engine.
For bbox-aware parsing, see benchmarks/methods/layout_graph_parser.py.
"""

from typing import List, Dict, Any, Optional
import re
import unicodedata
from rapidfuzz import process, fuzz


# ============================================================
# Utilities
# ============================================================

def _strip_accents(text: str) -> str:
    """Remove Vietnamese diacritics for keyword matching."""
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


DATE_PATTERN = re.compile(r'\d{2}[/.\-]\d{2}[/.\-]\d{4}')
ID_PATTERN = re.compile(r'\b\d{12}\b')
MANGLED_DATE_PATTERN = re.compile(r'\b\d{4,5}[/.\-]\d{4}\b|\b\d{2}[/.\-]\d{6,7}\b')

# Known CCCD issuing authorities (for fuzzy matching)
KNOWN_ISSUERS = [
    "CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI",
    "CỤC CẢNH SÁT ĐKQL CƯ TRÚ VÀ DLQG VỀ DÂN CƯ",
    "BỘ CÔNG AN",
    "CÔNG AN TỈNH SÓC TRĂNG",
    "CÔNG AN TỈNH AN GIANG",
    "CÔNG AN THÀNH PHỐ HỒ CHÍ MINH",
    "CÔNG AN THÀNH PHỐ HÀ NỘI",
    "CÔNG AN THÀNH PHỐ ĐÀ NẴNG",
    "CÔNG AN THÀNH PHỐ CẦN THƠ",
    "CÔNG AN THÀNH PHỐ HẢI PHÒNG",
]

# Bilingual label fragments to strip from address lines
_ADDRESS_LABELS = [
    "que quan", "place of origin", "noi thuong tru", "thuong tru",
    "place of residence", "residence", "origin", "place of",
]


# ============================================================
# Shared helpers
# ============================================================

def _normalize_mangled_date(raw: str) -> str:
    """Fix dates with a missing slash, e.g. '23102/2032' → '23/10/2032'."""
    digits = re.sub(r'[^\d]', '', raw)
    if len(digits) >= 8:
        return f"{digits[:2]}/{digits[2:4]}/{digits[-4:]}"
    return raw


def _find_label_line(lines: List[str], keywords: List[str],
                     *, fuzzy_threshold: int = 85) -> int:
    """Return index of the first line containing any keyword, or -1."""
    for i, line in enumerate(lines):
        norm = _strip_accents(line).lower()
        for kw in keywords:
            if kw in norm:
                return i
            if len(kw) > 5 and fuzz.partial_ratio(kw, norm) > fuzzy_threshold:
                return i
    return -1


def _is_garbage_line(line: str) -> bool:
    """True if the line is a date, long ID, or expiry/authority label."""
    norm = _strip_accents(line).lower()
    if DATE_PATTERN.search(line) or MANGLED_DATE_PATTERN.search(line):
        return True
    if re.search(r'\b\d{10,}\b', line):
        return True
    garbage_phrases = ["co gia tri", "expiry", "giam doc", "cuc truong"]
    return any(p in norm for p in garbage_phrases)


def _extract_all_dates(lines: List[str]) -> List[str]:
    """Extract all dates (including mangled ones) from the text."""
    dates: List[str] = []
    for line in lines:
        for m in DATE_PATTERN.finditer(line):
            dates.append(m.group(0))
        for m in MANGLED_DATE_PATTERN.finditer(line):
            norm = _normalize_mangled_date(m.group(0))
            if norm not in dates:
                dates.append(norm)
    return dates


def _extract_address(lines_subset: List[str], keywords: List[str]) -> str:
    """Extract an address value from a slice of lines starting at the label.

    Strips the label text from the first line and filters out garbage lines.
    """
    if not lines_subset:
        return ""

    subset = list(lines_subset)
    first = subset[0]
    norm = _strip_accents(first).lower()
    has_label = any(kw in norm for kw in keywords)

    if has_label:
        if ":" in first:
            content = first.split(":", 1)[-1].strip()
            subset[0] = content if len(content) > 2 else ""
        else:
            clean = norm
            for label in sorted(_ADDRESS_LABELS, key=len, reverse=True):
                clean = clean.replace(label, "")
            clean = re.sub(r'^[^\w]+', '', clean).strip()
            if len(clean) > 1:
                # Trim label prefix from original (preserves diacritics)
                subset[0] = first[len(first) - len(clean):].strip()
            else:
                subset[0] = ""

    valid = [l for l in subset if l and not _is_garbage_line(l)]
    return " ".join(valid).replace(" ,", ",").strip()


# ============================================================
# CCCD Front-side parser
# ============================================================

def parse_cccd_front(lines: List[str]) -> Dict[str, Any]:
    """Parse CCCD front side from flat OCR text lines."""
    fields: Dict[str, Any] = {}

    # --- ID Number (12 digits) ---
    for line in lines:
        m = ID_PATTERN.search(line)
        if m:
            fields["id_number"] = m.group(0)
            break

    # --- Full Name ---
    name_idx = _find_label_line(lines, ["ho va ten", "full name", "ho ten"])
    if name_idx != -1:
        line = lines[name_idx]
        if ":" in line and len(line.split(":")[-1].strip()) > 3:
            fields["full_name"] = line.split(":")[-1].strip()
        elif name_idx + 1 < len(lines):
            fields["full_name"] = lines[name_idx + 1].strip()

    # --- Date of Birth ---
    dob_idx = _find_label_line(lines, ["ngay sinh", "date of birth"])
    if dob_idx != -1:
        m = DATE_PATTERN.search(lines[dob_idx])
        if m:
            fields["date_of_birth"] = m.group(0)
        elif dob_idx + 1 < len(lines):
            m = DATE_PATTERN.search(lines[dob_idx + 1])
            if m:
                fields["date_of_birth"] = m.group(0)

    # --- Date of Expiry (from all dates heuristic) ---
    all_dates = _extract_all_dates(lines)
    if len(all_dates) >= 2:
        if "date_of_birth" not in fields:
            fields["date_of_birth"] = all_dates[0]
        fields["date_of_expiry"] = all_dates[-1]
    elif len(all_dates) == 1:
        if "date_of_birth" not in fields:
            fields["date_of_birth"] = all_dates[0]

    # --- Gender & Nationality ---
    for line in lines:
        norm = _strip_accents(line).lower()
        if "tinh" in norm or "sex" in norm:
            if "nam" in norm:
                fields["gender"] = "NAM"
            elif "nu" in norm or "nữ" in line.lower():
                fields["gender"] = "NU"
        if "tich" in norm or "national" in norm:
            if "viet" in norm:
                fields["nationality"] = "VIET NAM"

    # --- Place of Origin & Place of Residence ---
    origin_idx = _find_label_line(lines, ["que quan", "origin"])
    residence_idx = _find_label_line(lines, ["thuong tru", "residence"])

    if origin_idx != -1:
        end = residence_idx if residence_idx != -1 else len(lines)
        fields["place_of_origin"] = _extract_address(
            lines[origin_idx:end], ["que quan", "origin"])

    if residence_idx != -1:
        fields["place_of_residence"] = _extract_address(
            lines[residence_idx:], ["thuong tru", "residence"])

    return fields


# ============================================================
# CCCD Back-side parser
# ============================================================

def parse_cccd_back(lines: List[str]) -> Dict[str, Any]:
    """Parse CCCD back side from flat OCR text lines."""
    fields: Dict[str, Any] = {}

    # Locate section labels
    residence_idx = _find_label_line(lines, ["cu tru"])
    birth_idx = _find_label_line(lines, ["khai sinh"])

    # Issue & Expiry dates need special handling (both use "ngày tháng năm")
    issue_idx = expiry_idx = -1
    for i, line in enumerate(lines):
        norm = _strip_accents(line).lower()
        has_date_label = "ngay" in norm and "thang" in norm and "nam" in norm
        if not has_date_label:
            continue
        if "het han" in norm or "expiry" in norm:
            if expiry_idx == -1:
                expiry_idx = i
        elif issue_idx == -1:
            issue_idx = i

    # Build sorted section boundaries
    section_starts = sorted(
        idx for idx in (residence_idx, birth_idx, issue_idx, expiry_idx) if idx != -1
    )

    def _section_end(start: int) -> int:
        for idx in section_starts:
            if idx > start:
                return idx
        return len(lines)

    def _extract_multiline(start: int, label_keywords: List[str]) -> str:
        end = _section_end(start)
        subset = list(lines[start:end])
        first = subset[0]

        # Strip label from first line
        if ":" in first:
            content = first.split(":", 1)[-1].strip()
            subset[0] = content if len(content) > 2 else ""
        elif "/" in first:
            content = first.split("/", 1)[-1].strip()
            subset[0] = content if len(content) > 2 else ""
        else:
            pattern = r'(?i)(' + "|".join(re.escape(kw) for kw in label_keywords) + r')[^\w]*'
            clean = re.sub(pattern, '', first).strip()
            if clean == first.strip() and len(clean) > 15:
                parts = clean.split()
                if len(parts) > 5:
                    clean = " ".join(parts[4:])
            subset[0] = clean

        valid = [
            l for l in subset
            if l and not DATE_PATTERN.search(l)
            and "bộ công an" not in l.lower()
            and "bo cong an" not in l.lower()
            and "ministry" not in l.lower()
        ]
        return " ".join(valid).replace(" ,", ",").strip()

    def _extract_date(idx: int) -> str:
        nums = re.findall(r'\d+', lines[idx])
        if len(nums) >= 3:
            return f"{nums[-3].zfill(2)}/{nums[-2].zfill(2)}/{nums[-1]}"
        for line in lines[idx:_section_end(idx)]:
            m = DATE_PATTERN.search(line)
            if m:
                return m.group(0)
        return ""

    # --- Residence ---
    if residence_idx != -1:
        fields["place_of_residence"] = _extract_multiline(
            residence_idx, ["nơi cư trú", "noi cu tru", "place of residence"])

    # --- Place of Birth ---
    if birth_idx != -1:
        fields["place_of_birth"] = _extract_multiline(
            birth_idx, ["nơi đăng ký khai sinh", "noi dang ky khai sinh", "place of birth"])

    # --- Date of Issue ---
    if issue_idx != -1:
        fields["date_of_issue"] = _extract_date(issue_idx)
    else:
        # Fallback: old-style CCCD back has a single "Ngày, tháng, năm" line
        for line in lines:
            norm = _strip_accents(line).lower()
            if "ngay" in norm and "thang" in norm and "nam" in norm:
                nums = re.findall(r'\d+', line)
                if len(nums) >= 3:
                    fields["date_of_issue"] = f"{nums[-3].zfill(2)}/{nums[-2].zfill(2)}/{nums[-1]}"
                    break
            m = DATE_PATTERN.search(line)
            if m:
                fields["date_of_issue"] = m.group(0)
                break

    # --- Date of Expiry ---
    if expiry_idx != -1:
        fields["date_of_expiry"] = _extract_date(expiry_idx)

    # --- Place of Issue (issuing authority) ---
    for i, line in enumerate(lines):
        norm = _strip_accents(line).lower()
        issuer_keywords = ["giam doc", "cuc truong", "canh sat", "bo cong an"]
        if not any(kw in norm for kw in issuer_keywords):
            continue

        place = line
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            skip_words = ["director", "general", "ministry"]
            if not any(w in next_line.lower() for w in skip_words) and "0" not in next_line:
                place += " " + next_line

        place = re.sub(r'(?i)(CỤC TRƯỞNG|GIÁM ĐỐC|CỤC TRƯỜNG)', '', place).strip()

        match = process.extractOne(place, KNOWN_ISSUERS, scorer=fuzz.WRatio, score_cutoff=70.0)
        fields["place_of_issue"] = match[0] if match else place
        break

    return fields


# ============================================================
# Side detection & entry point
# ============================================================

def detect_cccd_side(lines: List[str]) -> str:
    """Auto-detect front vs back side from OCR content."""
    combined = _strip_accents(" ".join(lines)).lower()

    front_kw = ["ten", "name", "sinh", "birth", "que quan", "origin", "thuong tru", "residence"]
    back_kw = ["dac diem", "ngon tro", "cuc truong", "giam doc", "canh sat",
               "finger", "khai sinh", "cu tru", "het han", "bo cong an"]

    front_score = sum(1 for kw in front_kw if kw in combined)
    back_score = sum(1 for kw in back_kw if kw in combined)

    return "cccd_back" if back_score > front_score else "cccd"


def parse_document(lines: List[str], document_type: str) -> Dict[str, Any]:
    """Main entry point: parse OCR lines into structured fields."""
    if document_type == "cccd_auto":
        document_type = detect_cccd_side(lines)

    if document_type in ("cccd", "cccd_front"):
        return parse_cccd_front(lines)
    elif document_type == "cccd_back":
        return parse_cccd_back(lines)
    return {}
