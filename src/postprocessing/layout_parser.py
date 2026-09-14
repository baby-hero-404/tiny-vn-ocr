"""Layout Graph Parser v2 — bbox-aware field extraction for CCCD."""

import re
import unicodedata
from typing import Dict, List, Any, Optional, Tuple

from rapidfuzz import fuzz

date_pattern = re.compile(r'\d{2}[/.\-]\d{2}[/.\-]\d{4}')


def extract_mangled_date(text: str) -> str:
    match = re.search(r'\d{2}[/.\-]\d{2}[/.\-]\d{4}', text)
    if match:
        return match.group(0).replace(".", "/").replace("-", "/")
    match = re.search(r'\b\d{4,5}[/.\-]\d{4}\b|\b\d{2}[/.\-]\d{6,7}\b', text)
    if match:
        d_clean = re.sub(r'[^\d]', '', match.group(0))
        if len(d_clean) >= 8:
            return f"{d_clean[:2]}/{d_clean[2:4]}/{d_clean[-4:]}"
    return ""


def _strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize('NFD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


# ============================================================
# Step 1: Merge boxes on the same horizontal line
# ============================================================

# Removed merge_horizontal because RapidOCR naturally detects lines,
# and merging them can mistakenly merge separate columns (e.g. Expiry Date and Residence)


# ============================================================
# Step 2: Label detection
# ============================================================

# Front-side labels
FRONT_LABELS = {
    "id_number":    ["so", "no", "dinh danh", "personal identification"],
    "full_name":    ["ho va ten", "full name", "ho ten"],
    "dob":          ["ngay sinh", "date of birth"],
    "gender_nat":   ["gioi tinh", "sex", "quoc tich", "nationality"],
    "origin":       ["que quan", "place of origin"],
    "residence":    ["thuong tru", "place of residence", "noi thuong"],
    "expiry":       ["co gia tri", "date of expiry", "het han"],
}

# Back-side labels
BACK_LABELS = {
    "residence":    ["noi cu tru", "noi cum", "place of residence", "cu tru"],
    "birth_place":  ["khai sinh", "place of birth", "dang ky khai"],
    "issue_date":   ["ngay thang nam", "date of issue", "ngay cap", "date,month"],
    "expiry_date":  ["het han", "date of expiry", "ofexpity", "ofespity"],
    "issuer":       ["bo cong an", "ministry", "cuc canh sat", "cuc truong", "giam doc"],
}


def _classify_line(text: str, label_set: Dict[str, List[str]]) -> Optional[str]:
    """Return the label key if this line contains a label, else None."""
    norm = _strip_accents(text).lower()
    norm_padded = f" {norm} "
    for key, keywords in label_set.items():
        for kw in keywords:
            if f" {kw} " in norm_padded:
                return key
            if norm.startswith(kw + " ") or norm.endswith(" " + kw):
                return key
            if norm == kw:
                return key
            if len(kw) > 5 and fuzz.partial_ratio(kw, norm) > 85:
                return key
    return None


def _extract_inline_value(text: str) -> str:
    """Extract value part from a label line (after : or /)."""
    # Try colon first
    if ":" in text:
        val = text.split(":", 1)[1].strip()
        if len(val) > 2:
            return val
    # Try slash — but only if it separates VN/EN label, not part of address
    parts = text.split("/")
    if len(parts) >= 2:
        # The last segment after the last "/" might be the value
        last = parts[-1].strip()
        # Check if it looks like a value (not a label keyword)
        norm = _strip_accents(last).lower()
        label_words = {"name", "birth", "origin", "residence", "sex", "nationality",
                       "expiry", "issue", "no", "oforigin", "ofresidence", "ofbirth"}
        if not any(lw in norm for lw in label_words) and len(last) > 2:
            return last
    return ""


# ============================================================
# Step 3: CCCD Front-side layout parser
# ============================================================

def parse_cccd_front(elements: List[Dict], image=None, ocr_engine=None) -> Dict[str, str]:
    """Parse CCCD front side using bbox layout."""
    fields: Dict[str, str] = {}
    if not elements:
        return fields

    # 1. Sort elements top-to-bottom, then left-to-right
    lines = sorted(elements, key=lambda e: (e["center"][1], e["bbox"][0]))
    
    # Calculate height for each line if missing
    for el in lines:
        if "height" not in el:
            el["height"] = el["bbox"][3] - el["bbox"][1]
    full_text = " ".join(el["text"] for el in lines)

    # Determine the main content X range (right column of CCCD)
    # The main content is typically x > 200 (left side has photo + expiry)
    x_values = [el["bbox"][0] for el in lines]
    if x_values:
        # Find the most common x-start cluster (the main column)
        main_x_start = sorted(x_values)[len(x_values) // 3]  # rough estimate

    # Tag each line
    tagged: List[Tuple[int, Optional[str], Dict]] = []  # (index, label_key, element)
    for i, el in enumerate(lines):
        key = _classify_line(el["text"], FRONT_LABELS)
        tagged.append((i, key, el))

    # --- ID Number: regex on full text & Retry Logic ---
    id_text = full_text.replace(" ", "")
    id_match = re.search(r'\d{12}', id_text)
    
    # Import the semantic validator
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
    from src.postprocessing.validator import FieldValidator
    
    cand = id_match.group(0) if id_match else None
    
    if cand and FieldValidator.is_valid_cccd_id(cand):
        fields["id_number"] = cand
    else:
        if cand:
            fields["id_number"] = cand # fallback
            
        # Step 1: ID Retry & Multi-pass OCR (if invalid pattern or not found)
        if image is not None and ocr_engine is not None:
            # Find the bounding box of the line tagged as "id_number"
            id_el = next((el for _, key, el in tagged if key == "id_number"), None)
            if id_el:
                import cv2
                import numpy as np
                
                if "crop" in id_el:
                    crop = id_el["crop"]
                else:
                    xmin, ymin, xmax, ymax = id_el["bbox"]
                    pad_x = 3
                    pad_y_top = 6
                    pad_y_bottom = 4
                    c_xmin = max(0, int(xmin) - pad_x)
                    c_ymin = max(0, int(ymin) - pad_y_top)
                    c_xmax = min(image.shape[1], int(xmax) + pad_x)
                    c_ymax = min(image.shape[0], int(ymax) + pad_y_bottom)
                    crop = image[c_ymin:c_ymax, c_xmin:c_xmax]

                if crop is None or crop.size == 0:
                    pass
                else:
                    # Bước 2: Multi preprocessing
                    crop_orig = crop.copy()
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
                    sharpened_gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
                    crop_sharp = cv2.cvtColor(sharpened_gray, cv2.COLOR_GRAY2BGR)
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    crop_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

                    best_cand = cand
                    for c_img in [crop_orig, crop_sharp, crop_thresh]:
                        res, _ = ocr_engine.recognize_rapidocr(c_img)
                        if not res:
                            res, _ = ocr_engine.recognize_tesseract(c_img, field_type="id_number")
                        if not res:
                            res = ocr_engine.recognize_crop_vietocr(c_img)
                        res_clean = re.sub(r"\D", "", res)
                        m = re.search(r"\d{12}", res_clean)
                        if m:
                            val = m.group(0)
                            if FieldValidator.is_valid_cccd_id(val):
                                best_cand = val
                                break
                    fields["id_number"] = best_cand

    # --- Full Name ---
    for idx, key, el in tagged:
        if key == "full_name":
            # Check inline value
            inline = _extract_inline_value(el["text"])
            if inline and len(inline) > 3 and not re.search(r'\d', inline):
                fields["full_name"] = inline
            else:
                # Value is the next line below
                if idx + 1 < len(tagged):
                    next_el = tagged[idx + 1][2]
                    candidate = next_el["text"].strip()
                    # Must look like a name: mostly uppercase letters, no dates
                    if len(candidate) > 3 and not re.search(r'\d{2}/\d{2}', candidate):
                        fields["full_name"] = candidate
            break

    # --- Date of Birth ---
    for idx, key, el in tagged:
        if key == "dob":
            val = extract_mangled_date(el["text"])
            if val:
                fields["date_of_birth"] = val
            else:
                # Check next line
                if idx + 1 < len(tagged):
                    val = extract_mangled_date(tagged[idx + 1][2]["text"])
                    if val:
                        fields["date_of_birth"] = val
            break

    # --- Gender & Nationality ---
    for idx, key, el in tagged:
        if key == "gender_nat":
            norm = _strip_accents(el["text"]).lower()
            # Gender
            if "gender" not in fields:
                if re.search(r'\bnam\b', norm):
                    fields["gender"] = "Nam"
                elif re.search(r'\bnu\b', norm) or "nữ" in el["text"].lower():
                    fields["gender"] = "Nữ"
            # Nationality
            if "viet" in norm:
                fields["nationality"] = "Việt Nam"
            break

    # Gender fallback: search for standalone "Nam"/"Nữ" near DOB line
    if "gender" not in fields:
        dob_y = None
        for idx, key, el in tagged:
            if key == "dob":
                dob_y = el["center"][1]
                break
        if dob_y is not None:
            for el in lines:
                if abs(el["center"][1] - dob_y) < 80:
                    text_stripped = el["text"].strip()
                    if re.match(r'^(Nam|nam|NAM)$', text_stripped):
                        fields["gender"] = "Nam"
                    elif re.match(r'^(Nữ|nữ|NỮ|Nu|NU)$', text_stripped):
                        fields["gender"] = "Nữ"

    # --- Place of Origin & Residence ---
    # Find label indices
    origin_idx = residence_idx = expiry_idx = None
    for idx, key, el in tagged:
        if key == "origin" and origin_idx is None:
            origin_idx = idx
        elif key == "residence" and residence_idx is None:
            residence_idx = idx
        elif key == "expiry" and expiry_idx is None:
            expiry_idx = idx

    def _collect_value_block(start_idx: int, stop_idx: Optional[int]) -> str:
        """Collect value lines between start label and stop label/end."""
        if start_idx is None or start_idx >= len(tagged):
            return ""

        start_el = tagged[start_idx][2]
        start_x = start_el["bbox"][0]
        start_y = start_el["center"][1]
        
        # Stop Y: Find the next label that is in the SAME X column
        stop_y = start_y + 300  # generous default
        x_tolerance = max(start_el["height"] * 1.5, 30)
        for j in range(start_idx + 1, len(tagged)):
            if tagged[j][1] is not None:
                if abs(tagged[j][2]["bbox"][0] - start_x) <= x_tolerance:
                    stop_y = tagged[j][2]["center"][1]
                    break

        parts = []
        
        # Extract inline value by stripping all label keywords
        text_clean = _strip_all_labels(start_el["text"])
        if text_clean and len(text_clean) > 3 and not _is_garbage(text_clean):
            parts.append(text_clean)

        # Collect lines below the label, same X column
        for j in range(start_idx + 1, len(tagged)):
            j_key, j_el = tagged[j][1], tagged[j][2]
            
            # Stop if line is too far below
            if j_el["center"][1] >= stop_y - 5:  # -5 for safety margin
                break

            # If it's a label, skip it (we already checked it's not in the same column, otherwise stop_y would have caught it)
            if j_key is not None:
                continue

            # X-column check: value must be roughly in the same column or to the right
            # We don't want to include the left column (Expiry Date) when parsing Residence.
            # But we don't strictly bound the right side because addresses can be long.
            if j_el["bbox"][0] < start_x - x_tolerance:
                continue

            # Skip if it is a label in the left column that happened to bypass stop_y
            if j_key is not None and j_el["bbox"][0] < start_x - x_tolerance:
                continue

            text = j_el["text"].strip()
            if _is_garbage(text):
                continue

            parts.append(text)

        address = " ".join(parts).strip()
        address = address.replace(" ,", ",").strip()
        # Final sanity: if result doesn't look like an address, return empty
        if address and not _looks_like_address(address):
            return ""
        return address

    if origin_idx is not None:
        fields["place_of_origin"] = _collect_value_block(origin_idx, residence_idx)
    if residence_idx is not None:
        fields["place_of_residence"] = _collect_value_block(residence_idx, expiry_idx)

    # --- Date of Expiry: collect from expiry label or left-column ---
    for idx, key, el in tagged:
        if key == "expiry":
            val = extract_mangled_date(el["text"])
            if val:
                fields["date_of_expiry"] = val
            else:
                # Check next line
                if idx + 1 < len(tagged):
                    val = extract_mangled_date(tagged[idx + 1][2]["text"])
                    if val:
                        fields["date_of_expiry"] = val
            break

    # Expiry fallback: find all dates, take the last one that isn't DOB
    if "date_of_expiry" not in fields:
        all_dates = date_pattern.findall(full_text)
        dob = fields.get("date_of_birth", "")
        for d in reversed(all_dates):
            d_norm = d.replace(".", "/").replace("-", "/")
            if d_norm != dob:
                fields["date_of_expiry"] = d_norm
                break

    return fields


# ============================================================
# Step 4: CCCD Back-side layout parser
# ============================================================

def parse_cccd_back(elements: List[Dict]) -> Dict[str, str]:
    """Parse CCCD back side using bbox layout."""
    fields: Dict[str, str] = {}
    if not elements:
        return fields

    # 1. Sort elements top-to-bottom, then left-to-right
    lines = sorted(elements, key=lambda e: (e["center"][1], e["bbox"][0]))
    
    # Calculate height for each line if missing
    for el in lines:
        if "height" not in el:
            el["height"] = el["bbox"][3] - el["bbox"][1]
    full_text = " ".join(el["text"] for el in lines)
    date_pattern = re.compile(r'\d{2}[/.\-]\d{2}[/.\-]\d{4}')

    tagged: List[Tuple[int, Optional[str], Dict]] = []
    for i, el in enumerate(lines):
        key = _classify_line(el["text"], BACK_LABELS)
        tagged.append((i, key, el))

    # --- Residence ---
    for idx, key, el in tagged:
        if key == "residence":
            parts = []
            inline = _extract_inline_value(el["text"])
            if inline and not _is_garbage(inline):
                parts.append(inline)
            # Collect below until next label
            for j in range(idx + 1, len(tagged)):
                if tagged[j][1] is not None:
                    break
                text = tagged[j][2]["text"].strip()
                if _is_garbage(text):
                    continue
                parts.append(text)
            if parts:
                val = " ".join(parts)
                val = _strip_all_labels(val)
                fields["place_of_residence"] = val.replace(" ,", ",").strip()
            break

    # --- Place of Birth ---
    for idx, key, el in tagged:
        if key == "birth_place":
            parts = []
            inline = _extract_inline_value(el["text"])
            if inline and not _is_garbage(inline):
                parts.append(inline)
            for j in range(idx + 1, len(tagged)):
                if tagged[j][1] is not None:
                    break
                text = tagged[j][2]["text"].strip()
                if _is_garbage(text):
                    continue
                parts.append(text)
            if parts:
                val = " ".join(parts)
                val = _strip_all_labels(val)
                fields["place_of_birth"] = val.replace(" ,", ",").strip()
            break

    # --- Date of Issue ---
    for idx, key, el in tagged:
        if key == "issue_date":
            match = date_pattern.search(el["text"])
            if match:
                fields["date_of_issue"] = match.group(0).replace(".", "/").replace("-", "/")
            elif idx + 1 < len(tagged):
                match = date_pattern.search(tagged[idx + 1][2]["text"])
                if match:
                    fields["date_of_issue"] = match.group(0).replace(".", "/").replace("-", "/")
            break

    # --- Date of Expiry ---
    for idx, key, el in tagged:
        if key == "expiry_date":
            match = date_pattern.search(el["text"])
            if match:
                fields["date_of_expiry"] = match.group(0).replace(".", "/").replace("-", "/")
            elif idx + 1 < len(tagged):
                match = date_pattern.search(tagged[idx + 1][2]["text"])
                if match:
                    fields["date_of_expiry"] = match.group(0).replace(".", "/").replace("-", "/")
            break

    # --- Place of Issue (issuing authority) ---
    KNOWN_ISSUERS = [
        "CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI",
        "BỘ CÔNG AN",
        "CÔNG AN THÀNH PHỐ HÀ NỘI",
        "CÔNG AN THÀNH PHỐ HỒ CHÍ MINH",
        "CÔNG AN THÀNH PHỐ ĐÀ NẴNG",
        "CÔNG AN THÀNH PHỐ HẢI PHÒNG",
        "CÔNG AN THÀNH PHỐ CẦN THƠ",
    ]
    for idx, key, el in tagged:
        if key == "issuer":
            raw_text = el["text"]
            norm = _strip_accents(raw_text).upper()
            if "BO CONG AN" in norm:
                fields["place_of_issue"] = "BỘ CÔNG AN"
            elif "CANH SAT" in norm or "CUC CANH" in norm:
                fields["place_of_issue"] = "CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI"
            else:
                cleaned = re.sub(r'(?i)(CỤC\s*TRƯỞNG|CUC\s*TRUONG|GIÁM\s*ĐỐC|GIAM\s*DOC|CỤC\s*TRƯỜNG)', '', raw_text).strip()
                cleaned = re.sub(r'(?i)/?\s*MINISTRY.*$', '', cleaned).strip()
                from rapidfuzz import process as rfprocess
                match = rfprocess.extractOne(cleaned, KNOWN_ISSUERS, scorer=fuzz.partial_ratio, score_cutoff=60.0)
                if match:
                    fields["place_of_issue"] = match[0]
                else:
                    fields["place_of_issue"] = raw_text
            break

    return fields


# ============================================================
# Helpers
# ============================================================

_GARBAGE_PATTERNS = [
    re.compile(r'\d{2}[/.\-]\d{2}[/.\-]\d{4}'),     # dates
    re.compile(r'\b\d{10,}\b'),                        # long digit sequences
    re.compile(r'(?i)co gia tri'),                     # expiry label
    re.compile(r'(?i)date of ?expiry'),
    re.compile(r'(?i)citizen identity'),
    re.compile(r'(?i)(IDVNM|VNMCCS|BUICK)'),          # MRZ zone
]

# Common OCR garbage words (accent-stripped, lowercase) from noisy card edges
_GARBAGE_WORDS = {
    "duoc", "diem", "den", "dat", "dai", "anh", "ron",
    "truong", "nguoi", "gia", "the", "nay", "cho", "voi",
    "theo", "cua", "mot", "cac", "cung", "bay", "khong",
    "lam", "rat", "hay", "nhu", "bao", "qua", "nua",
}

def _is_garbage(text: str) -> bool:
    """Check if a text line is garbage (dates, IDs, labels, MRZ, or OCR noise)."""
    for pat in _GARBAGE_PATTERNS:
        if pat.search(text):
            return True
    
    words = text.strip().split()
    if not words:
        return True
    
    # Very short text with only 1-2 chars is garbage
    stripped = text.strip()
    if len(stripped) <= 2:
        return True
    
    # If the text contains too many common-word garbage (not proper nouns), it's noise
    norm_words = [_strip_accents(w).lower() for w in words]
    garbage_count = sum(1 for w in norm_words if w in _GARBAGE_WORDS)
    if len(words) >= 3 and garbage_count / len(words) > 0.4:
        return True
    
    return False


def _looks_like_address(text: str) -> bool:
    """Heuristic: does this text look like a Vietnamese address component?
    
    Vietnamese addresses typically contain:
    - Proper nouns (capitalized words)
    - Place markers: Ấp, Xã, Phường, TT, TP, Quận, Huyện
    - Commas separating hierarchical units
    """
    if not text or len(text.strip()) < 3:
        return False
    
    # Check for address markers
    address_markers = re.compile(
        r'(?i)(ấp|xã|phường|thị trấn|tt\b|tp\b|quận|huyện|thành phố|tỉnh|'
        r'khánh|hòa|hoà|châu|long|mỹ|đại|an|sóc|trăng|cần|thơ|giang|'
        r'thạnh|trị|xuyên|phú|tân|ngọc|văn|chính)',
        re.IGNORECASE
    )
    if address_markers.search(text):
        return True
    
    # Has commas (typical of hierarchical addresses)
    if ',' in text:
        return True
    
    # Mostly capitalized words (proper nouns)
    words = text.split()
    cap_count = sum(1 for w in words if w and w[0].isupper())
    if len(words) >= 2 and cap_count / len(words) >= 0.5:
        return True
    
    return False


_LABEL_FRAG_TOKENS = {
    "place", "of", "origin", "birth", "residence", "ongin", "ace", "ce",
    "esidence", "irth", "oforigin", "ofresidence", "ofbirth",
    "noi", "thuong", "tru", "que", "quan", "khai", "sinh", "cu",
    "ho", "va", "ten", "ngay", "nam", "gioi", "tinh", "quoc", "tich",
    "full", "name", "date", "sex", "nationality", "identity", "card",
}

def _strip_all_labels(text: str) -> str:
    """Aggressively remove all label keywords from the text, regardless of position."""
    if not text:
        return text
    
    # 1. Remove common multi-word label blocks using regex
    patterns = [
        r'(?i)place\s+of\s*(origin|residence|birth|ongin)',
        r'(?i)noi\s+thuong\s+tru',
        r'(?i)que\s+quan',
        r'(?i)noi\s+(dang\s+ky\s+)?khai\s+sinh',
        r'(?i)noi\s+cu\s+tru',
        r'(?i)ho\s+va\s+ten',
        r'(?i)full\s+name',
        r'(?i)ngay\s+sinh',
        r'(?i)date\s+of\s+birth',
    ]
    for p in patterns:
        text = re.sub(p, '', text)
        
    # 2. Clean up punctuation left behind (like ':', '/', 'I', '-')
    text = re.sub(r'^[\s/:;\-I|]+', '', text)
    
    # 3. Strip individual leftover garbage tokens from the START of the string
    words = text.split()
    while words and _strip_accents(words[0]).lower() in _LABEL_FRAG_TOKENS:
        words.pop(0)
        
    result = " ".join(words)
    result = re.sub(r'^[,;:\s/]+', '', result).strip()
    return result


# ============================================================
# Main entry point
# ============================================================

def layout_parse(elements: List[Dict], doc_type: str, image=None, ocr_engine=None) -> Dict[str, str]:
    """Parse document using layout-aware bbox extraction."""
    if doc_type in ("cccd", "cccd_front"):
        return parse_cccd_front(elements, image=image, ocr_engine=ocr_engine)
    elif doc_type == "cccd_back":
        return parse_cccd_back(elements)
    elif doc_type == "cccd_auto":
        # Auto-detect side from content
        all_text = " ".join(el["text"] for el in elements).lower()
        norm = _strip_accents(all_text)
        front_kw = ["ho va ten", "name", "sinh", "birth", "que quan", "origin"]
        back_kw = ["khai sinh", "cu tru", "cuc truong", "giam doc", "canh sat", "bo cong an", "ministry"]
        front_score = sum(1 for k in front_kw if k in norm)
        back_score = sum(1 for k in back_kw if k in norm)
        if back_score > front_score:
            return parse_cccd_back(elements)
        else:
            return parse_cccd_front(elements, image=image, ocr_engine=ocr_engine)
    return {}
