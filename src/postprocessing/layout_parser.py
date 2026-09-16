"""Layout Graph Parser v2 — bbox-aware field extraction for CCCD."""

import re
import unicodedata
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from rapidfuzz import fuzz

from src.postprocessing.cccd_id_utils import infer_birth_year_and_gender

date_pattern = re.compile(r'\d{2}[/.\-]\d{2}[/.\-]\d{4}')


def extract_mangled_date(text: str) -> str:
    if not text:
        return ""
    # Clean space between separated digits and slashes (e.g. '0 1/05/2021' -> '01/05/2021')
    clean = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    clean = re.sub(r'(\d)\s*[/.\-]\s*(\d)', r'\1/\2', clean)
    m = re.search(r'(?:[^\d]|^)(\d{1,2})/(\d{1,2})/(\d{4})', clean)
    if m:
        d, mth, y = m.group(1), m.group(2), m.group(3)
        if 1 <= int(d) <= 31 and 1 <= int(mth) <= 12 and 1900 <= int(y) <= 2099:
            return f"{int(d):02d}/{int(mth):02d}/{y}"
    m_loose = re.search(r'\b\d{4,5}[/.\-]\d{4}\b|\b\d{2}[/.\-]\d{6,7}\b', text)
    if m_loose:
        d_clean = re.sub(r'[^\d]', '', m_loose.group(0))
        if len(d_clean) >= 8:
            return f"{d_clean[:2]}/{d_clean[2:4]}/{d_clean[-4:]}"
    # Check for 8 contiguous digits DDMMYYYY without slashes (e.g. 29092022 -> 29/09/2022)
    m_8digits = re.search(r'\b(0[1-9]|[12]\d|3[01])(0[1-9]|1[0-2])(19\d{2}|20\d{2})\b', clean)
    if m_8digits:
        return f"{m_8digits.group(1)}/{m_8digits.group(2)}/{m_8digits.group(3)}"
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

# Front-side labels. Keep only the canonical Vietnamese/English wording plus
# genuinely distinct alternate phrasings here — OCR *typo* variants of these
# (missing spaces, misread letters) are handled generically by the fuzzy
# fallback in _classify_line, not by hardcoding every misread we've seen.
FRONT_LABELS = {
    "id_number":    ["so", "no", "dinh danh", "personal identification"],
    "full_name":    ["ho va ten", "full name", "ho ten", "ho, chu dem va ten"],
    "dob":          ["ngay sinh", "date of birth", "ngay, thang, nam sinh", "ngay thang nam sinh", "nam sinh", "birth"],
    "gender_nat":   ["gioi tinh", "sex", "quoc tich", "nationality"],
    "origin":       ["que quan", "place of origin"],
    "residence":    ["thuong tru", "place of residence", "noi cu tru", "cu tru"],
    "expiry":       ["co gia tri den", "co gia tri", "date of expiry", "het han", "gia tri den"],
}

# Back-side labels
BACK_LABELS = {
    "residence":    ["noi cu tru", "place of residence", "cu tru"],
    "birth_place":  ["khai sinh", "place of birth", "dang ky khai"],
    "issue_date":   ["ngay thang nam", "date of issue", "ngay cap", "date,month"],
    "expiry_date":  ["het han", "date of expiry"],
    "issuer":       ["bo cong an", "ministry", "cuc canh sat", "cuc truong", "giam doc", "nguyen quoc hung", "quoc hung", "to van hue", "vu xuan dung"],
}


def _classify_line(text: str, label_set: Dict[str, List[str]]) -> Optional[str]:
    """Return the label key if this line contains a label, else None."""
    norm = _strip_accents(text).lower()
    norm_padded = f" {norm} "

    # Priority check for dob vs other labels (front-side only — BACK_LABELS has
    # no "dob" key and uses "khai sinh"/"birth_place" for birth-related lines)
    if "dob" in label_set and "sinh" in norm and not any(w in norm for w in ["khai sinh", "noi sinh"]):
        if any(w in norm for w in ["ngay", "nam", "thang", "birth", "date"]):
            return "dob"

    # Pass 1: exact / substring match against canonical keywords — always
    # wins over fuzzy, and runs across all keys first so an early key's
    # fuzzy score can't preempt a later key's exact match.
    for key, keywords in label_set.items():
        for kw in keywords:
            if f" {kw} " in norm_padded:
                return key
            if norm.startswith(kw + " ") or norm.endswith(" " + kw):
                return key
            if norm == kw:
                return key

    # Pass 2: fuzzy fallback for OCR misreads of a canonical keyword (missing
    # spaces, swapped/dropped letters) — generalizes to typos we haven't seen
    # before instead of requiring every variant to be hardcoded.
    best_key, best_score = None, 0.0
    for key, keywords in label_set.items():
        for kw in keywords:
            if len(kw) < 8:
                continue
            score = fuzz.partial_ratio(kw, norm)
            if score > best_score:
                best_key, best_score = key, score
    if best_score > 80:
        return best_key
    return None


def _extract_inline_value(text: str) -> str:
    """Extract value part from a label line (after :, /, or English label prefix)."""
    # 1. Try colon first
    if ":" in text:
        val = text.split(":", 1)[1].strip()
        if len(val) > 2:
            return val

    # 2. Try slash (separates bilingual VN/EN label, e.g. "Nơi cư trú / Place of residence Quan Đinh Nam")
    parts = text.split("/")
    if len(parts) >= 2:
        last = parts[-1].strip()
        # Case A: English label prefix followed by value
        m = re.match(
            r'(?i)^(?:(?:p[li1a]ace|piace|placo|phaco|pace)?\s*(?:of|ool)?\s*(?:residence|redence|sedines|sedence|origin|birth|ongin)|ofresidence|oforigin|ofbirth|full\s*name|date\s*of\s*(?:birth|expiry|issue)|sex|nationality|no\.?)\s*[:;\-]?\s*(.+)$',
            last
        )
        if m:
            val = m.group(1).strip()
            if len(val) > 2:
                return val
        # Case B: The segment after slash is purely the value (no label keywords)
        norm = _strip_accents(last).lower()
        label_words = {"name", "birth", "origin", "residence", "sex", "nationality",
                       "expiry", "issue", "no", "oforigin", "ofresidence", "ofbirth"}
        if not any(lw in norm for lw in label_words) and len(last) > 2:
            return last

    return ""


def _refine_name_with_vision(crop: Optional[np.ndarray], name_text: str, ocr_engine: Optional[Any]) -> str:
    """Refine Vietnamese full name diacritics using targeted high-resolution vision sub-crops.
    
    Long text lines squashed into 32px height by Seq2Seq models can blur or truncate stacked diacritics
    (such as circumflex + tilde: Ễ). This optical verifier crops candidate words with ambiguous accents
    and runs a high-resolution sub-crop check, accepting optical refinement only when the base word matches.
    """
    if crop is None or not name_text or ocr_engine is None:
        return name_text
    words = name_text.split()
    if not words:
        return name_text

    H, W = crop.shape[:2]
    if W < 20 or H < 8:
        return name_text

    import cv2
    L = sum(len(w) for w in words) + len(words) - 1

    refined_words = []
    pos = 0
    for w in words:
        start_char = pos
        end_char = pos + len(w)
        pos = end_char + 1

        norm_w = _strip_accents(w).lower()
        # Words that could have stacked diacritics in Vietnamese names
        # (e.g. circumflex vowels: ê, â, ô, or ambiguous base words like nguyen)
        needs_check = any(c in w.lower() for c in ["ê", "â", "ô", "ư", "ơ"]) or norm_w == "nguyen"
        if not needs_check:
            refined_words.append(w)
            continue

        x1 = max(0, int((start_char - 0.5) / L * W))
        x2 = min(W, int((end_char + 0.5) / L * W))
        sub = crop[:, x1:x2]
        if sub.shape[1] < 10 or sub.shape[0] < 8:
            refined_words.append(w)
            continue

        padded = cv2.copyMakeBorder(sub, 4, 2, 4, 4, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        try:
            sub_pred = ocr_engine.recognize_crop_vietocr(padded).strip()
            # If the vision sub-crop cleanly recognized the exact same base word with different diacritics
            if _strip_accents(sub_pred).lower() == norm_w and sub_pred != w:
                refined_words.append(sub_pred.upper() if w.isupper() else sub_pred)
            else:
                refined_words.append(w)
        except Exception:
            refined_words.append(w)

    return " ".join(refined_words)


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
    if not (cand and FieldValidator.is_valid_cccd_id(cand)):
        # Cross-reference with RapidOCR results which don't hallucinate extra digits
        rapid_full = " ".join(el.get("rapid_text", "") for el in lines)
        m_rapid = re.search(r'\b\d{12}\b', rapid_full)
        if not m_rapid:
            m_rapid = re.search(r'\d{12}', rapid_full.replace(" ", ""))
        if m_rapid and FieldValidator.is_valid_cccd_id(m_rapid.group(0)):
            cand = m_rapid.group(0)
    
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
            cand_name = None
            cand_crop = None
            inline = _extract_inline_value(el["text"])
            if inline and len(inline) > 3 and not re.search(r'\d', inline) and not _is_garbage(inline):
                cand_name = inline
                cand_crop = el.get("crop")
            else:
                label_xmin = el["bbox"][0]
                label_ymin = el["bbox"][1]
                label_ymax = el["bbox"][3]
                for j in range(idx + 1, min(idx + 6, len(tagged))):
                    j_key, j_el = tagged[j][1], tagged[j][2]
                    if j_key is not None:
                        continue # Skip other section labels (e.g. dob)
                    cand_text = j_el["text"].strip()
                    if _is_garbage(cand_text) or re.search(r'\d', cand_text):
                        continue
                    cand_xmin = j_el["bbox"][0]
                    cand_ymin = j_el["bbox"][1]
                    # Must align horizontally within reasonable column offset
                    if abs(cand_xmin - label_xmin) > 120:
                        continue
                    # Must be positioned immediately below the label
                    if cand_ymin < label_ymin - 5 or cand_ymin > label_ymax + 100:
                        continue
                    cand_name = cand_text
                    cand_crop = j_el.get("crop")
                    break
            if cand_name:
                cand_name = _refine_name_with_vision(cand_crop, cand_name, ocr_engine)
                fields["full_name"] = cand_name
            break

    # --- Date of Birth ---
    for idx, key, el in tagged:
        if key == "dob":
            val = extract_mangled_date(el["text"])
            if val:
                fields["date_of_birth"] = val
            else:
                # Check next 1-2 lines
                for next_offset in (1, 2):
                    if idx + next_offset < len(tagged):
                        val = extract_mangled_date(tagged[idx + next_offset][2]["text"])
                        if val:
                            fields["date_of_birth"] = val
                            break
            break

    # DOB Fallback: Cross-check with CCCD ID or dates with year < 2015
    if "date_of_birth" not in fields:
        all_dates = date_pattern.findall(full_text)
        id_num = fields.get("id_number", "")
        id_info = infer_birth_year_and_gender(id_num)
        expected_birth_year = id_info[0] if id_info else None

        if expected_birth_year:
            for d in all_dates:
                d_norm = d.replace(".", "/").replace("-", "/")
                if d_norm.endswith(f"/{expected_birth_year}"):
                    fields["date_of_birth"] = d_norm
                    break

        if "date_of_birth" not in fields:
            for d in all_dates:
                d_norm = d.replace(".", "/").replace("-", "/")
                try:
                    yr = int(d_norm.split("/")[-1])
                    if yr < 2015:
                        fields["date_of_birth"] = d_norm
                        break
                except (ValueError, IndexError):
                    pass

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
        
        def _restore_slash(val: str, rapid_raw: Optional[str]) -> str:
            if not val or not rapid_raw:
                return val
            for m in re.finditer(r'\b(\d{1,4})/(\d{1,4})\b', rapid_raw):
                n1, n2 = m.group(1), m.group(2)
                mangled = f"{n1}1{n2}"
                if mangled in val:
                    val = val.replace(mangled, f"{n1}/{n2}")
            return val

        # Extract inline value by stripping all label keywords
        # If line has colon, value is strictly after colon. Otherwise check if non-label address text remains.
        if ":" in start_el["text"]:
            after_colon = _strip_all_labels(start_el["text"].split(":", 1)[1]).strip()
            after_colon = _restore_slash(after_colon, start_el.get("rapid_text"))
            if after_colon and len(after_colon) > 3 and not _is_garbage(after_colon):
                parts.append(after_colon)
        else:
            text_clean = _strip_all_labels(start_el["text"]).strip()
            text_clean = _restore_slash(text_clean, start_el.get("rapid_text"))
            if text_clean and len(text_clean) > 3 and not _is_garbage(text_clean):
                norm_clean = _strip_accents(text_clean).lower()
                if norm_clean not in ("que quan", "queguan", "place of origin", "placeoforigin", "noi thuong tru", "place of residence", "noi thurong tru"):
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
            text = _restore_slash(text, j_el.get("rapid_text"))
            text = re.sub(r'\b[ÁÂA]p\b', 'Ấp', text)
            if _is_garbage(text):
                continue

            # Deduplication: do not append identical or duplicate lines
            norm_text = _strip_accents(text).lower()
            if any(norm_text == _strip_accents(p).lower() or fuzz.ratio(norm_text, _strip_accents(p).lower()) > 85 for p in parts):
                continue

            parts.append(text)

        # Smart join address parts: insert comma if previous part was a unit or next part starts an admin unit
        if parts:
            address = parts[0].strip().rstrip(",")
            for p in parts[1:]:
                p_clean = p.strip()
                if not p_clean:
                    continue
                prev_is_unit = bool(re.search(r'\b(Ấp|Áp|Âp|Ap|Thôn|Bản|Khu\s+phố|Tổ|Khóm|Số|Đường|Phố)\b', address, re.IGNORECASE))
                curr_starts_unit = bool(re.match(r'^(?:Ấp|Áp|Âp|Ap|Xã|Phường|P\b|P\.|TT\b|TT\.|TX\b|TX\.|TP\b|TP\.|Quận|Q\b|Q\.|Huyện|H\b|H\.|Tỉnh)\b', p_clean, re.IGNORECASE))
                if prev_is_unit or curr_starts_unit or address.endswith(","):
                    address = f"{address}, {p_clean.lstrip(',')}"
                else:
                    address = f"{address} {p_clean}"
        else:
            address = ""
        from src.postprocessing.address_norm import clean_address_string, deduplicate_address_segments
        address = clean_address_string(address)
        address = deduplicate_address_segments(address)
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

    # Expiry fallback: find all dates, take the last one that isn't DOB and has year >= 2015
    if "date_of_expiry" not in fields:
        all_dates = date_pattern.findall(full_text)
        dob = fields.get("date_of_birth", "")
        for d in reversed(all_dates):
            d_norm = d.replace(".", "/").replace("-", "/")
            if d_norm != dob:
                try:
                    yr = int(d_norm.split("/")[-1])
                    if yr >= 2015:
                        fields["date_of_expiry"] = d_norm
                        break
                except (ValueError, IndexError):
                    pass

    return fields


# ============================================================
# Step 4: CCCD Back-side layout parser
# ============================================================

def parse_cccd_back(elements: List[Dict], image=None, ocr_engine=None) -> Dict[str, str]:
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

    # 0. Fast MRZ Extraction (TD1 format on CCCD back)
    try:
        from src.postprocessing.mrz_parser import parse_mrz_lines
        raw_texts = [el["text"] for el in lines]
        mrz_data = parse_mrz_lines(raw_texts)
        for k, v in mrz_data.items():
            if v:
                fields[k] = v
    except Exception:
        pass

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
                val = ", ".join(p.strip().rstrip(",") for p in parts if p.strip())
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
                val = ", ".join(p.strip().rstrip(",") for p in parts if p.strip())
                val = _strip_all_labels(val)
                fields["place_of_birth"] = val.replace(" ,", ",").strip()
            break

    # --- Date of Issue ---
    if "date_of_issue" not in fields:
        for idx, key, el in tagged:
            if key == "issue_date":
                val = extract_mangled_date(el["text"])
                if not val and idx + 1 < len(tagged):
                    val = extract_mangled_date(tagged[idx + 1][2]["text"])
                if val:
                    fields["date_of_issue"] = val
                break

    # Fallback for date_of_issue: scan any date on back side between 2014 and 2026
    if "date_of_issue" not in fields:
        for el in lines:
            val = extract_mangled_date(el["text"])
            if val and val != fields.get("date_of_expiry"):
                try:
                    yr = int(val.split("/")[-1])
                    if 2014 <= yr <= 2026:
                        fields["date_of_issue"] = val
                        break
                except (ValueError, IndexError):
                    pass

    # Fallback for date_of_issue using targeted crop and CLAHE if image is available
    if "date_of_issue" not in fields and image is not None:
        try:
            import cv2
            from rapidocr_onnxruntime import RapidOCR
            rocr = RapidOCR()
            h, w = image.shape[:2]

            # Strategy 1: Targeted strip directly above CANH SAT / issuer / director
            issuer_el = next((el for _, key, el in tagged if key == "issuer" or any(k in _strip_accents(el["text"]).upper() for k in ["CANH SAT", "CUC TRUONG", "QUOC HUNG", "VAN HUE", "TRAT TU"])), None)
            if issuer_el:
                c_ymin = int(issuer_el["bbox"][1])
                c_xmin = max(0, int(issuer_el["bbox"][0] - 150))
                c_xmax = min(w, int(issuer_el["bbox"][2] + 250))
                crop_above = image[max(0, c_ymin - 250):min(h, c_ymin + 20), c_xmin:c_xmax]
                if crop_above.size > 0:
                    res_above, _ = rocr(crop_above)
                    for r in res_above or []:
                        val = extract_mangled_date(r[1])
                        if val and val != fields.get("date_of_expiry"):
                            fields["date_of_issue"] = val
                            break

            # Strategy 2: Targeted CLAHE on top half if still not found
            if "date_of_issue" not in fields:
                top_half = image[:int(h * 0.6), :]
                gray = cv2.cvtColor(top_half, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl_bgr = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
                clahe_res, _ = rocr(cl_bgr)
                for r in clahe_res or []:
                    val = extract_mangled_date(r[1])
                    if val and val != fields.get("date_of_expiry"):
                        try:
                            yr = int(val.split("/")[-1])
                            if 2014 <= yr <= 2029:
                                fields["date_of_issue"] = val
                                break
                        except (ValueError, IndexError):
                            pass
        except Exception:
            pass

    # --- Date of Expiry ---
    if "date_of_expiry" not in fields:
        for idx, key, el in tagged:
            if key == "expiry_date":
                val = extract_mangled_date(el["text"])
                if not val and idx + 1 < len(tagged):
                    val = extract_mangled_date(tagged[idx + 1][2]["text"])
                if val:
                    fields["date_of_expiry"] = val
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
            elif any(k in norm for k in ["CANH SAT", "CUC CANH", "QUOC HUNG", "VAN HUE", "XUAN DUNG", "TRAT TU"]):
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

    # Authoritative fallback for chip-based cards or clear C06 markers
    if fields.get("place_of_issue") not in KNOWN_ISSUERS:
        all_back_text = " ".join(f"{el.get('text', '')} {el.get('rapid_text', '')}" for el in lines)
        norm_back = _strip_accents(all_back_text).upper()
        if any(k in norm_back for k in ["BO CONG AN", "MINISTRY OF PUBLIC SECURITY"]):
            fields["place_of_issue"] = "BỘ CÔNG AN"
        elif any(k in norm_back for k in ["IDVNM", "<<", "QUOC HUNG", "VAN HUE", "XUAN DUNG", "CANH SAT", "TRAT TU XA HOI", "CUC CANH"]):
            fields["place_of_issue"] = "CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI"

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
    re.compile(r'(?i)\b(ctrl|ctri|crtl|alt|shift|shif|esc|tab|caps|f\d+|enter|backspace|delete|copy|edit|exit|save|blend|layer|mask)[\d+a-z]*'), # keyboard noise
    re.compile(r'[+<>=~*^|\\]'),                      # shortcut / symbol noise
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
    "noi", "thuong", "tru", "que", "khai", "cu",
    "ho", "va", "ten", "ngay", "gioi",
    "full", "name", "date", "sex", "nationality", "identity", "card",
    "queguan", "placeoforigin", "placeofresidence", "thurong", "i"
}

def _strip_all_labels(text: str) -> str:
    """Aggressively remove all label keywords from the text, regardless of position."""
    if not text:
        return text
    
    # Strip boundary duplicate digit from misread colon attached to residence/origin label
    text = re.sub(r'(?i)\b(residence|origin|birth|tru|quán|quan)[.:;\s\-]*([12])(?=\s*\2\d+)', r'\1 ', text)
    
    # 1. Remove common multi-word label blocks using regex
    patterns = [
        r'(?i)\b(?:[nr]ói|[nr]ơi|[nr]oi|rồi)\s+(?:thương|thường|thuong|thurong)\s+(?:trú|tru|trui)\b',
        r'(?i)\b(?:place|phaco|placo|pace|placeool)\s*(?:of|ool)?\s*(?:residence|nadence|redence|sedines|sedence)\b',
        r'(?i)place\s*of\s*(origin|residence|birth|ongin)',
        r'(?i)noi\s*(thuong|thurong)\s*tru',
        r'(?i)nơi\s*(thường|thuong)\s*trú',
        r'(?i)que\s*[qg]uan',
        r'(?i)quê\s*[qg]uán',
        r'(?i)noi\s*(dang\s*ky\s*)?khai\s*sinh',
        r'(?i)nơi\s*(đăng\s*ký\s*)?khai\s*sinh',
        r'(?i)noi\s*cu\s*tru',
        r'(?i)nơi\s*cư\s*trú',
        r'(?i)ho\s*va\s*ten',
        r'(?i)họ\s*và\s*tên',
        r'(?i)full\s*name',
        r'(?i)ngay\s*sinh',
        r'(?i)ngày\s*sinh',
        r'(?i)date\s*of\s*birth',
        r'(?i)co\s*gia\s*tri\s*den',
        r'(?i)có\s*giá\s*trị\s*đến',
        r'(?i)date\s*of\s*expiry',
        r'(?i)citizen\s*identity\s*card',
        r'(?i)can\s*cuoc\s*cong\s*dan',
        r'(?i)căn\s*cước\s*công\s*dân',
    ]
    for p in patterns:
        text = re.sub(p, '', text)
        
    # 2. Clean up punctuation left behind (like ':', '/', 'I', '-')
    text = re.sub(r'^[\s/:;\-I|.]+', '', text)
    text = re.sub(r'[\s/:;\-I|.]+$', '', text)
    
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
        return parse_cccd_back(elements, image=image, ocr_engine=ocr_engine)
    elif doc_type == "cccd_auto":
        # Auto-detect side from content
        all_text = " ".join(el["text"] for el in elements).lower()
        norm = _strip_accents(all_text)
        front_kw = ["ho va ten", "name", "sinh", "birth", "que quan", "origin"]
        back_kw = ["khai sinh", "cu tru", "cuc truong", "giam doc", "canh sat", "bo cong an", "ministry"]
        front_score = sum(1 for k in front_kw if k in norm)
        back_score = sum(1 for k in back_kw if k in norm)
        if back_score > front_score:
            return parse_cccd_back(elements, image=image, ocr_engine=ocr_engine)
        else:
            return parse_cccd_front(elements, image=image, ocr_engine=ocr_engine)
    return {}
