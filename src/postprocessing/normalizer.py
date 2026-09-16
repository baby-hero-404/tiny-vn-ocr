"""Text normalization, character substitution, regex validation, and fuzzy matching."""

from typing import List, Optional
import re
from rapidfuzz import process, fuzz

DIGIT_CONFUSION_MAP = {
    "O": "0",
    "o": "0",
    "I": "1",
    "i": "1",
    "l": "1",
    "|": "1",
    "S": "5",
    "s": "5",
    "Z": "2",
    "z": "2",
    "B": "8",
    "G": "6",
    "D": "0",
}

NUMERIC_FIELDS = {"id_number", "license_number", "engine_number", "chassis_number"}
DATE_FIELDS = {"date_of_birth", "date_of_issue", "date_of_expiry"}
STANDARD_NATIONALITIES = ["VIET NAM", "VIỆT NAM"]
STANDARD_GENDERS = ["NAM", "NỮ", "NU", "NƯ"]
STANDARD_LICENSE_CLASSES = ["A1", "A2", "A3", "A4", "A", "B1", "B2", "B", "C", "D", "E", "F", "FB2", "FC", "FD", "FE"]


def normalize_digits(text: str) -> str:
    """Replace common OCR character confusions with digits."""
    result = []
    for char in text:
        if char in DIGIT_CONFUSION_MAP:
            result.append(DIGIT_CONFUSION_MAP[char])
        else:
            result.append(char)
    cleaned = "".join(result)
    return re.sub(r"[^\d]", "", cleaned)


def normalize_date(text: str) -> str:
    """Normalize date format to DD/MM/YYYY."""
    cleaned = re.sub(r"[\.\-\s]+", "/", text.strip())
    parts = cleaned.split("/")

    if len(parts) == 3:
        day = normalize_digits(parts[0]).zfill(2)
        month = normalize_digits(parts[1]).zfill(2)
        year = normalize_digits(parts[2])

        if len(day) == 2 and len(month) == 2 and len(year) == 4:
            return f"{day}/{month}/{year}"

    digits = normalize_digits(cleaned)
    if len(digits) == 8:
        return f"{digits[:2]}/{digits[2:4]}/{digits[4:]}"

    return cleaned


def fuzzy_correct(text: str, dictionary: List[str], score_cutoff: float = 60.0) -> str:
    """Perform fuzzy matching against a list of valid candidate strings."""
    if not text or not dictionary:
        return text

    match = process.extractOne(text, dictionary, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
    if match:
        return match[0]
    return text


KNOWN_QUAN_DISTRICTS = [
    "Ba Đình", "Bình Thuỷ", "Bình Thạnh", "Bình Tân", "Bắc Từ Liêm", "Cái Răng",
    "Cầu Giấy", "Cẩm Lệ", "Dương Kinh", "Gò Vấp", "Hai Bà Trưng", "Hoàn Kiếm",
    "Hoàng Mai", "Hà Đông", "Hải An", "Hải Châu", "Hồng Bàng", "Kiến An",
    "Liên Chiểu", "Long Biên", "Lê Chân", "Nam Từ Liêm", "Ngô Quyền", "Ngũ Hành Sơn",
    "Ninh Kiều", "Phú Nhuận", "Sơn Trà", "Thanh Khê", "Thanh Xuân", "Thốt Nốt",
    "Tân Bình", "Tân Phú", "Tây Hồ", "Ô Môn", "Đống Đa", "Đồ Sơn",
]


def normalize_quan_district(text: str) -> str:
    """Normalize 'Quan' to 'Quận' only when it refers to a genuine administrative district (by number or official name)."""
    import unicodedata
    # 1. Numbered districts: Quan 1 -> Quận 1
    text = re.sub(r'\b[Qq]uan\s+(\d+)\b', r'Quận \1', text)
    # 2. Named official districts
    for d in sorted(KNOWN_QUAN_DISTRICTS, key=len, reverse=True):
        d_words = d.split()
        word_patterns = []
        for w in d_words:
            nfkd = unicodedata.normalize("NFD", w)
            clean_w = "".join(c for c in nfkd if not unicodedata.combining(c)).replace("đ", "d").replace("Đ", "D")
            word_patterns.append(f"(?:{re.escape(w)}|{re.escape(clean_w)})")
        district_pat = r"\s+".join(word_patterns)
        pattern = r"\b[Qq]uan\s+(" + district_pat + r")\b"
        text = re.sub(pattern, f"Quận {d}", text, flags=re.IGNORECASE)
    return text


def normalize_field(text: str, field_type: str) -> str:
    """Normalize OCR text based on field type rules."""
    if not text:
        return ""

    cleaned = text.strip()

    if field_type in NUMERIC_FIELDS:
        cleaned = normalize_digits(cleaned)

    elif field_type in DATE_FIELDS:
        cleaned = normalize_date(cleaned)

    elif field_type == "gender":
        cleaned_up = cleaned.upper()
        cleaned_up = re.sub(r"[^\w\s]", "", cleaned_up)
        if any(k in cleaned_up for k in ["NU", "NỮ", "NƯ", "FEMALE"]):
            cleaned = "Nữ"
        elif "NAM" in cleaned_up:
            cleaned = "Nam"
        else:
            matched = fuzzy_correct(cleaned_up, STANDARD_GENDERS, score_cutoff=50.0)
            if matched in ("NỮ", "NU", "NƯ"):
                cleaned = "Nữ"
            elif matched == "NAM":
                cleaned = "Nam"

    elif field_type == "nationality":
        cleaned = cleaned.upper()
        matched = fuzzy_correct(cleaned, STANDARD_NATIONALITIES, score_cutoff=60.0)
        if "VIET" in cleaned or "VIỆT" in cleaned or "VIET" in matched or "VIỆT" in matched:
            cleaned = "VIET NAM"

    elif field_type == "license_class":
        cleaned = cleaned.upper().replace(" ", "")
        cleaned = fuzzy_correct(cleaned, STANDARD_LICENSE_CLASSES, score_cutoff=70.0)

    elif field_type in ("full_name", "owner_name"):
        cleaned = re.sub(r"[\d\._\-\+\*\?\!]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        try:
            from src.postprocessing.vn_name_dict import correct_full_name
            cleaned = correct_full_name(cleaned).upper()
        except Exception:
            cleaned = cleaned.upper()

    elif field_type in ("address", "place_of_origin", "place_of_residence", "place_of_birth", "place_of_issue"):
        try:
            from src.postprocessing.address_norm import clean_address_string
            cleaned = clean_address_string(cleaned)
        except Exception:
            cleaned = re.sub(r'[:]+', '', cleaned)
        # Fix CamelCase stuck words (e.g. ApMinhDuy -> Ap Minh Duy)
        cleaned = re.sub(r'([a-zđàáạảãèéẹẻẽìíịỉĩòóọỏõùúụủũưừứựửữơờớợởỡỳýỵỷỹ])([A-ZĐÀÁẠẢÃÈÉẸẺẼÌÍỊỈĨÒÓỌỎÕÙÚỤỦŨƯỪỨỰỬỮƠỜỚỢỞỠỲÝỴỶỸ])', r'\1 \2', cleaned)
        cleaned = re.sub(r'\b(Ap|ap)\b', 'Ấp', cleaned)
        cleaned = re.sub(r'\b(Xa|xa)\b', 'Xã', cleaned)
        cleaned = re.sub(r'\b(Phuong|phuong)\b', 'Phường', cleaned)
        cleaned = normalize_quan_district(cleaned)
        cleaned = re.sub(r'\b(Huyen|huyen)\b', 'Huyện', cleaned)
        cleaned = re.sub(r'\b(Tinh|tinh)\b', 'Tỉnh', cleaned)
        cleaned = re.sub(r'\s*,\s*', ', ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned = cleaned.replace(":", "").strip()

    return cleaned


def validate_field(text: str, field_type: str) -> bool:
    """Validate string format using regex and enum checks."""
    if not text:
        return False

    if field_type == "id_number":
        return bool(re.match(r"^\d{12}$", text))

    if field_type in DATE_FIELDS:
        return bool(re.match(r"^\d{2}/\d{2}/\d{4}$", text))

    if field_type == "gender":
        return text in ("Nam", "Nữ")

    if field_type == "license_class":
        return text in STANDARD_LICENSE_CLASSES

    return len(text) > 0
