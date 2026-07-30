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
            cleaned = "NU"
        elif "NAM" in cleaned_up:
            cleaned = "NAM"
        else:
            matched = fuzzy_correct(cleaned_up, STANDARD_GENDERS, score_cutoff=50.0)
            if matched in ("NỮ", "NU", "NƯ"):
                cleaned = "NU"
            elif matched == "NAM":
                cleaned = "NAM"

    elif field_type == "nationality":
        cleaned = cleaned.upper()
        matched = fuzzy_correct(cleaned, STANDARD_NATIONALITIES, score_cutoff=60.0)
        if "VIET" in cleaned or "VIỆT" in cleaned or "VIET" in matched or "VIỆT" in matched:
            cleaned = "VIET NAM"

    elif field_type == "license_class":
        cleaned = cleaned.upper().replace(" ", "")
        cleaned = fuzzy_correct(cleaned, STANDARD_LICENSE_CLASSES, score_cutoff=70.0)

    elif field_type in ("full_name", "owner_name"):
        cleaned = cleaned.upper()
        cleaned = re.sub(r"[\d\._\-\+\*\?\!]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

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
        return text in ("NAM", "NU")

    if field_type == "license_class":
        return text in STANDARD_LICENSE_CLASSES

    return len(text) > 0
