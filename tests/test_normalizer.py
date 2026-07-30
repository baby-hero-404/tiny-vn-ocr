"""Unit tests for text post-processing and normalization module."""

from src.postprocessing.normalizer import (
    normalize_field,
    validate_field,
    fuzzy_correct,
    normalize_digits,
    normalize_date,
)


def test_character_substitution_numeric():
    # 'O' -> '0', 'S' -> '5', 'I' -> '1'
    raw_text = "O79123S6789I"
    normalized = normalize_field(raw_text, "id_number")
    assert normalized == "079123567891"
    assert validate_field(normalized, "id_number") is True


def test_normalize_date():
    assert normalize_field("01.01.1995", "date_of_birth") == "01/01/1995"
    assert normalize_field("01-01-1995", "date_of_birth") == "01/01/1995"
    assert validate_field("01/01/1995", "date_of_birth") is True


def test_gender_normalization_and_fuzzy():
    assert normalize_field("NAM", "gender") == "NAM"
    assert normalize_field("NƯ", "gender") == "NU"
    assert normalize_field("NAAM", "gender") == "NAM"


def test_fuzzy_correct():
    candidates = ["VIET NAM", "CAMPUCHIA", "LAO"]
    assert fuzzy_correct("VIETNAM", candidates) == "VIET NAM"


def test_full_name_normalization():
    raw_name = "NGUYEN123 VAN A!!"
    normalized = normalize_field(raw_name, "full_name")
    assert normalized == "NGUYEN VAN A"
