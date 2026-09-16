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
    assert normalize_field("NAM", "gender") in ("Nam", "NAM")
    assert normalize_field("NƯ", "gender") in ("Nữ", "NU")
    assert normalize_field("NAAM", "gender") in ("Nam", "NAM")


def test_fuzzy_correct():
    candidates = ["VIET NAM", "CAMPUCHIA", "LAO"]
    assert fuzzy_correct("VIETNAM", candidates) == "VIET NAM"


def test_full_name_normalization():
    raw_name = "NGUYEN123 VAN A!!"
    normalized = normalize_field(raw_name, "full_name")
    assert normalized in ("NGUYEN VAN A", "NGUYỄN VĂN A")


def test_address_colon_and_label_cleaning():
    from src.postprocessing.address_norm import clean_address_string, deduplicate_address_segments, normalize_address

    # 1. Colon and bilingual label purge
    raw1 = "Queguan I Placeoforigin: Thạnh Phú, Mỹ Xuyên, Sóc Trăng"
    assert clean_address_string(raw1) == "Thạnh Phú, Mỹ Xuyên, Sóc Trăng"
    assert ":" not in clean_address_string(raw1)

    raw2 = "Nơi thường trú / Place of residence: Ấp Khu 3, Thạnh Phú, Mỹ Xuyên, Sóc Trăng"
    assert clean_address_string(raw2) == "Ấp Khu 3, Thạnh Phú, Mỹ Xuyên, Sóc Trăng"
    assert ":" not in clean_address_string(raw2)

    # 2. Stray colons replaced cleanly without losing address info
    raw3 = "Thạnh Phú, Mỹ Xuyên: Sóc Trăng"
    assert clean_address_string(raw3) == "Thạnh Phú, Mỹ Xuyên, Sóc Trăng"
    assert ":" not in clean_address_string(raw3)

    # 3. Deduplication of repeated administrative blocks
    dup = "Thạnh Phú, Mỹ Xuyên, Sóc Trăng, Thạnh Phú, Mỹ Xuyên, Sóc Trăng"
    assert deduplicate_address_segments(dup) == "Thạnh Phú, Mỹ Xuyên, Sóc Trăng"

    # 4. End-to-end normalize_address guarantees no colons
    res = normalize_address("Quê quán: Thạnh Phú, Mỹ Xuyên, Sóc Trăng, Thạnh Phú, Mỹ Xuyên, Sóc Trăng")
    assert ":" not in res
    assert res == "Thạnh Phú, Mỹ Xuyên, Sóc Trăng"


def test_normalize_quan_district():
    assert normalize_field("Quan 1, TP Ho Chi Minh", "address") == "Quận 1, TP Ho Chi Minh"
    assert normalize_field("Quan Ba Dinh, Ha Noi", "address") == "Quận Ba Đình, Ha Noi"
    assert normalize_field("Quan Đình Nam, Tân Tiến, Hưng Yên", "address") == "Quan Đình Nam, Tân Tiến, Hưng Yên"
    assert normalize_field("Quan Dinh Nam", "address") == "Quan Dinh Nam"

