"""Shared helpers for decoding info encoded in a 12-digit CCCD ID number.

Digit layout: [province:3][gender+century:1][birth_year:2][sequence:6]
The gender/century digit encodes both the holder's gender and which century
they were born in:
  0/1 -> 1900s male/female, 2/3 -> 2000s male/female, 4/5 -> 2100s male/female
"""

from typing import Optional, Tuple

_CENTURY_BASE_BY_GENDER_CODE = {
    0: 1900, 1: 1900,
    2: 2000, 3: 2000,
    4: 2100, 5: 2100,
}


def infer_birth_year_and_gender(id_number: str) -> Optional[Tuple[int, str]]:
    """Decode (birth_year, gender) from a 12-digit CCCD ID number.

    Returns None if id_number is not a valid 12-digit numeric ID, or if the
    gender/century digit (position 4) is outside the defined 0-5 range —
    such an ID is malformed and should not be trusted for inference.
    gender is "Nam" or "Nữ" (the canonical form used across the codebase).
    """
    if len(id_number) != 12 or not id_number.isdigit():
        return None
    gender_code = int(id_number[3])
    if gender_code not in _CENTURY_BASE_BY_GENDER_CODE:
        return None
    century_base = _CENTURY_BASE_BY_GENDER_CODE[gender_code]
    birth_year = century_base + int(id_number[4:6])
    gender = "Nam" if gender_code % 2 == 0 else "Nữ"
    return birth_year, gender
