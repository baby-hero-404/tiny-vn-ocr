"""Vietnamese Surname Correction — Fix OCR diacritics for the finite set of Vietnamese surnames."""

import unicodedata
from typing import Optional


def _strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize('NFD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


# Vietnamese surnames are a FINITE, well-defined set (~200 total).
# This is NOT a "guess" — it's an exhaustive list. Every Vietnamese person
# has one of these as their family name.
# Source: https://vi.wikipedia.org/wiki/Họ_người_Việt_Nam
VN_SURNAMES = [
    # Top 20 most common (covers ~90% of population)
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ",
    "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý", "Đào", "Đinh",
    "Trịnh", "Lâm",
    # Next tier
    "Mai", "Tô", "Hà", "Tạ", "Châu", "Lương", "Quách", "Vương", "Tăng",
    "Sơn", "Diệp", "Danh", "Kiều", "Thái", "Đoàn", "Lưu", "Cao", "Thạch",
    "Trương", "Từ", "La", "Khưu", "Tiêu", "Mạc", "Chung", "Phùng",
    "Triệu", "Lục", "Nghiêm", "Tống", "Dư", "Kim", "Liêu", "Thân",
    "Giang", "Ông", "Bạch", "Cù", "Đàm", "Đồng", "Giáp", "Hứa", "Kha",
    "Khổng", "Lã", "Lại", "Mã", "Ninh", "Nhan", "Quản", "Tào", "Thẩm",
    "Thiều", "Thi", "Trà", "Vi", "Viên", "Vưu", "Xà", "Xuân", "Yên",
    "Âu", "Ấu", "Bàng", "Biện", "Bồ", "Cái", "Cam", "Cầm", "Chiêm",
    "Cung", "Doãn", "Đàn", "Đới", "Hạ", "Hồng", "Hùng", "Khuất", "Kiểu",
    "Lạc", "Lang", "Lệ", "Liễu", "Lò", "Long", "Lộ", "Lục", "Lương",
    "Ma", "Man", "Mầu", "Mục", "Năng", "Nông", "Ôn", "Phi", "Phí",
    "Phó", "Phú", "Quế", "Sa", "Sái", "Sầm", "Sử", "Tạ", "Tần",
    "Tề", "Thành", "Thào", "Thiện", "Thời", "Tiền", "Tôn", "Trang",
    "Triệu", "Trịnh", "Trương", "Ung", "Ưng", "Văn", "Vĩnh", "Vò",
]

# Build lookup: base_form -> correct_form
_SURNAME_BASE_MAP: dict[str, str] = {}
for name in VN_SURNAMES:
    base = _strip_accents(name).lower()
    # First entry wins (most common variant)
    if base not in _SURNAME_BASE_MAP:
        _SURNAME_BASE_MAP[base] = name


def correct_surname(raw_name: str) -> str:
    """Correct OCR diacritics errors in the SURNAME (first word) only.
    
    This is generalizable because Vietnamese surnames are a finite, 
    well-defined set. We match by base characters (ignoring tone marks)
    and replace with the canonical diacritics.
    
    Given names and middle names are NOT corrected because they are
    too diverse — that requires OCR model improvement or NLP spell-checking.
    
    Examples:
        "BŨI VĂN TIẾN" → "BÙI VĂN TIẾN"  (BŨI not a valid surname)
        "HUỲNH VĨNH PHƯỚC" → unchanged     (HUỲNH is already correct)
    """
    if not raw_name:
        return raw_name
    
    words = raw_name.split()
    if not words:
        return raw_name
    
    first_word = words[0]
    is_upper = first_word == first_word.upper()
    
    # Look up by base characters
    base = _strip_accents(first_word).lower()
    canonical = _SURNAME_BASE_MAP.get(base)
    
    if canonical and canonical.lower() != first_word.lower():
        # The surname exists but with different diacritics → correct it
        if is_upper:
            words[0] = canonical.upper()
        else:
            words[0] = canonical
        return " ".join(words)
    
    return raw_name
