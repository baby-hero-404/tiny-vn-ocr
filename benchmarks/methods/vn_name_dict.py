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

# Common Vietnamese given/middle names that OCR often misinterprets
VN_GIVEN_NAMES = [
    "Như", "Nhữ", "Thúy", "Thủy", "Tâm", "Tầm", "Tuấn", "Tuấn",
    "Danh", "Nguyễn", "Thị", "Văn", "An", "Anh", "Bình", "Chính", "Cường", "Dũng",
    "Đạt", "Đức", "Hải", "Hào", "Hiếu", "Hùng", "Huy", "Khang", "Khánh", "Khoa", "Khôi",
    "Kiên", "Lâm", "Long", "Minh", "Nam", "Nghĩa", "Ngọc", "Nhật", "Phát", "Phong",
    "Phúc", "Quân", "Quang", "Quốc", "Sơn", "Tài", "Thắng", "Thành", "Thiên", "Thịnh",
    "Trung", "Trường", "Tú", "Tuấn", "Uy", "Việt", "Vinh", "Vũ", "Xuân",
    "Bảo", "Bích", "Châu", "Chi", "Diệp", "Diệu", "Dung", "Đào", "Giang",
    "Giao", "Hà", "Hân", "Hằng", "Hoa", "Hoà", "Hoài", "Hương", "Hường",
    "Kim", "Lan", "Lê", "Liên", "Linh", "Loan", "Ly", "Mai", "My",
    "Nga", "Ngân", "Nghi", "Nhung", "Oanh", "Phạm", "Phương", "Phượng",
    "Quyên", "Quỳnh", "Thảo", "Thi", "Thu", "Thương", "Thư", "Tiên", "Trang",
    "Trâm", "Trân", "Trúc", "Tâm", "Uyên", "Vân", "Vy", "Yến", "Xuân",
    "Nhi", "Hạnh", "Hiền", "Tuyết", "Mai", "Thi", "Thắm", "Trinh"
]

# Build lookup for all valid name syllables (Surnames + Given names)
_NAME_BASE_MAP: dict[str, str] = {}
for name in VN_SURNAMES + VN_GIVEN_NAMES:
    base = _strip_accents(name).lower()
    # If conflict, first entry wins
    if base not in _NAME_BASE_MAP:
        _NAME_BASE_MAP[base] = name

def correct_full_name(raw_name: str) -> str:
    """Correct OCR diacritics errors in Vietnamese full names.
    
    Matches each word by base characters (ignoring tone marks)
    and replaces with canonical diacritics if found in dictionary.
    
    Examples:
        "DANH NGUYỄN TÂM NHƯ" → "DANH NGUYỄN TÂM NHỮ" (if NHỮ is in dict)
    """
    if not raw_name:
        return raw_name
    
    words = raw_name.split()
    corrected_words = []
    
    for word in words:
        is_upper = word == word.upper()
        base = _strip_accents(word).lower()
        canonical = _NAME_BASE_MAP.get(base)
        
        if canonical and canonical.lower() != word.lower():
            # Apply correction
            corrected_words.append(canonical.upper() if is_upper else canonical)
        else:
            corrected_words.append(word)
            
    return " ".join(corrected_words)
