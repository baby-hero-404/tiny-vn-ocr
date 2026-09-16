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
    "Nga", "Ngân", "Nghi", "Nhung", "Oanh", "Phạm", "Phương", "Phượng", "Phụng",
    "Quyên", "Quỳnh", "Thảo", "Thi", "Thu", "Thương", "Thư", "Tiên", "Trang",
    "Trâm", "Trân", "Trúc", "Tâm", "Uyên", "Vân", "Vy", "Yến", "Xuân",
    "Nhi", "Hạnh", "Hiền", "Tuyết", "Mai", "Thi", "Thắm", "Trinh", "Trình",
    "Hoài", "Linh", "Bùi", "Minh", "Duy", "Hòa", "Tú", "Đức", "Trí", "Hoàng",
    "Việt", "Quang", "Phương", "Thành", "Kiên", "Khánh", "Cường", "Giang",
    "Hương", "Hà", "Hải", "Tuấn", "Tiến", "Tùng", "Khoa", "Phong", "Thắng",
    "Thịnh", "Hữu", "Vĩnh", "Trọng", "Phước", "Công", "Chí", "Thế", "Khải",
    "Khôi", "Nhật", "Huyền", "Thanh", "Nguyên", "Văn", "Thị", "Ngọc", "Mỹ",
    "Đạt", "Lộc", "Phát", "Tài", "Phú", "Quý", "Hưng", "Thuận", "Hiếu", "Toàn",
    "Khang", "Ninh", "Châu", "Bảo", "Đăng", "Điền", "Lợi", "Chiến"
]

# Build lookup for all valid name syllables (Surnames + Given names)
_NAME_BASE_MAP: dict[str, str] = {}
for name in VN_SURNAMES + VN_GIVEN_NAMES:
    base = _strip_accents(name).lower()
    # If conflict, first entry wins
    if base not in _NAME_BASE_MAP:
        _NAME_BASE_MAP[base] = name

# Multiple accented names can share the same unaccented base (e.g. "ha" ->
# "Hà" or "Hạ"), and _NAME_BASE_MAP only keeps one. So a word that is
# ALREADY a recognized name — just not the map's chosen variant — must not
# be overwritten; only look up base-map corrections for words that aren't
# themselves already a known, correctly-accented name.
_VALID_NAMES_LOWER = {n.lower() for n in VN_SURNAMES + VN_GIVEN_NAMES}

import re

def split_joined_name(raw_name: str) -> str:
    """Split concatenated names like 'BUIHOAILINH' or 'BuiHoaiLinh' into 'BUI HOAI LINH'."""
    if not raw_name:
        return raw_name
    cleaned = raw_name.strip()
    
    # 1. Split CamelCase (e.g. BuiHoaiLinh -> Bui Hoai Linh)
    cleaned = re.sub(r'([a-zđ])([A-ZĐ])', r'\1 \2', cleaned)
    
    if " " in cleaned:
        parts = cleaned.split()
        res = []
        for p in parts:
            if len(p) >= 7 and _strip_accents(p).isalpha():
                res.append(split_joined_name(p))
            else:
                res.append(p)
        return " ".join(res)
        
    # 2. All-caps / all-lowercase joined string (e.g. BUIHOAILINH)
    norm = _strip_accents(cleaned).lower()
    n = len(norm)
    if n < 4:
        return cleaned
        
    all_syllables = set(_NAME_BASE_MAP.keys())
    dp = [None] * (n + 1)
    dp[0] = []
    
    for i in range(n):
        if dp[i] is None:
            continue
        for l in range(2, min(8, n - i + 1)):
            sub = norm[i:i+l]
            if sub in all_syllables:
                if dp[i + l] is None or len(dp[i]) + 1 < len(dp[i + l]):
                    dp[i + l] = dp[i] + [cleaned[i:i+l]]
                    
    if dp[n] and len(dp[n]) >= 2:
        return " ".join(dp[n])
        
    return cleaned

def correct_full_name(raw_name: str) -> str:
    """Correct OCR diacritics errors and split concatenated words in Vietnamese full names.
    
    Examples:
        "BUIHOAILINH" → "BÙI HOÀI LINH"
        "DANH TRI HOÀNG" → "DANH TRÍ HOÀNG"
    """
    if not raw_name:
        return raw_name
    
    raw_name = split_joined_name(raw_name)
    words = raw_name.split()
    corrected_words = []
    
    for word in words:
        is_upper = word == word.upper()
        base = _strip_accents(word).lower()

        # If the word (with whatever accents it has) is already a recognized
        # name, trust it as-is — the base map only holds one canonical
        # variant per base and would otherwise clobber an equally valid but
        # differently-accented name (e.g. "Hạ" -> wrongly forced to "Hà").
        # Only fall back to base-map lookup to fix OCR diacritic misreads
        # for words that aren't themselves already a known valid name
        # (e.g. unaccented "TRI" -> "TRÍ", or a genuinely wrong accent).
        canonical = None if word.lower() in _VALID_NAMES_LOWER else _NAME_BASE_MAP.get(base)
        if canonical:
            corrected_words.append(canonical.upper() if is_upper else canonical)
        else:
            corrected_words.append(word)
            
    return " ".join(corrected_words)
