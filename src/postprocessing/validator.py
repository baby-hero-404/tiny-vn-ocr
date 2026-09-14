import re
import unicodedata
from typing import Dict, Optional
from rapidfuzz import process, fuzz

# Exhaustive list of 63 provinces in Vietnam
VN_PROVINCES = [
    "An Giang", "Bà Rịa - Vũng Tàu", "Bắc Giang", "Bắc Kạn", "Bạc Liêu", "Bắc Ninh", "Bến Tre",
    "Bình Định", "Bình Dương", "Bình Phước", "Bình Thuận", "Cà Mau", "Cao Bằng", "Đắk Lắk",
    "Đắk Nông", "Điện Biên", "Đồng Nai", "Đồng Tháp", "Gia Lai", "Hà Giang", "Hà Nam", "Hà Tĩnh",
    "Hải Dương", "Hậu Giang", "Hòa Bình", "Hưng Yên", "Khánh Hòa", "Kiên Giang", "Kon Tum",
    "Lai Châu", "Lâm Đồng", "Lạng Sơn", "Lào Cai", "Long An", "Nam Định", "Nghệ An", "Ninh Bình",
    "Ninh Thuận", "Phú Thọ", "Quảng Bình", "Quảng Nam", "Quảng Ngãi", "Quảng Ninh", "Quảng Trị",
    "Sóc Trăng", "Sơn La", "Tây Ninh", "Thái Bình", "Thái Nguyên", "Thanh Hóa", "Thừa Thiên Huế",
    "Tiền Giang", "Trà Vinh", "Tuyên Quang", "Vĩnh Long", "Vĩnh Phúc", "Yên Bái", "Phú Yên",
    "Cần Thơ", "Đà Nẵng", "Hải Phòng", "Hà Nội", "TP Hồ Chí Minh", "TP. Hồ Chí Minh"
]


def _strip_accents(text: str) -> str:
    """Remove Vietnamese diacritics, e.g. for matching OCR text that lost its dấu."""
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# Map "ha noi" -> "Hà Nội" so accent-stripped OCR text still matches correctly.
_VN_PROVINCES_NOACCENT = {_strip_accents(p).lower(): p for p in VN_PROVINCES}


def _match_province(text: str, score_cutoff: float = 70.0) -> Optional[str]:
    """Fuzzy-match text against the province list, accent-insensitive first."""
    if not text:
        return None

    stripped = _strip_accents(text).lower()
    match = process.extractOne(
        stripped, _VN_PROVINCES_NOACCENT.keys(), scorer=fuzz.WRatio, score_cutoff=score_cutoff
    )
    if match:
        return _VN_PROVINCES_NOACCENT[match[0]]

    # Fall back to accented matching in case stripping hurt the score
    # (e.g. text already carries correct diacritics).
    match = process.extractOne(text, VN_PROVINCES, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
    return match[0] if match else None

class FieldValidator:
    """Validates and normalizes fields parsed from CCCD."""
    
    @staticmethod
    def validate_id(id_str: str) -> str:
        # Strip non-digits
        clean = re.sub(r'\D', '', id_str)
        if len(clean) == 12:
            return clean
        return id_str
        
    @staticmethod
    def is_valid_cccd_id(id_str: str) -> bool:
        """Check if an ID string matches the CCCD format (12 digits, valid province and gender code)."""
        clean = re.sub(r'\D', '', id_str)
        if len(clean) != 12:
            return False
            
        province_code = int(clean[0:3])
        if not (1 <= province_code <= 96):
            return False
            
        gender_code = int(clean[3])
        # Currently we only expect people born in 1900s (0/1) or 2000s (2/3). 
        if not (0 <= gender_code <= 3):
            return False
            
        return True
        
    @staticmethod
    def validate_date(date_str: str) -> str:
        # Rollback: Validation caused date format regression, returning raw parsed string instead
        return date_str
        
    @staticmethod
    def validate_gender(gender_str: str) -> str:
        raw_g = gender_str.upper()
        if "NAM" in raw_g or "NÀM" in raw_g:
            return "Nam"
        elif "NỮ" in raw_g or "NU" in raw_g or "NO" in raw_g:
            return "Nữ"
        return gender_str
        
    @staticmethod
    def validate_nationality(nat_str: str) -> str:
        raw_n = nat_str.upper()
        if fuzz.partial_ratio("VIET NAM", raw_n) > 70 or ("VI" in raw_n and "N" in raw_n):
            return "Việt Nam"
        return nat_str

    @staticmethod
    def normalize_address(raw_address: str) -> str:
        """Correct spelling of the province (last comma segment) via fuzzy match
        against the 63-province list. District/ward detail is left as raw OCR
        text on purpose — no hardcoded gazetteer, since minor errors there are
        acceptable for this project."""
        if not raw_address:
            return ""

        tmp_address = raw_address.replace("TP. HCM", "TP Hồ Chí Minh")
        parts = [p.strip() for p in tmp_address.split(",") if p.strip()]

        if not parts:
            return raw_address

        province_raw = parts[-1]
        matched = _match_province(province_raw)

        if matched:
            parts[-1] = matched

        return ", ".join([p for p in parts if p])

    @staticmethod
    def _has_recognized_province(address: str) -> bool:
        """True if the address's last comma segment fuzzy-matches a known province."""
        if not address:
            return False
        last_part = address.split(",")[-1].strip()
        return _match_province(last_part) is not None

    @classmethod
    def _cross_fill_addresses(cls, validated: Dict[str, str]) -> Dict[str, str]:
        """place_of_origin and place_of_residence are very often the same on a
        Vietnamese CCCD. If OCR/parsing clearly failed on one (no recognizable
        province) but the other looks solid, borrow the good one instead of
        leaving unusable text — a same-field fallback beats a hardcoded
        address database, and minor detail mismatches are acceptable here."""
        origin = validated.get("place_of_origin", "")
        residence = validated.get("place_of_residence", "")
        if not origin or not residence:
            return validated

        origin_ok = cls._has_recognized_province(origin)
        residence_ok = cls._has_recognized_province(residence)

        if origin_ok == residence_ok:
            return validated  # both fine, or both unrecoverable: don't guess

        if origin_ok:
            validated["place_of_residence"] = origin
        else:
            validated["place_of_origin"] = residence

        return validated

    @classmethod
    def validate_all(cls, fields: Dict[str, str]) -> Dict[str, str]:
        """Apply validation and normalization to all fields in the dict."""
        validated = fields.copy()
        
        if "id_number" in validated:
            validated["id_number"] = cls.validate_id(validated["id_number"])
            
        for date_field in ["date_of_birth", "date_of_expiry", "date_of_issue"]:
            if date_field in validated:
                validated[date_field] = cls.validate_date(validated[date_field])
                
        if "gender" in validated:
            validated["gender"] = cls.validate_gender(validated["gender"])
            
        if "nationality" in validated:
            validated["nationality"] = cls.validate_nationality(validated["nationality"])
            
        for addr_field in ["place_of_origin", "place_of_residence", "place_of_issue", "place_of_birth"]:
            if addr_field in validated:
                validated[addr_field] = cls.normalize_address(validated[addr_field])

        validated = cls._cross_fill_addresses(validated)

        return validated
