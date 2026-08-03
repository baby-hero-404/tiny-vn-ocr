import re
from typing import Dict
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
    def validate_date(date_str: str) -> str:
        # Extract dd/mm/yyyy format
        m = re.search(r'(\d{2})[/.\- ]?(\d{2})[/.\- ]?(\d{4})', date_str)
        if m:
            return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
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
        """Correct spelling for the province part of the address and add missing commas if possible."""
        if not raw_address:
            return ""
            
        # Standardize aliases
        tmp_address = raw_address.replace("TP. HCM", "TP Hồ Chí Minh")
        
        parts = [p.strip() for p in tmp_address.split(",") if p.strip()]
        
        # Try to fix missing commas (e.g. "Ấp Đại Ân Đại Tâm Mỹ Xuyên Sóc Trăng")
        # If there are no commas, we find the province and split it out
        if len(parts) == 1:
            # Find closest matching province in the string
            best_match = process.extractOne(parts[0], VN_PROVINCES, scorer=fuzz.partial_ratio, score_cutoff=80.0)
            if best_match:
                province = best_match[0]
                # Find where it starts in the original string using case-insensitive search
                idx = parts[0].lower().rfind(province.lower())
                if idx != -1:
                    rest = parts[0][:idx].strip()
                    parts = [rest, province]
                    
        if not parts:
            return raw_address
            
        # The last part is usually the province
        province_raw = parts[-1]
        
        # Use fuzzy matching to find the closest province
        match = process.extractOne(province_raw, VN_PROVINCES, scorer=fuzz.WRatio, score_cutoff=70.0)
        
        if match:
            parts[-1] = match[0]
            
        return ", ".join([p for p in parts if p])

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
                
        return validated
