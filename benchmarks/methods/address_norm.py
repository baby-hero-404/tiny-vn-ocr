"""Method using Enhanced Parser + Address Normalization (Vietnamese Post-correction)."""

import numpy as np
from typing import Dict, List
from rapidfuzz import process, fuzz

from benchmarks.registry import register_method
from src.ocr.engine import OCREngine
from benchmarks.methods.enhanced_parser import enhanced_parse

_ENGINE = OCREngine()

# Mock dictionary of provinces/cities (could be expanded to districts/wards)
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

def normalize_address(raw_address: str) -> str:
    """Correct spelling for the province part of the address."""
    if not raw_address:
        return ""
        
    parts = [p.strip() for p in raw_address.split(",")]
    if not parts:
        return raw_address
        
    # The last part is usually the province
    province_raw = parts[-1]
    
    # Use fuzzy matching to find the closest province
    match = process.extractOne(province_raw, VN_PROVINCES, scorer=fuzz.WRatio, score_cutoff=70.0)
    
    if match:
        parts[-1] = match[0]
        
    return ", ".join(parts)

def normalize_gender_nationality(fields: Dict[str, str]) -> Dict[str, str]:
    if "gender" in fields:
        raw_g = fields["gender"].upper()
        if "NAM" in raw_g or "NÀM" in raw_g:
            fields["gender"] = "Nam"
        elif "NỮ" in raw_g or "NU" in raw_g or "NO" in raw_g:
            fields["gender"] = "Nữ"
            
    if "nationality" in fields:
        raw_n = fields["nationality"].upper()
        if fuzz.partial_ratio("VIET NAM", raw_n) > 70 or ("VI" in raw_n and "N" in raw_n):
            fields["nationality"] = "Việt Nam"
            
    return fields

@register_method("address_norm_vietocr")
def method_address_norm_vietocr(image: np.ndarray, doc_type: str) -> Dict[str, str]:
    """Extract text using VietOCR, parse it, and then normalize the address fields."""
    lines = _ENGINE.recognize_lines_vietocr(image)
    fields = enhanced_parse(lines, doc_type)
    
    if "place_of_origin" in fields:
        fields["place_of_origin"] = normalize_address(fields["place_of_origin"])
        
    if "place_of_residence" in fields:
        fields["place_of_residence"] = normalize_address(fields["place_of_residence"])
        
    fields = normalize_gender_nationality(fields)
            
    return fields
