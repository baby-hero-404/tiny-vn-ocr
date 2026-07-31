"""Method using baseline VietOCR but with an enhanced heuristic parser."""

import re
import numpy as np
from typing import Dict, List
from rapidfuzz import process, fuzz

from benchmarks.registry import register_method
from src.ocr.engine import OCREngine
from src.postprocessing.parser import _strip_accents, detect_cccd_side, KNOWN_ISSUERS

_ENGINE = OCREngine()

def extract_date(text: str) -> str:
    """Extract standard date DD/MM/YYYY from text."""
    match = re.search(r'\d{2}[/.\-]\d{2}[/.\-]\d{4}', text)
    if match:
        return match.group(0).replace(".", "/").replace("-", "/")
    return ""

def enhanced_parse_cccd_front(lines: List[str]) -> Dict[str, str]:
    fields = {}
    
    # Clean lines
    clean_lines = [l.strip() for l in lines if l.strip()]
    full_text = " ".join(clean_lines)
    
    # 1. ID Number (find any 12 continuous digits, ignoring spaces)
    id_text = full_text.replace(" ", "")
    id_match = re.search(r'\b\d{12}\b', id_text)
    if id_match:
        fields["id_number"] = id_match.group(0)
        
    # 2. Extract ALL dates
    dates = []
    def normalize_mangled_date(d_str: str) -> str:
        d_clean = re.sub(r'[^\d]', '', d_str)
        if len(d_clean) >= 8:
            return f"{d_clean[:2]}/{d_clean[2:4]}/{d_clean[-4:]}"
        return d_str.replace(".", "/").replace("-", "/")

    for line in clean_lines:
        for match in re.finditer(r'\d{2}[/.\-]\d{2}[/.\-]\d{4}', line):
            d = match.group(0).replace(".", "/").replace("-", "/")
            if d not in dates:
                dates.append(d)
        for match in re.finditer(r'\b\d{4,5}[/.\-]\d{4}\b|\b\d{2}[/.\-]\d{6,7}\b', line):
            norm = normalize_mangled_date(match.group(0))
            if norm not in dates:
                dates.append(norm)
                
    if len(dates) >= 2:
        if "date_of_birth" not in fields:
            fields["date_of_birth"] = dates[0]
        fields["date_of_expiry"] = dates[-1]
    elif len(dates) == 1:
        if "date_of_birth" not in fields:
            fields["date_of_birth"] = dates[0]
        elif "date_of_expiry" not in fields:
            fields["date_of_expiry"] = dates[0]

    # 3. Gender
    combined_lower = _strip_accents(full_text.lower())
    if "nam" in combined_lower and not "nu" in combined_lower:
        fields["gender"] = "NAM"
    elif "nu" in combined_lower:
        fields["gender"] = "NỮ"
        
    # 4. Nationality
    if "viet nam" in combined_lower:
        fields["nationality"] = "Việt Nam"

    # 5. Find line indices using fuzzy matching for labels
    name_idx, dob_idx, origin_idx, residence_idx = -1, -1, -1, -1
    
    for i, line in enumerate(clean_lines):
        norm_line = _strip_accents(line.lower())
        
        if name_idx == -1 and ("ho va ten" in norm_line or "full name" in norm_line):
            name_idx = i
        if dob_idx == -1 and ("ngay sinh" in norm_line or "date of birth" in norm_line):
            dob_idx = i
        if origin_idx == -1 and ("que quan" in norm_line or "place of origin" in norm_line):
            origin_idx = i
        if residence_idx == -1 and ("thuong tru" in norm_line or "place of residence" in norm_line):
            residence_idx = i

    # Fallback to simple "ten" "sinh" if fuzzy fails
    if name_idx == -1:
        for i, line in enumerate(clean_lines):
            norm = _strip_accents(line.lower())
            if "ten:" in norm or "name:" in norm or "tên:" in line.lower():
                name_idx = i; break

    # Extract Full Name
    if name_idx != -1:
        line = clean_lines[name_idx]
        if ":" in line:
            name = line.split(":", 1)[1].strip()
            if len(name) > 3:
                fields["full_name"] = name
        else:
            name_parts = []
            max_idx = dob_idx if dob_idx != -1 else min(name_idx + 3, len(clean_lines))
            for i in range(name_idx + 1, max_idx):
                part = clean_lines[i]
                if not re.search(r'\d', part) and len(part) > 2:
                    name_parts.append(part)
            if name_parts:
                fields["full_name"] = " ".join(name_parts)
                
    # Extract Origin & Residence
    def extract_address_block(start_idx: int, end_idx: int, labels_to_remove: List[str]) -> str:
        if start_idx == -1:
            return ""
        
        end = end_idx if end_idx != -1 else len(clean_lines)
        # Prevent runaway if end_idx is wrong
        if end <= start_idx: 
            end = len(clean_lines)
            
        block = clean_lines[start_idx:end]
        
        # Clean first line of the block
        first_line = block[0]
        if ":" in first_line:
            val = first_line.split(":", 1)[1].strip()
            block[0] = val if len(val) > 2 else ""
        else:
            # Remove labels via regex
            regex = r'(?i)(' + "|".join(labels_to_remove) + r')[^\w]*'
            clean_first = re.sub(regex, '', first_line).strip()
            block[0] = clean_first
            
        # Filter out obvious garbage from the block
        valid_lines = []
        for l in block:
            if not l: continue
            norm = _strip_accents(l.lower())
            if "co gia tri den" in norm or "date of expiry" in norm:
                break
            if re.search(r'(?:co|co).*?(?:den|den)', norm):
                break
            if re.search(r'\d{2}/\d{2}/\d{4}|\b\d{4,5}[/.\-]\d{4}\b|\b\d{2}[/.\-]\d{6,7}\b', l):
                continue
            if re.search(r'\b\d{10,}\b', l):
                continue
            valid_lines.append(l)
            
        return " ".join(valid_lines).replace(" ,", ",").strip()

    fields["place_of_origin"] = extract_address_block(origin_idx, residence_idx, ["quê quán", "que quan", "place of origin"])
    fields["place_of_residence"] = extract_address_block(residence_idx, len(clean_lines), ["nơi thường trú", "thuong tru", "place of residence", "noi th"])

    return fields

def enhanced_parse_cccd_back(lines: List[str]) -> Dict[str, str]:
    fields = {}
    clean_lines = [l.strip() for l in lines if l.strip()]
    
    # Extract ALL dates
    dates = []
    for line in clean_lines:
        for match in re.finditer(r'\d{2}[/.\-]\d{2}[/.\-]\d{4}', line):
            d = match.group(0).replace(".", "/").replace("-", "/")
            if d not in dates:
                dates.append(d)
                
    # Also extract date in format "Ngày 23 tháng 06 năm 2023"
    for line in clean_lines:
        norm = _strip_accents(line.lower())
        if "ngay" in norm and "thang" in norm and "nam" in norm:
            nums = re.findall(r'\d+', line)
            if len(nums) >= 3:
                d = f"{nums[-3].zfill(2)}/{nums[-2].zfill(2)}/{nums[-1]}"
                if d not in dates:
                    dates.append(d)
    
    if len(dates) == 1:
        fields["date_of_issue"] = dates[0]
    elif len(dates) >= 2:
        fields["date_of_issue"] = dates[0]
        fields["date_of_expiry"] = dates[-1]

    # Find indices
    residence_idx, birth_idx = -1, -1
    for i, line in enumerate(clean_lines):
        norm = _strip_accents(line.lower())
        if residence_idx == -1 and "cu tru" in norm:
            residence_idx = i
        if birth_idx == -1 and "khai sinh" in norm:
            birth_idx = i

    def extract_address_block(start_idx: int, end_idx: int, labels: List[str]) -> str:
        if start_idx == -1: return ""
        end = end_idx if (end_idx != -1 and end_idx > start_idx) else len(clean_lines)
        
        block = clean_lines[start_idx:end]
        first = block[0]
        if ":" in first:
            val = first.split(":", 1)[1].strip()
            block[0] = val if len(val) > 2 else ""
        elif "/" in first:
            val = first.split("/", 1)[1].strip()
            block[0] = val if len(val) > 2 else ""
        else:
            regex = r'(?i)(' + "|".join(labels) + r')[^\w]*'
            clean_first = re.sub(regex, '', first).strip()
            if clean_first == first.strip() and len(clean_first) > 15:
                parts = clean_first.split()
                if len(parts) > 5:
                    clean_first = " ".join(parts[4:])
            block[0] = clean_first
            
        valid_lines = []
        for l in block:
            if not l: continue
            if re.search(r'\d{2}/\d{2}/\d{4}', l): break
            if "bo cong an" in _strip_accents(l.lower()): break
            valid_lines.append(l)
            
        return " ".join(valid_lines).replace(" ,", ",").strip()

    if residence_idx != -1:
        fields["place_of_residence"] = extract_address_block(residence_idx, birth_idx, ["nơi cư trú", "noi cu tru", "place of residence"])
    if birth_idx != -1:
        fields["place_of_birth"] = extract_address_block(birth_idx, len(clean_lines), ["nơi đăng ký khai sinh", "noi dang ky khai sinh", "place of birth"])

    # Extract issuing authority
    for i, line in enumerate(clean_lines):
        norm = _strip_accents(line.lower())
        if "giam doc" in norm or "cuc truong" in norm or "canh sat" in norm or "bo cong an" in norm:
            place = line
            if i + 1 < len(clean_lines):
                next_line = clean_lines[i+1]
                if not re.search(r'\d', next_line) and "director" not in next_line.lower():
                    place += " " + next_line
                    
            place = re.sub(r'(?i)(CỤC TRƯỞNG|GIÁM ĐỐC|CỤC TRƯỜNG)', '', place).strip()
            
            # Fuzzy correct
            match_result = process.extractOne(place, KNOWN_ISSUERS, scorer=fuzz.WRatio, score_cutoff=55.0)
            if match_result:
                place = match_result[0]
                
            fields["place_of_issue"] = place
            break

    return fields

def enhanced_parse(lines: List[str], document_type: str) -> Dict[str, str]:
    if document_type == "cccd_auto":
        document_type = detect_cccd_side(lines)
        
    if document_type == "cccd":
        return enhanced_parse_cccd_front(lines)
    elif document_type == "cccd_back":
        return enhanced_parse_cccd_back(lines)
    return {}

@register_method("enhanced_parser_vietocr")
def method_enhanced_parser_vietocr(image: np.ndarray, doc_type: str) -> Dict[str, str]:
    """Run VietOCR and parse using the improved heuristic parser."""
    lines = _ENGINE.recognize_lines_vietocr(image)
    fields = enhanced_parse(lines, doc_type)
    return fields
