"""Method using Enhanced Parser + Address Normalization (Vietnamese Post-correction)."""

import numpy as np
from typing import Dict, List
from rapidfuzz import process, fuzz
import json
import os
import re
import unicodedata

def _strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize('NFD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _unaccented_scorer(s1: str, s2: str, score_cutoff: float = 0.0) -> float:
    return fuzz.token_sort_ratio(_strip_accents(s1), _strip_accents(s2), score_cutoff=score_cutoff)


def clean_address_string(raw: str) -> str:
    """Purge colons, bilingual labels, and non-address OCR noise."""
    if not raw:
        return ""
    
    s = raw.strip()
    
    # 1. If colon exists, strip label prefix if detected
    if ":" in s:
        prefix, remainder = s.split(":", 1)
        norm_prefix = _strip_accents(prefix).lower()
        is_label = any(kw in norm_prefix for kw in [
            "que quan", "queguan", "origin", "thuong tru", "thurong tru", "residence", 
            "cu tru", "khai sinh", "birth", "dia chi", "address", "hktt", "dkhktt", 
            "place of", "placeof", "noi thuong", "noi cu", "noi dang ky", "noi khai sinh"
        ])
        if is_label:
            s = remainder.strip()
        else:
            # If not a recognized label prefix, colon is an OCR typo for comma
            s = s.replace(":", ", ")

    # 2. Strip any remaining colons and semicolons (colons never belong in Vietnamese addresses)
    s = s.replace(":", "").replace(";", ", ")
    
    # 3. Strip bilingual slash: keep slash only between numbers/alphanumeric house codes (e.g. 217/4, 3/2)
    s = re.sub(r"(?<!\d)/|/(?!\d)", " ", s)

    # 3a. Strip boundary duplicate digit from misread colon attached to residence/origin label (e.g. "residence2217" -> "residence 217")
    s = re.sub(r'(?i)\b(residence|origin|birth|tru|quán|quan)[.:;\s\-]*([12])(?=\s*\2\d+)', r'\1 ', s)

    # 3b. Common OCR diacritic misread: "Áp" (not a Vietnamese address term) is
    # almost always a misread of "Ấp" (hamlet/village unit prefix).
    s = re.sub(r"\bÁp\b", "Ấp", s)

    # 3c. Admin abbreviation glued to the name with a hyphen instead of a space
    # (e.g. "TP-Sóc Trăng" -> "TP Sóc Trăng").
    s = re.sub(r"\b(TP|TX|TT|Q|H|P)\s*-\s*", r"\1 ", s)

    # 3d. Normalize abbreviation dots (e.g. "TT." -> "TT", "TP." -> "TP", "P." -> "P")
    s = re.sub(r"\b(TP|TX|TT|Q|H|P)\.\s*", r"\1 ", s)

    # 3e. Insert comma before administrative units if preceded by a word without a comma
    # (e.g. "Ấp Chợ Cũ TT Mỹ Xuyên" -> "Ấp Chợ Cũ, TT Mỹ Xuyên")
    s = re.sub(r'(?<=[^\s,])\s+(?=(?:TT|TX|TP|Xã|Phường|Thị trấn|Quận|Huyện)\b)', ', ', s)

    # 3f. Fix isolated administrative abbreviations (e.g. ", TT, " -> ", TT ")
    s = re.sub(r',\s*(TP|TX|TT|Q|H|P)\s*,\s*', r', \1 ', s)
    
    # 4. Remove isolated pipes or bilingual separator tokens and keyboard noise
    s = re.sub(r"\s+[|I]\s+", " ", s)
    s = re.sub(r"[|~_^*@#\$%]+", " ", s)
    s = re.sub(r'(?i)\b(ctr[li1]|crtl|shif[ti1]|al[ti1])[\d+a-z]*\b', '', s)

    # 5. Clean common bilingual labels that might appear anywhere without colon
    bilingual_patterns = [
        r"(?i)\b(?:[nr]ói|[nr]ơi|[nr]oi|rồi)\s+(?:thương|thường|thuong|thurong)\s+(?:trú|tru|trui)\b",
        r"(?i)\b(?:place|phaco|placo|pace|placeool)\s*(?:of|ool)?\s*(?:residence|nadence|redence|sedines|sedence)\b",
        r"(?i)place\s*of\s*(origin|residence|birth|ongin)",
        r"(?i)noi\s*(thuong|thurong)\s*tru",
        r"(?i)nơi\s*(thường|thuong)\s*trú",
        r"(?i)que\s*[qg]uan",
        r"(?i)quê\s*[qg]uán",
        r"(?i)noi\s*(dang\s*ky\s*)?khai\s*sinh",
        r"(?i)nơi\s*(đăng\s*ký\s*)?khai\s*sinh",
        r"(?i)noi\s*cu\s*tru",
        r"(?i)nơi\s*cư\s*trú",
        r"(?i)citizen\s*identity\s*card",
        r"(?i)can\s*cuoc\s*cong\s*dan",
        r"(?i)căn\s*cước\s*công\s*dân",
        r"(?i)date\s*of\s*expiry",
        r"(?i)co\s*gia\s*tri\s*den",
        r"(?i)có\s*giá\s*trị\s*đến",
        r"(?i)dia\s*chi",
        r"(?i)địa\s*chỉ",
    ]
    for p in bilingual_patterns:
        s = re.sub(p, "", s)

    # 6. Normalize commas and spaces
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[\s,.\-/]+", "", s)
    s = re.sub(r"[\s,.\-/]+$", "", s)
    
    return s.strip()


def deduplicate_address_segments(address: str) -> str:
    """Remove repeating administrative segments or duplicate phrases."""
    if not address:
        return ""
    parts = [p.strip() for p in address.split(",") if p.strip()]
    if len(parts) <= 1:
        return address
    
    # Pass 1: Remove adjacent identical segments (case/accent insensitive)
    clean_parts = []
    for p in parts:
        if not clean_parts:
            clean_parts.append(p)
            continue
        prev_norm = _strip_accents(clean_parts[-1]).lower()
        curr_norm = _strip_accents(p).lower()
        if curr_norm == prev_norm or fuzz.ratio(curr_norm, prev_norm) > 90:
            continue
        clean_parts.append(p)
    parts = clean_parts

    # Pass 2: Detect full block repetition (e.g., A, B, C, A, B, C)
    n = len(parts)
    for block_len in range(1, n // 2 + 1):
        suffix1 = ", ".join(parts[-block_len:])
        suffix2 = ", ".join(parts[-2*block_len:-block_len])
        if fuzz.ratio(_strip_accents(suffix1).lower(), _strip_accents(suffix2).lower()) > 90:
            parts = parts[:-block_len]
            break

    # Pass 3: Drop non-adjacent segments that duplicate an earlier segment
    # (e.g. province name appended twice due to a stray OCR line: "..., Sóc
    # Trăng, Sóc Trăng"), keeping the first occurrence.
    seen_norm: List[str] = []
    deduped = []
    for p in parts:
        norm = _strip_accents(p).lower()
        if any(norm == s or fuzz.ratio(norm, s) > 92 for s in seen_norm):
            continue
        seen_norm.append(norm)
        deduped.append(p)
    parts = deduped

    return ", ".join(parts)


class HierarchicalAddressNormalizer:
    def __init__(self, db_path: str = "resources/hanhchinhvn_tree.json"):
        self.admin_paths = []
        if not os.path.exists(db_path):
            alt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "resources", "hanhchinhvn_tree.json"))
            if os.path.exists(alt_path):
                db_path = alt_path
        if os.path.exists(db_path):
            with open(db_path, "r", encoding="utf-8") as f:
                tree = json.load(f)
            
            paths_set = set()
            
            for p_code, p_data in tree.items():
                p_name = p_data["name"]
                p_wt = p_data["name_with_type"]
                p_variants = [p_name, p_wt]
                if p_wt.startswith("Thành phố "):
                    p_variants.append(f"TP {p_name}")

                for pv in p_variants:
                    paths_set.add(pv)

                if "quan-huyen" in p_data:
                    for d_code, d_data in p_data["quan-huyen"].items():
                        d_name = d_data["name"]
                        d_wt = d_data["name_with_type"]
                        d_variants = [d_wt, d_name]
                        if d_wt.startswith("Thành phố "):
                            d_variants.extend([f"TP {d_name}", f"TP.{d_name}"])
                        elif d_wt.startswith("Thị xã "):
                            d_variants.extend([f"TX {d_name}", f"TX.{d_name}"])
                        elif d_wt.startswith("Thị trấn "):
                            d_variants.extend([f"TT {d_name}", f"TT.{d_name}"])
                        elif d_wt.startswith("Quận "):
                            d_variants.extend([f"Q {d_name}", f"Q. {d_name}"])
                        elif d_wt.startswith("Huyện "):
                            d_variants.extend([f"H. {d_name}"])

                        for dv in d_variants:
                            for pv in p_variants:
                                paths_set.add(f"{dv}, {pv}")

                        if "xa-phuong" in d_data:
                            for w_code, w_data in d_data["xa-phuong"].items():
                                w_name = w_data["name"]
                                w_wt = w_data["name_with_type"]
                                w_is_num = w_name.isdigit()
                                if w_is_num:
                                    w_variants = [f"Phường {w_name}", f"P{w_name}", f"P. {w_name}", f"P.{w_name}"]
                                else:
                                    w_variants = [w_name, w_wt]
                                    if w_wt.startswith("Thị trấn "):
                                        w_variants.extend([f"TT {w_name}", f"TT.{w_name}"])

                                for wv in w_variants:
                                    for dv in d_variants:
                                        for pv in p_variants:
                                            paths_set.add(f"{wv}, {dv}, {pv}")

            self.admin_paths = list(paths_set)

    def normalize(self, raw_address: str) -> str:
        if not raw_address or not self.admin_paths:
            return clean_address_string(raw_address)
            
        # Step 1: Clean colons, labels, and noise
        raw_address = clean_address_string(raw_address)

        # Step 2: Pre-process raw_address to fix joined prefixes & missing spaces
        raw_address = re.sub(r'\b(Ap|AP)([A-ZĐÀÁẠẢÃÈÉẸẺẼÌÍỊỈĨÒÓỌỎÕÙÚỤỦŨƯỪỨỰỬỮƠỜỚỢỞỠỲÝỴỶỸa-zđàáạảãèéẹẻẽìíịỉĩòóọỏõùúụủũưừứựửữơờớợởỡỳýỵỷỹ])', r'Ấp \2', raw_address)
        raw_address = re.sub(r'\b(Xa|XA)([A-ZĐÀÁẠẢÃÈÉẸẺẼÌÍỊỈĨÒÓỌỎÕÙÚỤỦŨƯỪỨỰỬỮƠỜỚỢỞỠỲÝỴỶỸa-zđàáạảãèéẹẻẽìíịỉĩòóọỏõùúụủũưừứựửữơờớợởỡỳýỵỷỹ])', r'Xã \2', raw_address)
        raw_address = re.sub(r'([a-zđàáạảãèéẹẻẽìíịỉĩòóọỏõùúụủũưừứựửữơờớợởỡỳýỵỷỹ])([A-ZĐÀÁẠẢÃÈÉẸẺẼÌÍỊỈĨÒÓỌỎÕÙÚỤỦŨƯỪỨỰỬỮƠỜỚỢỞỠỲÝỴỶỸ])', r'\1 \2', raw_address)
        raw_address = re.sub(r'\b(Ap|ap)\b', 'Ấp', raw_address)
        raw_address = re.sub(r'\b(Xa|xa)\b', 'Xã', raw_address)
        
        # Step 3: Fix common OCR diacritic errors in Vietnamese addresses
        raw_address = re.sub(r'\bMình\s+Duy\b', 'Minh Duy', raw_address)
        raw_address = re.sub(r'\b(Họa Tự|Hóa Tú|Họa Tu|Hóa Tu)\b', 'Hòa Tú', raw_address)
        raw_address = re.sub(r'\b(Cần Thó|Cần Thờ|Cân Thơ|Cân Tho|Can Tho)\b', 'Cần Thơ', raw_address)
        raw_address = re.sub(r'\bĐinh\s+Nam\b', 'Đình Nam', raw_address)

        raw_address = re.sub(r'\s*,\s*', ', ', raw_address)
        raw_address = re.sub(r'\s+', ' ', raw_address).strip()

        parts = [p.strip() for p in raw_address.split(",") if p.strip()]
        if not parts:
            return raw_address
            
        best_match_str = ""
        best_score = 0
        best_suffix_len = 0
        
        # Step 4: Try matching the last N segments (up to 3 for Ward, District, Province)
        max_segments = min(3, len(parts))
        for i in range(1, max_segments + 1):
            suffix = ", ".join(parts[-i:])
            match = process.extractOne(suffix, self.admin_paths, scorer=_unaccented_scorer, score_cutoff=65.0)
            if match:
                first_raw = parts[-i]
                db_segs = [p.strip() for p in match[0].split(",")]
                first_db = db_segs[0]
                
                align_score = fuzz.ratio(_strip_accents(first_db), _strip_accents(first_raw))
                first_raw_words = _strip_accents(first_raw).split()
                first_db_words = _strip_accents(first_db).split()
                if len(first_raw_words) > len(first_db_words):
                    trailing_str = " ".join(first_raw_words[-len(first_db_words):])
                    trailing_score = fuzz.ratio(_strip_accents(first_db), trailing_str)
                    align_score = max(align_score, trailing_score)
                if align_score < 65:
                    continue
                    
                score = match[1]
                partial = fuzz.partial_ratio(_strip_accents(match[0]), _strip_accents(suffix))
                
                combined = score * 0.4 + partial * 0.6 + (i * 8)
                
                if combined > best_score:
                    best_score = combined
                    best_match_str = match[0]
                    best_suffix_len = i
                    
        if best_match_str:
            # Reconstruct address: Unmatched prefix + Matched DB string
            prefix_parts = parts[:-best_suffix_len]
            first_raw_segment = parts[-best_suffix_len]
            db_segments = [p.strip() for p in best_match_str.split(",")]
            first_db_segment = db_segments[0]
            
            raw_words = first_raw_segment.split()
            best_split_idx = 0
            best_split_score = -1
            
            for split_idx in range(len(raw_words)):
                suffix_str = " ".join(raw_words[split_idx:])
                score = fuzz.token_sort_ratio(suffix_str, first_db_segment)
                if score >= best_split_score:
                    best_split_score = score
                    best_split_idx = split_idx
            
            if best_split_idx > 0 and best_split_score > 50:
                prefix = " ".join(raw_words[:best_split_idx])
                prefix_parts.append(prefix)
            
            # Step 5: Deduplicate prefix parts that already exist inside best_match_str
            db_segs_lower = [_strip_accents(p).strip() for p in best_match_str.split(",")]
            clean_prefix = []
            for p in prefix_parts:
                norm_p = _strip_accents(p).strip()
                is_unit_prefix = bool(re.match(r'^(?:ap|thon|ban|to|khu|so)\b', norm_p))
                if not any(norm_p == db_s or (fuzz.ratio(norm_p, db_s) > 90 and not is_unit_prefix) for db_s in db_segs_lower):
                    clean_prefix.append(p)
            prefix_parts = clean_prefix
                
            if prefix_parts:
                res_addr = ", ".join(prefix_parts) + ", " + best_match_str
            else:
                res_addr = best_match_str
        else:
            res_addr = raw_address

        # Step 6: Final deduplication and colon-free assurance
        res_addr = deduplicate_address_segments(res_addr)
        return clean_address_string(res_addr)

_NORMALIZER = HierarchicalAddressNormalizer()

def normalize_address(raw_address: str) -> str:
    """Correct spelling using hierarchical database."""
    return _NORMALIZER.normalize(raw_address)

def reconcile_residence_with_origin(fields: Dict[str, str]) -> Dict[str, str]:
    """Repair place_of_residence's administrative tail using place_of_origin.

    On Vietnamese CCCD, "quê quán" (origin) and "nơi thường trú" (residence)
    almost always share the same ward/district/province path — residence is
    just origin plus a house number/hamlet prefix. Since place_of_origin is
    consistently the more reliably-OCR'd field (it's a short line with no
    house-number noise), use it to fix a residence tail that was truncated,
    duplicated or mis-split by OCR/normalization — without touching a
    residence whose tail genuinely doesn't resemble origin at all (that's
    a real "moved elsewhere" case, not an OCR error).
    """
    origin = fields.get("place_of_origin")
    residence = fields.get("place_of_residence")
    if not origin or not residence:
        return fields

    origin_norm = _strip_accents(origin).lower()
    parts = [p.strip() for p in residence.split(",") if p.strip()]
    if not parts:
        return fields

    # If residence already contains a complete, verified administrative tail matching an admin path,
    # do NOT overwrite its specific abbreviations/wording with origin!
    if len(parts) >= 3:
        tail = ", ".join(parts[-3:])
        tail_2 = ", ".join(parts[-2:])
        if _NORMALIZER.admin_paths:
            m_tail = process.extractOne(tail, _NORMALIZER.admin_paths, scorer=_unaccented_scorer, score_cutoff=90.0)
            if not m_tail:
                m_tail = process.extractOne(tail_2, _NORMALIZER.admin_paths, scorer=_unaccented_scorer, score_cutoff=90.0)
            if m_tail:
                return fields

    # Find the split point whose suffix best matches origin as a whole.
    best_idx, best_score = len(parts), 0.0
    for i in range(len(parts)):
        suffix = ", ".join(parts[i:])
        score = fuzz.ratio(_strip_accents(suffix).lower(), origin_norm)
        if score > best_score:
            best_score = score
            best_idx = i

    # Only repair when the tail is a close-but-imperfect match to origin
    # (OCR noise). A low score means residence is plausibly a different
    # place — leave it alone rather than overwrite real data.
    if best_score < 60:
        return fields

    # If parts[best_idx] is a ward/commune/town (e.g. TT..., Phường..., Xã...)
    # but origin doesn't have it, don't erase the ward! Keep it in prefix_parts.
    matched_seg = parts[best_idx]
    norm_matched = _strip_accents(matched_seg).lower()
    is_ward_level = bool(re.match(r'^(?:tt|tx|phuong|xa|thi tran|p\b|p\.)\b', norm_matched))
    origin_has_ward = norm_matched in origin_norm

    if is_ward_level and not origin_has_ward:
        prefix_parts = parts[:best_idx + 1]
    else:
        prefix_parts = parts[:best_idx]

    res = ", ".join(prefix_parts + [origin]) if prefix_parts else origin
    fields["place_of_residence"] = deduplicate_address_segments(res)
    return fields


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
