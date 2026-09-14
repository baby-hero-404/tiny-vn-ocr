"""Method using Enhanced Parser + Address Normalization (Vietnamese Post-correction)."""

import numpy as np
from typing import Dict, List
from rapidfuzz import process, fuzz
import json
import os

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
            
            def add_aliases(name, path, path_with_type):
                paths_set.add(name)
                paths_set.add(path)
                paths_set.add(path_with_type)
                # Create shorthand aliases
                # e.g., "Thành phố Sóc Trăng" -> "TP Sóc Trăng"
                replacements = [
                    ("Thành phố ", "TP "),
                    ("Thành phố ", "TP."),
                    ("Thị xã ", "TX "),
                    ("Thị xã ", "TX."),
                    ("Thị trấn ", "TT "),
                    ("Thị trấn ", "TT."),
                    ("Quận ", "Q. "),
                    ("Quận ", "Q"),
                    ("Phường ", "P. "),
                    ("Phường ", "P"),
                    ("Huyện ", "H. "),
                    ("Tỉnh ", "")
                ]
                for old, new in replacements:
                    if old in path_with_type:
                        paths_set.add(path_with_type.replace(old, new))
                        paths_set.add(path_with_type.replace(old, new).replace(" ,", ",").strip())

            for p_code, p_data in tree.items():
                add_aliases(p_data["name"], p_data["name"], p_data["name_with_type"])
                if "quan-huyen" in p_data:
                    for d_code, d_data in p_data["quan-huyen"].items():
                        add_aliases(d_data["name"], d_data["path"], d_data["path_with_type"])
                        if "xa-phuong" in d_data:
                            for w_code, w_data in d_data["xa-phuong"].items():
                                add_aliases(w_data["name"], w_data["path"], w_data["path_with_type"])
            self.admin_paths = list(paths_set)

    def normalize(self, raw_address: str) -> str:
        if not raw_address or not self.admin_paths:
            return raw_address
            
        # Pre-process raw_address to fix common missing spaces after hyphens
        raw_address = raw_address.replace("TP-", "TP ").replace("TX-", "TX ").replace("TT.", "TT ")
        parts = [p.strip() for p in raw_address.split(",")]
        if not parts:
            return raw_address
            
        best_match_str = ""
        best_score = 0
        best_suffix_len = 0
        
        # Try matching the last N segments (up to 3 for Ward, District, Province)
        max_segments = min(3, len(parts))
        for i in range(1, max_segments + 1):
            suffix = ", ".join(parts[-i:])
            match = process.extractOne(suffix, self.admin_paths, scorer=fuzz.token_sort_ratio, score_cutoff=65.0)
            if match:
                first_raw = parts[-i]
                db_segs = [p.strip() for p in match[0].split(",")]
                first_db = db_segs[0]
                
                # Check if the first segment aligns structurally
                align_score = fuzz.partial_ratio(first_db.lower(), first_raw.lower())
                if align_score < 50:
                    continue
                    
                score = match[1]
                partial = fuzz.partial_ratio(match[0].lower(), suffix.lower())
                
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
                
            if prefix_parts:
                return ", ".join(prefix_parts) + ", " + best_match_str
            else:
                return best_match_str
                
        return raw_address

_NORMALIZER = HierarchicalAddressNormalizer()

def normalize_address(raw_address: str) -> str:
    """Correct spelling using hierarchical database."""
    return _NORMALIZER.normalize(raw_address)

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
