"""Method using VietOCR but parsing via bounding box layout logic."""

import re
import numpy as np
from typing import Dict, List, Any
from rapidfuzz import process, fuzz

from benchmarks.registry import register_method
from src.ocr.engine import OCREngine
from src.postprocessing.parser import _strip_accents

_ENGINE = OCREngine()

def _find_closest_node(
    ref_center_x: float,
    ref_center_y: float,
    elements: List[Dict],
    direction: str,
    max_dist_y: float = 100,
    max_dist_x: float = 300
) -> Dict:
    """Find the closest node to the right or below."""
    best_node = None
    min_dist = float('inf')
    
    for el in elements:
        cx, cy = el["center"]
        
        # Calculate distances
        dx = cx - ref_center_x
        dy = cy - ref_center_y
        
        if direction == "right":
            # Must be strictly to the right, and roughly on the same Y line
            if dx > 0 and abs(dy) < max_dist_y and dx < max_dist_x:
                if dx < min_dist:
                    min_dist = dx
                    best_node = el
        elif direction == "below":
            # Must be below, and roughly aligned on X
            if dy > 0 and abs(dx) < max_dist_x and dy < max_dist_y:
                dist = dy + abs(dx) * 2  # Penalize X misalignment heavily
                if dist < min_dist:
                    min_dist = dist
                    best_node = el
                    
    return best_node

def _find_label_node(elements: List[Dict], label_keywords: List[str]) -> Dict:
    """Find the node that matches the label keywords using fuzzy string matching."""
    for el in elements:
        text_norm = _strip_accents(el["text"].lower())
        for kw in label_keywords:
            if kw in text_norm:
                return el
            # Or use rapidfuzz if partial match is high
            if fuzz.partial_ratio(kw, text_norm) > 85:
                return el
    return None

def layout_aware_parse_cccd_front(elements: List[Dict]) -> Dict[str, str]:
    fields = {}
    if not elements:
        return fields
        
    # Combine all text to extract globally format-specific things first
    full_text = " ".join([e["text"] for e in elements])
    
    # 1. ID Number (can be found reliably via regex anywhere in the text)
    id_text = full_text.replace(" ", "")
    id_match = re.search(r'\b\d{12}\b', id_text)
    if id_match:
        fields["id_number"] = id_match.group(0)
        
    # 2. Gender & Nationality (Classification)
    combined_lower = _strip_accents(full_text.lower())
    if "nam" in combined_lower and not "nu" in combined_lower:
        fields["gender"] = "NAM"
    elif "nu" in combined_lower:
        fields["gender"] = "NỮ"
    if "viet nam" in combined_lower:
        fields["nationality"] = "Việt Nam"

    # Layout parsing for Name, DOB, Origin, Residence
    
    # --- Full Name ---
    name_label = _find_label_node(elements, ["ho va ten", "full name"])
    if name_label:
        # Sometimes the name is on the same line after a colon
        if ":" in name_label["text"] and len(name_label["text"].split(":", 1)[1].strip()) > 3:
            fields["full_name"] = name_label["text"].split(":", 1)[1].strip()
        else:
            # Usually below
            node = _find_closest_node(name_label["center"][0], name_label["center"][1], elements, "below", max_dist_y=150)
            if node:
                fields["full_name"] = node["text"]

    # --- Date of Birth ---
    dob_label = _find_label_node(elements, ["ngay sinh", "date of birth"])
    if dob_label:
        node = _find_closest_node(dob_label["center"][0], dob_label["center"][1], elements, "right", max_dist_y=30)
        if node:
            match = re.search(r'\d{2}[/.\-]\d{2}[/.\-]\d{4}', node["text"])
            if match:
                fields["date_of_birth"] = match.group(0).replace(".", "/").replace("-", "/")
            else:
                fields["date_of_birth"] = node["text"]

    # If no DOB found, try global regex
    if "date_of_birth" not in fields:
        match = re.search(r'\d{2}[/.\-]\d{2}[/.\-]\d{4}', full_text)
        if match:
            fields["date_of_birth"] = match.group(0).replace(".", "/").replace("-", "/")

    # --- Place of Origin and Residence ---
    # These can be multi-line. We find the label, and collect all text boxes below it, until we hit the next label.
    origin_label = _find_label_node(elements, ["que quan", "place of origin"])
    residence_label = _find_label_node(elements, ["thuong tru", "place of residence"])
    expiry_label = _find_label_node(elements, ["co gia tri den", "date of expiry"])
    
    def extract_multiline_block(start_label, end_label, default_max_y):
        if not start_label:
            return ""
            
        start_y = start_label["center"][1]
        end_y = end_label["center"][1] if end_label else default_max_y
        
        # Gather all nodes that are between start_y and end_y
        # And their x center is somewhat aligned (to avoid taking the photo text if any)
        block_nodes = []
        for el in elements:
            cy = el["center"][1]
            if start_y - 10 < cy < end_y:
                block_nodes.append(el)
                
        # Sort top to bottom, left to right
        block_nodes.sort(key=lambda x: (x["center"][1] // 15, x["center"][0]))
        
        text_parts = [n["text"] for n in block_nodes]
        block_text = " ".join(text_parts)
        
        # Remove the label text from the block
        if ":" in block_text:
            block_text = block_text.split(":", 1)[1]
        else:
            block_text = re.sub(r'(?i)(quê quán|que quan|place of origin|nơi thường trú|thuong tru|place of residence|noi th)[^\w]*', '', block_text)
            
        return block_text.replace(" ,", ",").strip()

    if origin_label:
        fields["place_of_origin"] = extract_multiline_block(origin_label, residence_label, origin_label["center"][1] + 150)
    
    if residence_label:
        fields["place_of_residence"] = extract_multiline_block(residence_label, expiry_label, residence_label["center"][1] + 200)

    return fields

@register_method("layout_aware_vietocr")
def method_layout_aware_vietocr(image: np.ndarray, doc_type: str) -> Dict[str, str]:
    """Extract text using VietOCR with bounding boxes, then parse using spatial layout."""
    # Run the new engine method
    elements = _ENGINE.recognize_lines_vietocr_with_bboxes(image)
    
    # We currently only wrote layout parser for CCCD front.
    # For back, it falls back to empty unless we add layout_aware_parse_cccd_back.
    # To keep things simple, if doc_type is cccd_back, we just return empty dict for now,
    # or we can fallback to enhanced_parser for the back.
    # The benchmark will just show low accuracy for the back.
    # Let's fallback to enhanced parser for the back to keep overall accuracy reasonable.
    
    if doc_type == "cccd" or doc_type == "cccd_front":
        return layout_aware_parse_cccd_front(elements)
    else:
        # Fallback to enhanced parser for back
        from benchmarks.methods.enhanced_parser import enhanced_parse_cccd_back
        lines = [el["text"] for el in elements]
        return enhanced_parse_cccd_back(lines)
