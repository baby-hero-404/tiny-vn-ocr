"""Method using a Graph-based spatial parser for VietOCR bounding boxes."""

import re
import numpy as np
from typing import Dict, List, Any
from rapidfuzz import process, fuzz

from benchmarks.registry import register_method
from src.ocr.engine import OCREngine
from src.postprocessing.parser import _strip_accents

_ENGINE = OCREngine()

class Node:
    def __init__(self, element: Dict[str, Any]):
        self.text = element["text"]
        self.bbox = element["bbox"]  # [xmin, ymin, xmax, ymax]
        self.center = element["center"]
        self.height = element.get("height", self.bbox[3] - self.bbox[1])
        
        self.xmin, self.ymin, self.xmax, self.ymax = self.bbox
        
        # Edges
        self.right_nodes: List['Node'] = []
        self.below_nodes: List['Node'] = []

def _x_overlap(n1: Node, n2: Node) -> float:
    overlap = max(0, min(n1.xmax, n2.xmax) - max(n1.xmin, n2.xmin))
    # Return percentage of overlap relative to the smaller width
    w1 = n1.xmax - n1.xmin
    w2 = n2.xmax - n2.xmin
    if w1 == 0 or w2 == 0: return 0.0
    return overlap / min(w1, w2)

def _y_overlap(n1: Node, n2: Node) -> float:
    overlap = max(0, min(n1.ymax, n2.ymax) - max(n1.ymin, n2.ymin))
    h1 = n1.ymax - n1.ymin
    h2 = n2.ymax - n2.ymin
    if h1 == 0 or h2 == 0: return 0.0
    return overlap / min(h1, h2)

def merge_horizontal_boxes(elements: List[Dict]) -> List[Dict]:
    if not elements:
        return []
        
    # Sort elements by Y first, then X
    sorted_els = sorted(elements, key=lambda e: (e["center"][1], e["center"][0]))
    
    merged = []
    current_line = []
    
    for el in sorted_els:
        if not current_line:
            current_line.append(el)
            continue
            
        last = current_line[-1]
        
        # Check if same line
        y_overlap = max(0, min(el["bbox"][3], last["bbox"][3]) - max(el["bbox"][1], last["bbox"][1]))
        min_h = min(el["bbox"][3] - el["bbox"][1], last["bbox"][3] - last["bbox"][1])
        is_same_line = False
        
        if min_h > 0 and (y_overlap / min_h) > 0.4:
            x_dist = el["bbox"][0] - last["bbox"][2]
            if x_dist < min_h * 4: # Allow a reasonable gap
                is_same_line = True
                
        if is_same_line:
            current_line.append(el)
        else:
            merged.append(current_line)
            current_line = [el]
            
    if current_line:
        merged.append(current_line)
        
    final_elements = []
    for line in merged:
        text = " ".join([e["text"] for e in line])
        xmin = min(e["bbox"][0] for e in line)
        ymin = min(e["bbox"][1] for e in line)
        xmax = max(e["bbox"][2] for e in line)
        ymax = max(e["bbox"][3] for e in line)
        
        final_elements.append({
            "text": text,
            "bbox": [xmin, ymin, xmax, ymax],
            "center": [(xmin + xmax)/2, (ymin + ymax)/2],
            "height": ymax - ymin
        })
        
    return final_elements

def build_graph(elements: List[Dict]) -> List[Node]:
    nodes = [Node(el) for el in elements]
    
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if i == j: continue
            
            # n2 is to the RIGHT of n1 if:
            # - n2 is to the right (n2.xmin > n1.center[0])
            # - they share significant Y overlap (e.g., > 30%)
            if n2.xmin > n1.center[0] and _y_overlap(n1, n2) > 0.3:
                n1.right_nodes.append(n2)
                
            # n2 is BELOW n1 if:
            # - n2 is physically below (n2.center[1] > n1.center[1])
            # - they share some X overlap or are very close in X
            if n2.center[1] > n1.center[1]:
                # Either they overlap in X, or n2 starts slightly before/after n1
                x_dist = min(abs(n2.xmin - n1.xmin), abs(n2.center[0] - n1.center[0]))
                if _x_overlap(n1, n2) > 0.1 or x_dist < n1.height * 2:
                    n1.below_nodes.append(n2)
                    
    # Sort edges by distance
    for n in nodes:
        n.right_nodes.sort(key=lambda x: x.xmin - n.xmax)
        n.below_nodes.sort(key=lambda x: x.ymin - n.ymax)
        
    return nodes

def find_node_by_label(nodes: List[Node], keywords: List[str]) -> Node:
    for n in nodes:
        text_norm = _strip_accents(n.text.lower())
        for kw in keywords:
            if kw in text_norm:
                return n
            if fuzz.partial_ratio(kw, text_norm) > 85:
                return n
    return None

def graph_based_parse_cccd(elements: List[Dict]) -> Dict[str, str]:
    fields = {}
    if not elements:
        return fields
        
    nodes = build_graph(elements)
    full_text = " ".join([n.text for n in nodes])
    
    # 1. Regex fallbacks for standard formats
    id_text = full_text.replace(" ", "")
    id_match = re.search(r'\b\d{12}\b', id_text)
    if id_match:
        fields["id_number"] = id_match.group(0)
        
    combined_lower = _strip_accents(full_text.lower())
    if "nam" in combined_lower and not "nu" in combined_lower:
        fields["gender"] = "NAM"
    elif "nu" in combined_lower:
        fields["gender"] = "NỮ"
    if "viet nam" in combined_lower:
        fields["nationality"] = "Việt Nam"

    # --- Full Name ---
    name_node = find_node_by_label(nodes, ["ho va ten", "full name"])
    dob_node = find_node_by_label(nodes, ["ngay sinh", "date of birth"])
    
    if name_node:
        if ":" in name_node.text and len(name_node.text.split(":", 1)[1].strip()) > 3:
            fields["full_name"] = name_node.text.split(":", 1)[1].strip()
        else:
            name_blocks = []
            max_y = dob_node.center[1] if dob_node else name_node.center[1] + 100
            for bn in name_node.below_nodes:
                if bn.center[1] < max_y and not re.search(r'\d', bn.text) and len(bn.text) > 2:
                    name_blocks.append(bn.text)
            if name_blocks:
                fields["full_name"] = " ".join(name_blocks)

    # --- Date of Birth ---
    dob_node = find_node_by_label(nodes, ["ngay sinh", "date of birth"])
    if dob_node:
        # Check right nodes first
        val = None
        if dob_node.right_nodes:
            val = dob_node.right_nodes[0].text
        elif dob_node.below_nodes:
            val = dob_node.below_nodes[0].text
            
        if val:
            match = re.search(r'\d{2}[/.\-]\d{2}[/.\-]\d{4}', val)
            if match:
                fields["date_of_birth"] = match.group(0).replace(".", "/").replace("-", "/")

    if "date_of_birth" not in fields:
        match = re.search(r'\d{2}[/.\-]\d{2}[/.\-]\d{4}', full_text)
        if match:
            fields["date_of_birth"] = match.group(0).replace(".", "/").replace("-", "/")

    # --- Origin & Residence (Multi-line) ---
    origin_node = find_node_by_label(nodes, ["que quan", "place of origin"])
    residence_node = find_node_by_label(nodes, ["thuong tru", "place of residence"])
    expiry_node = find_node_by_label(nodes, ["co gia tri den", "date of expiry"])
    
    def extract_graph_block(start_node: Node, end_node: Node, default_max_y: float) -> str:
        if not start_node:
            return ""
            
        end_y = end_node.center[1] if end_node else default_max_y
        
        # We collect all nodes that are below the start_node, but above end_y
        # And we use X overlap to ensure they belong to the value column
        
        block = []
        for n in nodes:
            if n.center[1] > start_node.center[1] - 10 and n.center[1] < end_y:
                # Is it roughly on the same X or to the right?
                if n.xmin > start_node.xmin - start_node.height * 2:
                    block.append(n)
                    
        block.sort(key=lambda x: (x.center[1] // 15, x.xmin))
        
        text = " ".join([b.text for b in block])
        if ":" in text:
            text = text.split(":", 1)[1]
        text = re.sub(r'(?i)(quê quán|que quan|place of origin|nơi thường trú|thuong tru|place of residence|noi th)[^\w]*', '', text)
        return text.replace(" ,", ",").strip()

    if origin_node:
        fields["place_of_origin"] = extract_graph_block(origin_node, residence_node, origin_node.center[1] + 150)
    if residence_node:
        fields["place_of_residence"] = extract_graph_block(residence_node, expiry_node, residence_node.center[1] + 200)

    return fields

@register_method("graph_based_vietocr")
def method_graph_based_vietocr(image: np.ndarray, doc_type: str) -> Dict[str, str]:
    raw_elements = _ENGINE.recognize_lines_vietocr_with_bboxes(image)
    elements = merge_horizontal_boxes(raw_elements)
    
    if doc_type in ["cccd", "cccd_front"]:
        return graph_based_parse_cccd(elements)
    else:
        from benchmarks.methods.enhanced_parser import enhanced_parse_cccd_back
        lines = [el["text"] for el in elements]
        return enhanced_parse_cccd_back(lines)
