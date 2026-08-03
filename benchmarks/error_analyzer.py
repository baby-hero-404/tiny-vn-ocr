"""Error Analysis Scaffolding for OCR Benchmarks."""

import json
import argparse
from pathlib import Path
from collections import defaultdict

def analyze_errors(report_path: str, target_method: str = "best_practice_vietocr"):
    """
    Analyzes the benchmark report and groups errors into buckets to facilitate
    root cause analysis (e.g. OCR recognition error vs Parser/Logic error).
    """
    with open(report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    evals = data.get("evaluations", {}).get(target_method, [])
    if not evals:
        print(f"No evaluations found for {target_method}")
        return
        
    buckets = defaultdict(list)
    
    for e in evals:
        img_name = e["image_name"]
        truth = e.get("truth_fields", {})
        pred = e.get("pred_fields", {})
        
        has_error = False
        error_details = []
        
        for field, t_val in truth.items():
            if not str(t_val).strip():
                continue
                
            p_val = str(pred.get(field, "")).strip()
            
            if t_val != p_val:
                has_error = True
                error_details.append({
                    "field": field,
                    "truth": t_val,
                    "pred": p_val
                })
                
                # Simple categorization heuristic
                if p_val == "":
                    buckets["missing_extraction"].append((img_name, field, t_val, p_val))
                elif len(t_val) > 0 and len(p_val) > 0:
                    # If lengths are similar, probably OCR spelling mistake
                    if abs(len(t_val) - len(p_val)) <= 2:
                        buckets["ocr_recognition_suspect"].append((img_name, field, t_val, p_val))
                    else:
                        # Major discrepancy, maybe parser boundary error or missed detection
                        buckets["parser_boundary_suspect"].append((img_name, field, t_val, p_val))
                        
    print(f"=== Error Analysis for {target_method} ===")
    for bucket_name, errors in buckets.items():
        print(f"\\n--- Bucket: {bucket_name} ({len(errors)} cases) ---")
        # Only show up to 10 examples per bucket
        for err in errors[:10]:
            img, f, t, p = err
            print(f"  {img[:15]}... | Field: {f:15s} | Truth: '{t}' | Pred: '{p}'")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze benchmark errors.")
    parser.add_argument("report_json", help="Path to the report.json file")
    parser.add_argument("--method", default="validated_best_practice_vietocr", help="Method to analyze")
    args = parser.parse_args()
    
    analyze_errors(args.report_json, args.method)
