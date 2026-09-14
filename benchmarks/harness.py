"""Benchmark harness to run registered OCR methods against the dataset."""

import json
import time
import cv2
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

from benchmarks.registry import get_all_methods
from benchmarks.metrics import evaluate_predictions, evaluate_field_wise, evaluate_critical_fields

def discover_dataset(resources_dir: Path) -> List[Dict[str, Any]]:
    """
    Find all pairs of image and ground truth JSON in the resources directory.
    Assumes naming convention: image.jpg + image_expect.json
    """
    dataset = []
    
    # We will search recursively in resources/ for _expect.json or _expected.json
    expect_files = list(resources_dir.rglob("*_expect.json"))
    expect_files.extend(resources_dir.rglob("*_expected.json"))
    
    for expect_file in sorted(set(expect_files)):
        stem = expect_file.stem.replace("_expect", "").replace("_expected", "")
        parent_dir = expect_file.parent
        
        # Try to find corresponding image in the same directory
        img_path = parent_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = parent_dir / f"{stem}.png"
            
        if not img_path.exists():
            print(f"Warning: Skipping {expect_file.name}, matching image not found.")
            continue
            
        with open(expect_file, 'r', encoding='utf-8') as f:
            truth_data = json.load(f)
            
        dataset.append({
            "image_path": img_path,
            "expect_path": expect_file,
            "truth_data": truth_data,
            "doc_type": truth_data.get("document_type", "cccd")
        })
        
    return dataset

def run_benchmarks(resources_dir: str = "resources") -> Dict[str, Any]:
    """
    Run all registered methods against the dataset in resources_dir.
    Returns a dictionary with full benchmark results.
    """
    dataset = discover_dataset(Path(resources_dir))
    if not dataset:
        print("No dataset found.")
        return {}
        
    methods = get_all_methods()
    print(f"Found {len(dataset)} items. Running {len(methods)} methods...")
    
    results = {
        "dataset_size": len(dataset),
        "methods": list(methods.keys()),
        "evaluations": {method_name: [] for method_name in methods.keys()},
        "summary": {}
    }
    
    # Process each image, looping over methods per image
    for item in tqdm(dataset, desc="Evaluating Images", unit="img"):
        img_path = item["image_path"]
        doc_type = item["doc_type"]
        truth_data = item["truth_data"]
        truth_fields = truth_data.get("fields", {})
        
        # We define which fields to check based on doc type
        fields_to_check = list(truth_fields.keys())
        if not fields_to_check:
            fields_to_check = ["id_number", "full_name", "date_of_birth", "gender", "place_of_origin", "place_of_residence", "date_of_expiry", "date_of_issue", "place_of_issue"]
        
        # Load image once per item
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue
            
        for method_name, method_func in methods.items():
            start_time = time.time()
            try:
                pred_fields = method_func(image, doc_type)
            except Exception as e:
                print(f"Error running {method_name} on {img_path.name}: {e}")
                pred_fields = {}
                
            processing_time_ms = (time.time() - start_time) * 1000
            
            total_fields, exact_matches, total_cer = evaluate_predictions(truth_fields, pred_fields, fields_to_check)
            field_wise = evaluate_field_wise(truth_fields, pred_fields, fields_to_check)
            is_critical_match = evaluate_critical_fields(truth_fields, pred_fields, ["id_number", "full_name", "date_of_birth"])
            
            results["evaluations"][method_name].append({
                "image_name": img_path.name,
                "doc_type": doc_type,
                "truth_fields": truth_fields,
                "pred_fields": pred_fields,
                "metrics": {
                    "total_fields": total_fields,
                    "exact_matches": exact_matches,
                    "total_cer": total_cer,
                    "processing_time_ms": processing_time_ms,
                    "is_critical_match": is_critical_match
                },
                "field_wise": field_wise
            })
            
    # Aggregate summary
    for method_name in methods.keys():
        evals = results["evaluations"][method_name]
        if not evals:
            continue
            
        total_fields_all = sum(e["metrics"]["total_fields"] for e in evals)
        exact_matches_all = sum(e["metrics"]["exact_matches"] for e in evals)
        total_cer_all = sum(e["metrics"]["total_cer"] for e in evals)
        total_time_all = sum(e["metrics"]["processing_time_ms"] for e in evals)
        critical_matches_all = sum(1 for e in evals if e["metrics"].get("is_critical_match"))
        
        acc = (exact_matches_all / total_fields_all) * 100 if total_fields_all > 0 else 0
        avg_cer = (total_cer_all / total_fields_all) * 100 if total_fields_all > 0 else 0
        avg_time = total_time_all / len(evals) if evals else 0
        critical_acc = (critical_matches_all / len(evals)) * 100 if evals else 0
        
        # Aggregate field-wise accuracy, including by doc_type
        field_stats: Dict[str, Dict[str, int]] = {}  # field -> {"matches": n, "total": n, "cer_sum": float}
        doctype_field_stats: Dict[str, Dict[str, Dict[str, int]]] = {} # doc_type -> field -> stats
        
        for e in evals:
            dtype = e.get("doc_type", "unknown")
            if dtype not in doctype_field_stats:
                doctype_field_stats[dtype] = {}
                
            for field_name, fdata in e.get("field_wise", {}).items():
                # Global stats
                if field_name not in field_stats:
                    field_stats[field_name] = {"matches": 0, "total": 0, "cer_sum": 0.0}
                field_stats[field_name]["total"] += 1
                field_stats[field_name]["matches"] += fdata["exact_match"]
                field_stats[field_name]["cer_sum"] += fdata["cer"]
                
                # By doc_type stats
                if field_name not in doctype_field_stats[dtype]:
                    doctype_field_stats[dtype][field_name] = {"matches": 0, "total": 0}
                doctype_field_stats[dtype][field_name]["total"] += 1
                doctype_field_stats[dtype][field_name]["matches"] += fdata["exact_match"]
        
        field_accuracy = {}
        for fname, fstats in field_stats.items():
            field_accuracy[fname] = {
                "accuracy_percent": (fstats["matches"] / fstats["total"] * 100) if fstats["total"] > 0 else 0,
                "avg_cer_percent": (fstats["cer_sum"] / fstats["total"] * 100) if fstats["total"] > 0 else 0,
                "exact_matches": fstats["matches"],
                "total": fstats["total"],
            }
        
        doctype_accuracy = {}
        for dtype, d_stats in doctype_field_stats.items():
            doctype_accuracy[dtype] = {}
            for fname, fstats in d_stats.items():
                doctype_accuracy[dtype][fname] = {
                    "accuracy_percent": (fstats["matches"] / fstats["total"] * 100) if fstats["total"] > 0 else 0,
                    "exact_matches": fstats["matches"],
                    "total": fstats["total"],
                }
                
        results["summary"][method_name] = {
            "exact_matches": exact_matches_all,
            "total_fields": total_fields_all,
            "accuracy_percent": acc,
            "cer_percent": avg_cer,
            "avg_time_ms": avg_time,
            "images_processed": len(evals),
            "field_accuracy": field_accuracy,
            "doctype_accuracy": doctype_accuracy,
            "critical_matches": critical_matches_all,
            "critical_accuracy_percent": critical_acc
        }
        
    return results
