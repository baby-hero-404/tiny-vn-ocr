"""Benchmark comparing Baseline vs Resource-Optimized OCR Pipeline."""

import time
import json
import cv2
import torch
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

from src.pipeline import DocumentOCRPipeline
from src.ocr.engine import OCREngine


def run_eval(pipeline: DocumentOCRPipeline, dataset: List[Dict[str, Any]], desc: str) -> Dict[str, Any]:
    total_fields = 0
    exact_matches = 0
    latencies = []

    print(f"\n--- Testing {desc} ---")
    for item in dataset:
        img_path = item["image_path"]
        expect_fields = item["expect_fields"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        t0 = time.perf_counter()
        resp = pipeline.process(img, "cccd")
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        pred_fields = resp.fields
        for k, expected_v in expect_fields.items():
            total_fields += 1
            pred_v = pred_fields.get(k, "")
            if str(expected_v).strip().lower() == str(pred_v).strip().lower():
                exact_matches += 1

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    accuracy = (exact_matches / total_fields * 100.0) if total_fields else 0.0

    return {
        "desc": desc,
        "avg_latency_s": avg_latency,
        "total_latency_s": sum(latencies),
        "exact_matches": exact_matches,
        "total_fields": total_fields,
        "accuracy_pct": accuracy,
    }


def main():
    cccd_dir = Path("resources/cccd")
    expect_files = sorted(list(cccd_dir.glob("*_expect.json")))

    dataset = []
    for ef in expect_files:
        stem = ef.stem.replace("_expect", "")
        img_path = cccd_dir / f"{stem}.jpg"
        if not img_path.exists():
            continue
        with open(ef, "r", encoding="utf-8") as f:
            truth = json.load(f)
        dataset.append({
            "image_path": img_path,
            "expect_fields": truth.get("fields", {}),
        })

    print(f"Loaded {len(dataset)} test cases.")

    # 1. Test Baseline (default threads, sequential)
    baseline_pipeline = DocumentOCRPipeline()
    # Warmup
    warmup_img = cv2.imread(str(dataset[0]["image_path"]))
    baseline_pipeline.process(warmup_img, "cccd")
    
    res_baseline = run_eval(baseline_pipeline, dataset, "Baseline Pipeline")

    # 2. Test Thread-Tuned (threads=2)
    torch.set_num_threads(2)
    engine_opt = OCREngine()
    opt_pipeline = DocumentOCRPipeline(ocr_engine=engine_opt)
    res_opt = run_eval(opt_pipeline, dataset, "Thread-Tuned Pipeline (PyTorch threads=2)")

    # 3. Test Dual-Side (Front + Back)
    front_img = cv2.imread("resources/cccd/mtruoc.jpg")
    back_img = cv2.imread("resources/cccd/msau.jpg")

    print("\n--- Testing Dual-Sided Processing ---")
    # Sequential
    t0 = time.perf_counter()
    resp_seq = opt_pipeline.process_dual(front_img, back_img, "cccd")
    t_seq = time.perf_counter() - t0

    # Parallel
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_a = ex.submit(opt_pipeline.ocr_engine.recognize_lines_vietocr_with_bboxes, front_img)
        f_b = ex.submit(opt_pipeline.ocr_engine.recognize_lines_vietocr_with_bboxes, back_img)
        el_a, el_b = f_a.result(), f_b.result()
    t_par_ocr = time.perf_counter() - t0

    print("\n================ BENCHMARK RESULTS ================")
    print(f"{'Method':45s} | {'Avg Latency':12s} | {'Accuracy':10s} | {'Matches':10s}")
    print("-" * 85)
    print(f"{res_baseline['desc']:45s} | {res_baseline['avg_latency_s']:10.2f}s | {res_baseline['accuracy_pct']:8.1f}% | {res_baseline['exact_matches']}/{res_baseline['total_fields']}")
    print(f"{res_opt['desc']:45s} | {res_opt['avg_latency_s']:10.2f}s | {res_opt['accuracy_pct']:8.1f}% | {res_opt['exact_matches']}/{res_opt['total_fields']}")
    print("-" * 85)
    print(f"Dual Sequential Full Time: {t_seq:.2f}s")
    print(f"Dual Parallel OCR Only:     {t_par_ocr:.2f}s")
    print("====================================================")


if __name__ == "__main__":
    main()
