"""Report generation for benchmark results."""

import json
import datetime
from pathlib import Path
from typing import Dict, Any

def generate_reports(results: Dict[str, Any], output_dir: str = "benchmarks/results") -> tuple[Path, Path]:
    """
    Generate JSON and Markdown reports from benchmark results.
    Returns (json_path, md_path).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_path = out_path / f"report_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    md_path = out_path / f"report_{timestamp}.md"
    _write_markdown_report(results, md_path)
    
    return json_path, md_path

def _write_markdown_report(results: Dict[str, Any], filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# OCR Benchmark Report\n\n")
        f.write(f"**Dataset Size**: {results.get('dataset_size', 0)} images\n")
        f.write(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary Ranking\n\n")
        f.write("| Rank | Method | Exact Match Accuracy | CER (%) | Avg Time (ms) |\n")
        f.write("|---|---|---|---|---|\n")
        
        # Sort methods by accuracy (descending), then CER (ascending)
        summary = results.get("summary", {})
        sorted_methods = sorted(
            summary.items(),
            key=lambda x: (x[1]["accuracy_percent"], -x[1]["cer_percent"]),
            reverse=True
        )
        
        for i, (method_name, stats) in enumerate(sorted_methods, 1):
            acc = stats["accuracy_percent"]
            cer = stats["cer_percent"]
            time_ms = stats["avg_time_ms"]
            exact = stats["exact_matches"]
            total = stats["total_fields"]
            f.write(f"| {i} | `{method_name}` | {acc:.2f}% ({exact}/{total}) | {cer:.2f}% | {time_ms:.0f} |\n")
            
        f.write("\n## Detailed Results\n\n")
        for method_name, _ in sorted_methods:
            f.write(f"### {method_name}\n\n")
            evals = results.get("evaluations", {}).get(method_name, [])
            if not evals:
                f.write("No evaluations.\n\n")
                continue
                
            f.write("| Image | Exact | CER | Time (ms) |\n")
            f.write("|---|---|---|---|\n")
            for e in evals:
                img_name = e["image_name"]
                m = e["metrics"]
                acc = (m["exact_matches"] / m["total_fields"] * 100) if m["total_fields"] > 0 else 0
                cer = (m["total_cer"] / m["total_fields"] * 100) if m["total_fields"] > 0 else 0
                f.write(f"| {img_name} | {m['exact_matches']}/{m['total_fields']} ({acc:.1f}%) | {cer:.1f}% | {m['processing_time_ms']:.0f} |\n")
            f.write("\n")
