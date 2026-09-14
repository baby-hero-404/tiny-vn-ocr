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
        f.write("| Rank | Method | Exact Match | Critical (ID+Name+DOB) | CER (%) | Avg Time (ms) |\n")
        f.write("|---|---|---|---|---|---|\n")
        
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
            crit_matches = stats.get("critical_matches", 0)
            crit_total = stats.get("images_processed", 0)
            crit_acc = stats.get("critical_accuracy_percent", 0.0)
            f.write(f"| {i} | `{method_name}` | {acc:.2f}% ({exact}/{total}) | {crit_acc:.2f}% ({crit_matches}/{crit_total}) | {cer:.2f}% | {time_ms:.0f} |\n")
        
        # Field-wise Accuracy section
        f.write("\n## Field-wise Accuracy\n\n")
        
        # Collect all field names across all methods
        all_fields = set()
        for method_name, stats in sorted_methods:
            field_acc = stats.get("field_accuracy", {})
            all_fields.update(field_acc.keys())
        
        # Canonical field order
        field_order = [
            "id_number", "full_name", "date_of_birth", "date_of_expiry",
            "gender", "nationality", "place_of_origin", "place_of_residence",
            "date_of_issue", "place_of_issue", "place_of_birth",
        ]
        ordered_fields = [f for f in field_order if f in all_fields]
        ordered_fields += sorted(all_fields - set(field_order))
        
        if ordered_fields:
            # Header
            f.write("| Method |")
            for field in ordered_fields:
                short_name = field.replace("place_of_", "").replace("date_of_", "")
                f.write(f" {short_name} |")
            f.write("\n")
            
            f.write("|---|")
            for _ in ordered_fields:
                f.write("---|")
            f.write("\n")
            
            # Data rows
            for method_name, stats in sorted_methods:
                field_acc = stats.get("field_accuracy", {})
                f.write(f"| `{method_name}` |")
                for field in ordered_fields:
                    fa = field_acc.get(field, {})
                    if fa:
                        acc_pct = fa["accuracy_percent"]
                        matches = fa["exact_matches"]
                        total = fa["total"]
                        f.write(f" {acc_pct:.0f}% ({matches}/{total}) |")
                    else:
                        f.write(" — |")
                f.write("\n")
            
        f.write("\n## Field-wise Accuracy by Document Type\n\n")
        
        # Determine all document types present
        all_dtypes = set()
        for _, stats in sorted_methods:
            all_dtypes.update(stats.get("doctype_accuracy", {}).keys())
            
        for dtype in sorted(all_dtypes):
            f.write(f"### Document Type: `{dtype}`\n\n")
            f.write("| Method |")
            for field in ordered_fields:
                short_name = field.replace("place_of_", "").replace("date_of_", "")
                f.write(f" {short_name} |")
            f.write("\n")
            
            f.write("|---|")
            for _ in ordered_fields:
                f.write("---|")
            f.write("\n")
            
            for method_name, stats in sorted_methods:
                dtype_acc = stats.get("doctype_accuracy", {}).get(dtype, {})
                f.write(f"| `{method_name}` |")
                for field in ordered_fields:
                    fa = dtype_acc.get(field, {})
                    if fa:
                        acc_pct = fa["accuracy_percent"]
                        matches = fa["exact_matches"]
                        total = fa["total"]
                        f.write(f" {acc_pct:.0f}% ({matches}/{total}) |")
                    else:
                        f.write(" N/A |")
                f.write("\n")
            f.write("\n")
            
        f.write("## Detailed Results\n\n")
        for method_name, _ in sorted_methods:
            f.write(f"### {method_name}\n\n")
            evals = results.get("evaluations", {}).get(method_name, [])
            if not evals:
                f.write("No evaluations.\n\n")
                continue
                
            f.write("| Image | Exact | Critical | CER | Time (ms) |\n")
            f.write("|---|---|---|---|---|\n")
            for e in evals:
                img_name = e["image_name"]
                m = e["metrics"]
                acc = (m["exact_matches"] / m["total_fields"] * 100) if m["total_fields"] > 0 else 0
                cer = (m["total_cer"] / m["total_fields"] * 100) if m["total_fields"] > 0 else 0
                crit = "✅" if m.get("is_critical_match") else "❌"
                f.write(f"| {img_name} | {m['exact_matches']}/{m['total_fields']} ({acc:.1f}%) | {crit} | {cer:.1f}% | {m['processing_time_ms']:.0f} |\n")
            f.write("\n")
