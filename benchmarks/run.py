"""CLI entrypoint for running benchmarks."""

import sys
from pathlib import Path

# Important: import methods to trigger registration
import benchmarks.methods
from benchmarks.harness import run_benchmarks
from benchmarks.report import generate_reports

def main():
    print("Starting OCR Benchmark...")
    results = run_benchmarks(resources_dir="resources")
    
    if not results:
        print("Benchmark failed or no dataset found.")
        sys.exit(1)
        
    print("\nGenerating reports...")
    json_path, md_path = generate_reports(results)
    print(f"JSON Report: {json_path}")
    print(f"Markdown Report: {md_path}")
    
    # Print summary to stdout
    print("\n=== SUMMARY ===")
    summary = results.get("summary", {})
    sorted_methods = sorted(
        summary.items(),
        key=lambda x: (x[1]["accuracy_percent"], -x[1]["cer_percent"]),
        reverse=True
    )
    for i, (method_name, stats) in enumerate(sorted_methods, 1):
        acc = stats["accuracy_percent"]
        cer = stats["cer_percent"]
        print(f"{i}. {method_name:30s} | Acc: {acc:5.1f}% | CER: {cer:5.1f}%")
    
    # Print field-wise accuracy for top method
    if sorted_methods:
        top_name, top_stats = sorted_methods[0]
        field_acc = top_stats.get("field_accuracy", {})
        if field_acc:
            print(f"\n=== FIELD-WISE ACCURACY (top: {top_name}) ===")
            for fname, fdata in sorted(field_acc.items()):
                acc_pct = fdata["accuracy_percent"]
                matches = fdata["exact_matches"]
                total = fdata["total"]
                print(f"  {fname:25s} | {acc_pct:5.1f}% ({matches}/{total})")

if __name__ == "__main__":
    main()
