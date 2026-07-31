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

if __name__ == "__main__":
    main()
