# Implementation Map: OCR Benchmark Restructure

**Goal:** Create a structured benchmark harness to compare multiple OCR methods without modifying the `src/` directory or ground truth data.
**Tech Stack:** Python 3.9+, RapidOCR, VietOCR, Pytest (optional for tests)

---

## Phase 1: Benchmark Framework Core

### Create Registry and Harness

**Why:**
We need a dynamic way to register and run multiple OCR methods without hardcoding them in the evaluation loop.

**Depends on:** None

**Files:**
- `benchmarks/__init__.py`
- `benchmarks/registry.py`
- `benchmarks/harness.py`
- `benchmarks/metrics.py`

**Changes:**
- [x] Create `benchmarks/` package directory
- [x] Implement `@register_method` decorator and central registry in `registry.py`
- [x] Implement `metrics.py` with Exact Match and Levenshtein CER calculations
- [x] Implement `harness.py` to iterate over images, invoke registered methods, and compute metrics

**Verify:**
- [x] Can register a dummy method and retrieve it
- [x] Harness correctly loads `_expect.json` without writing any files

---

## Phase 2: Migrate Baseline Methods

### Implement Method Wrappers

**Why:**
The existing baseline methods (RapidOCR only and RapidOCR+VietOCR) need to conform to the new `(image, doc_type) -> dict` contract.

**Depends on:** Phase 1

**Files:**
- `benchmarks/methods/__init__.py`
- `benchmarks/methods/baseline_rapidocr.py`
- `benchmarks/methods/baseline_vietocr.py`
- `benchmarks/methods/enhanced_preprocessing.py`

**Changes:**
- [x] Implement `method_rapidocr` using `OCREngine().recognize_lines` and `parse_document`
- [x] Implement `method_vietocr` using `OCREngine().recognize_lines_vietocr` and `parse_document`
- [x] Implement `method_enhanced` using custom CLAHE/Sharpening before passing to OCR
- [x] Register all methods using `@register_method`

**Verify:**
- [x] Methods correctly return field dictionaries
- [x] No exceptions during method execution

---

## Phase 3: Reporting and CLI

### Build the CLI Runner and Report Generator

**Why:**
We need a simple command to run the suite and generate persistent JSON and Markdown reports.

**Depends on:** Phase 1, Phase 2

**Files:**
- `benchmarks/run.py`
- `benchmarks/report.py`

**Changes:**
- [x] Implement Markdown report generation (summary table, per-field accuracy)
- [x] Implement JSON report generation (full dump of predictions vs ground truth)
- [x] Create `run.py` CLI script (using `argparse` or just `sys.argv`)
- [x] Ensure output is saved to `benchmarks/results/` (create dir if not exists)

**Verify:**
- [x] `python -m benchmarks.run` executes successfully
- [x] Markdown report contains correct tables and rankings
- [x] Ground truth folder `resources/cccd/` remains untouched

---

## Phase 4: Clean Up & Evaluate

### Run Benchmark and Remove Old Scripts

**Why:**
Run the benchmark to determine the best method, and clean up the old ad-hoc scripts to reduce technical debt.

**Depends on:** Phase 3

**Files:**
- `experiment.py` (Delete)
- `evaluate.py` (Delete)

**Changes:**
- [x] Run full benchmark suite
- [x] Review results and identify the best performing method (update `problem_description.md` if needed)
- [x] Delete `experiment.py` and `evaluate.py`

**Verify:**
- [x] Benchmark results are clear and actionable
- [x] Project directory is cleaner
