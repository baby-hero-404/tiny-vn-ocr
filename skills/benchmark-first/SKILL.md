---
name: benchmark-first
description: "Use before implementing or merging any change to OCR/postprocessing logic (src/ocr, src/postprocessing, src/pipeline.py). Triggers on: new extraction method, parser rule change, normalization change, accuracy claim, 'improve accuracy', 'cải thiện độ chính xác'."
allowed-tools: Read, Write, Edit, Grep, Glob, Bash
---

# Benchmark-First Development

## Overview

This project exposes OCR extraction quality through `src/` (the production API,
served via `src/api/app.py` → `src/pipeline.py`). Any change to extraction/
parsing/normalization logic must be proven on the benchmark dataset **before**
it is considered done — not after, not "it looks right in a manual test".

**Core principle:** No accuracy claim without a benchmark report. No production
change without the exact production code path having been run through
`benchmarks/run.py`.

## The Iron Law

```
NO NEW/CHANGED SRC LOGIC WITHOUT A BEFORE/AFTER BENCHMARK REPORT
```

If you changed anything under `src/ocr/`, `src/postprocessing/`, or
`src/pipeline.py`, you are not done until you have run the benchmark suite
and can show the accuracy delta.

## Why this project needs it specifically

- `src/pipeline.py::DocumentOCRPipeline` is exposed to the API. It also has
  logic (side auto-detect, dual-image merge, ROI fallback, MRZ cross-check)
  that does **not** exist in the lighter benchmark methods like
  `best_practice_vietocr`. A method that scores well in isolation can still
  regress once composed inside `DocumentOCRPipeline`.
- To close that gap, `benchmarks/methods/best_practice_vietocr.py` registers a
  `production_pipeline` method that wraps `src.pipeline.DocumentOCRPipeline`
  directly (`method_production_pipeline`). **This is the method that reflects
  what users actually get from the API.** Always check its row in the report,
  not just `best_practice_vietocr`.
- Several postprocessing modules (`address_norm.py`, `cccd_rules.py`,
  `vn_name_dict.py`, `layout_parser.py`) are imported directly by both
  `src/pipeline.py` and `benchmarks/methods/best_practice_vietocr.py`. This is
  correct and must stay this way — do not let a benchmark method fork into its
  own duplicate copy of postprocessing logic (older files like
  `benchmarks/methods/cccd_rules.py`, `enhanced_parser.py`,
  `vn_name_dict.py`, `layout_graph_parser.py` are historical baselines kept
  only for comparison; never wire new src/ logic changes into those instead
  of the real src/ modules).

## Workflow

### 1. Research/Prototype in `benchmarks/`, not in `src/`

For a new idea (new parsing rule, new normalization heuristic, new fallback):

1. Prototype it inside a **new or existing benchmark method**
   (`benchmarks/methods/<name>.py`), registered via `@register_method(...)`.
   It's fine to prototype logic inline here first.
2. Run `make benchmarks` (or `uv run python -m benchmarks.run`) to get a
   baseline report before touching `src/`.
3. Only after the prototype beats the current best method (or a specific
   regressing field is fixed) on accuracy/CER/field-wise metrics, port the
   logic into the real `src/` module.

### 2. Implement into `src/`, keep it wired into benchmarks

1. Move the logic into the appropriate `src/postprocessing/*.py` or
   `src/ocr/*.py` module.
2. Make sure a benchmark method imports it from `src.*` (not a duplicated
   copy) — `best_practice_vietocr` and `production_pipeline` must both
   exercise the new code path.
3. If the change affects `src/pipeline.py` directly (orchestration, fallback
   logic, dual-image merge, resize thresholds, etc.), you MUST rely on the
   `production_pipeline` benchmark row — `best_practice_vietocr` alone will
   not catch it.

### 3. Re-run the full benchmark and record before/after

1. `uv run python -m benchmarks.run`
2. Compare against the most recent report in `benchmarks/results/` for:
   - overall `accuracy_percent` / `cer_percent`
   - field-wise accuracy for the fields you touched
   - critical-field validation (id_number, dob, expiry, etc.)
3. If a field regressed, do not merge — fix or revert, then re-run.
4. Keep the resulting `benchmarks/results/report_*.md` — it is the evidence
   for the "why" behind the change (link it from the PR description /
   `docs/ai/` decision log per the project's CLAUDE.md documentation rule).

### 4. Only then update tests / open a PR

`tests/test_pipeline.py` and `tests/test_normalizer.py` cover unit-level
correctness; they do not replace the benchmark run — both are required.

## Red flags — stop and go back to step 1

- "I improved the regex, should be more accurate" with no benchmark run.
- A fix applied only inside a `benchmarks/methods/*.py` file that is not
  imported by `src/` (means production never gets the fix).
- A fix applied only inside `src/` without checking the `production_pipeline`
  benchmark row (means the fix wasn't actually verified end-to-end).
- Duplicated logic added to one of the legacy standalone benchmark files
  (`benchmarks/methods/cccd_rules.py`, `vn_name_dict.py`, `enhanced_parser.py`,
  `layout_graph_parser.py`) instead of the shared `src/postprocessing/*`
  modules — these legacy files are frozen baselines, not where new logic goes.
