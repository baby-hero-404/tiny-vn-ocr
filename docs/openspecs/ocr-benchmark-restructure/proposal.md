# Proposal: OCR Benchmark Restructure

## Problem

The current codebase mixes production pipeline code (`src/`) with ad-hoc experiment scripts (`experiment.py`, `evaluate.py`) at the project root. This creates several issues:

1. **No structured experiment framework** — adding a new OCR method requires copy-pasting boilerplate in `experiment.py`
2. **Experiment scripts write output files into `resources/cccd/`** — polluting the read-only ground truth directory
3. **No separation between "methods" and "evaluation harness"** — adding preprocessing variations or parser tweaks requires editing the monolithic experiment script
4. **No persistent report generation** — results are only printed to stdout and lost

## Goal

Restructure the project so that:
- Each OCR **method** (detection + recognition + post-processing combo) is a self-contained, pluggable unit
- A single **benchmark harness** loads all methods, runs them against `resources/cccd/`, and produces a structured JSON + Markdown report
- Adding a new method = adding one file, not editing the harness
- Ground truth data (all folders in `resources/`) is never modified

## Success

Running `python -m benchmarks.run` produces a ranked report showing which OCR method achieves the highest Exact Match accuracy and lowest CER across all 12 CCCD samples, without modifying any file under `resources/`.

## Decisions

| Decision | Rationale |
|---|---|
| **New `benchmarks/` package at project root** | Keeps experiment code separate from production `src/`. Avoids import confusion. |
| **Method = a function `(image, doc_type) → dict[str, str]`** | Simplest possible contract. Each method returns extracted fields dict. No framework overhead. |
| **Results written to `benchmarks/results/`** | Persistent, git-trackable. Not mixed with ground truth. |
| **Start with 4-5 methods, expand if accuracy < 80%** | RapidOCR-only, VietOCR 2-stage, preprocessing variations, parser-enhanced. If all < 80%, add EasyOCR / PaddleOCR. |
| **Reuse existing `src/` modules** | Don't duplicate — import `OCREngine`, `parse_document`, `enhance_*` from `src/`. |

## Trade-offs

| Gain | Cost |
|---|---|
| Clean separation of concerns | One-time refactoring effort |
| Easy to add new methods | Slightly more files than a single script |
| Reproducible reports | Need to remember to re-run benchmarks after changes |

## Out of Scope

- Changing the production pipeline (`src/pipeline.py`)
- Modifying ground truth data in `resources/cccd/`
- Training custom OCR models
- GPU-dependent methods (everything must run on CPU)
- Non-CCCD document types

## Impact

| Area | Files |
|---|---|
| New | `benchmarks/__init__.py`, `benchmarks/harness.py`, `benchmarks/methods/*.py`, `benchmarks/run.py` |
| Untouched | Everything under `src/`, `resources/cccd/`, `tests/` |
| Can delete later | `experiment.py`, `evaluate.py` (superseded by `benchmarks/`) |
