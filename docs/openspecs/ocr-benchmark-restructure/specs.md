# Expected Behavior: OCR Benchmark Restructure

## Scenario: Run full benchmark suite
**When:**
- User runs `python -m benchmarks.run`

**Then:**
- All registered methods are discovered and executed against all 12 CCCD images
- Per-image, per-field comparison against `*_expect.json` is computed
- Summary table printed to stdout with ranked methods
- Detailed JSON report saved to `benchmarks/results/report_<timestamp>.json`
- Markdown report saved to `benchmarks/results/report_<timestamp>.md`

## Scenario: Add a new OCR method
**When:**
- Developer creates `benchmarks/methods/my_new_method.py` with a function decorated `@register_method("my_method_name")`

**Then:**
- The harness automatically discovers and includes it in the next benchmark run
- No changes to `harness.py` or `run.py` required

## Scenario: Ground truth integrity
**When:**
- Any benchmark is executed

**Then:**
- No file under `resources/cccd/` is created, modified, or deleted
- Ground truth is loaded read-only

## Rules

- **Method contract**: Every method must implement `def extract(image: np.ndarray, doc_type: str) -> dict[str, str]` — receives a BGR image + document type string, returns a flat dict of field_name → field_value.
- **Auto-detection**: If `doc_type` in the expect JSON is `cccd` or `cccd_back`, the method receives that value. Methods may also implement auto-detection internally.
- **Evaluation metrics**: Exact Match (field-level), CER (Levenshtein-based), Processing Time (wall clock ms). All three are mandatory.
- **Ranking**: Methods are ranked by Exact Match accuracy (primary), then CER (secondary, lower is better).

## Constraints

- All methods must run on **CPU only** (no CUDA).
- Methods must not modify the input image array (operate on copies).
- Benchmark must complete within **30 minutes** for the full 12-image dataset on a standard machine.
