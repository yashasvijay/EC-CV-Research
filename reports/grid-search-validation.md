# Adaptive grid search validation

Date: 2026-10-12

Command: `.venv/bin/python -m pytest -q`

Result: **32 passed in 57.01 seconds**.

Six grid-search tests cover:

- Pareto dominance, mixed min/max objectives, exact ties and missing metrics.
- The requested default grid has 48 combinations.
- Run-level means/sample SD and filtering by absolute baseline HOTA loss.
- A real CLI search on two synthetic configurations using ONNX Runtime and
  TrackEval, including baseline evaluation, full-frame accuracy metrics,
  JSON/CSV exports, successful resume and incompatible-resume rejection.
- Interrupted-search retries preserving the previous attempt directory.
- Video input without ground truth, with accuracy kept null.

Existing Week 1/2 tests also passed. Documentation links and attribution checks,
syntax parsing and `git diff --check` passed.

The synthetic fixtures have independently known detections/ground truth. Their
inference reduction and accuracy validate plumbing and do not establish a real
MOT17 grid-search result. The requested 48-run real search was not launched as
part of implementation. The existing MOT17-04 saved runs were inspected and
confirmed complete over 1,050 frames: baseline 2.83499 FPS; adaptive 3.08336 FPS;
adaptive detection fraction 0.70095238. These saved summaries had not yet been
accuracy-evaluated at inspection time.
