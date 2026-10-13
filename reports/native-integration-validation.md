# Native comparison and opt-in integration validation

Date: 2026-10-12

Command: `.venv/bin/python -m pytest -q tests/test_native_integration.py`

Result: **9 passed in 4.45 seconds**.

Verified scheduling parity for adaptive/baseline/fixed modes, disabled-native
execution delegating to the current Python implementation without extension
loading, temporary-copy application of the direct pipeline/CLI patch, benchmark
JSON/JSONL/resource exports, refusal to time an active search by default, and
isolated native CLI execution using actual ONNX Runtime on four synthetic frames.
Native CLI metadata includes backend, extension version and binary hash.

Benchmark reporting was exercised with three diagnostic calls per backend and
suite. These tiny synthetic measurements are not reported as performance results.
The intended default 10,000-call timing comparison was not launched while the
real grid search was running. No performance or accuracy gain is established.

All code hashes recorded by the active grid-search manifest were verified
unchanged after this work. `src/pipeline.py`, `scripts/run_phase1.py`, the grid
search source, active manifests and per-run records were not edited. No active
process was restarted. Runtime dependencies and native binaries were unchanged.

The standalone runner and new native helpers are available now. The requested
direct constructor flag and standard CLI switch are fully prepared in
`csrc/patches/native-gating-integration.patch` but remain unapplied while the
search is active. The apply helper refuses active searches and validates before/
after source hashes. The patch was tested only in a temporary copy of the source.

Documentation and source attribution checks and `git diff --check` passed.
