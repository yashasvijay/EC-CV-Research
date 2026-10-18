# Native gating comparison and opt-in integration

> Current scope (2026-10-18): C++ gating remains optional and separate from the primary async runtime. Validate the prepared patch against current source before applying it; do not reinterpret native correctness as hardware speedup. See [verified performance](verified-performance.md).

## Preserve experiment provenance

The completed historical grid started a new pipeline process for each configuration and hashed
its Python source files for resume/reproducibility. Changing `src/pipeline.py`
or `scripts/run_phase1.py` before it finishes would change later job inputs,
even if a native flag were disabled. Those original experiment artifacts are preserved; current source versions have since changed.
The implementation provides an isolated native CLI and a guarded, unapplied direct-integration patch. It is optional historical work, not a dependency of the async runtime.

This work does not replace the LK solver or tracking reductions in the production
pipeline. The opt-in flag routes **scheduling decisions** to the C++ engine;
LK, box association, reseeding and detector execution continue using the existing
Python pipeline. The benchmark also measures the native motion-reduction API
separately to inform a future tracker integration.

## Standalone comparison

Run from the repository root after the active grid search finishes:

```bash
source .venv/bin/activate
python -m scripts.benchmark_native --iterations 10000 --output results/benchmark_native
```

The compiled module is loaded from `csrc/build/python`; use
`--native-module-dir /path/to/module` to select another build. The Python ABI
and architecture must match. No installation into the pipeline environment is
performed. The benchmark uses the compiled module from the earlier native build.

Defaults measure 10,000 iterations for every backend in these suites:

- **Scheduling:** Python min/max and the current branch priority; C++ with
  NumPy packing included; and a prepacked C++ diagnostic. The primary reported
  scheduling ratio compares Python against native **with packing**.
- **Motion reductions:** the NumPy LK reference and native forward/backward
  consistency, median translation, drift confidence and displacement. Native
  result conversion back to NumPy is included to make output contracts comparable.

Use `--scope decisions` or `--scope motion` to select one suite, `--repeats 5`
for repeated blocks, and `--hold-gil` to assess GIL-release overhead. Run order
is randomized with a recorded seed. Synthetic cases include stable tracks,
confidence/motion/interval triggers, no tracks and rejected feature points.
Numerical/decision parity is checked before measurement; no inference model,
video source, LK solver or ground truth is executed.

The script refuses timing while a recorded grid manifest has status `running`.
`--allow-concurrent` is only for diagnostic/smoke checks: concurrent results are
labeled with active searches and should not be used as uncontended performance
measurements. A stale running manifest also causes refusal; resolve the original
search's state instead of silently editing it to bypass the guard.

Output:

- `calls.jsonl`: every call's microsecond latency, backend/suite/repetition,
  sampled CPU/RSS and CPU refresh flag.
- `summary.json`: mean/p50/p95/p99 latency, calls/second, instrumented throughput,
  per-block CPU time/usage and peak RSS, repeated mean/SD, module hash, runtime
  versions and Python/native ratios. One repetition gives null SD.

Ratios above 1 indicate lower native latency; below 1 indicate slower native
execution. Packing and binding overhead may outweigh savings for small tracking
sets. The microbenchmark does not establish video FPS, MOTA/HOTA, watts or an
end-to-end speedup. These metrics remain null where not measured.

Only the callable is inside the latency window. Resource sampling, result
destruction and JSONL output are outside it and included in instrumented block
throughput/CPU measurements. CPU percentages use 100% for one logical core;
RSS includes process/runtime state. Repeated quantiles are averaged per-block
quantiles rather than pooled call quantiles.

## Isolated flag available now

After concurrent timing experiments finish, the isolated runner accepts the
same options as `scripts.run_phase1` plus the native switch:

```bash
python -m scripts.run_native_phase1 \
  --model models/yolo11n.onnx \
  --source data/raw/MOT17/train/MOT17-04-FRCNN/img1 \
  --mode adaptive --output results/adaptive-04-native \
  --threads 1 --max-interval 10 --min-confidence 0.6 --max-displacement 0.15 \
  --use-native-gating
```

Without the flag, it delegates to the original Python runner. With it,
`src/native_pipeline.py` uses the immutable C++ engine configuration and logs
`gating_ms`, `gating_backend` and module version/hash. Inputs are valid float64
metric vectors; packing is inside pipeline timing. Invalid native metrics
force a defensive refresh, as documented in the native module. SORT currently
rejects the native switch because this integration targets LK scheduling.

The wrapper temporarily rebinds factories only inside its own new interpreter
process; it never modifies the original source files, another process, grid
commands, manifests or Python runtime packages. Default tracker behavior and
trigger selection delegate directly to the existing pipeline.

## Direct pipeline/CLI patch after completion

The direct constructor option `Pipeline(..., use_native_gating=True)` and
`python -m scripts.run_phase1 ... --use-native-gating` are implemented in
`csrc/patches/native-gating-integration.patch`, but are **not yet applied** to
the live files. Apply only when searches have completed:

```bash
python -m scripts.apply_native_integration
python -m scripts.apply_native_integration --apply
PYTHONPATH="$PWD/csrc/build/python" python -m scripts.run_phase1 \
  --model models/yolo11n.onnx \
  --source data/raw/MOT17/train/MOT17-04-FRCNN/img1 \
  --mode adaptive --output results/adaptive-04-native-direct \
  --threads 1 --use-native-gating
```

The apply helper rejects active searches, checks original source hashes and
patch integrity, runs `git apply --check`, and verifies the applied hashes.
It refuses stale patches rather than overwriting newer edits. Applying the
patch changes source fingerprints; future searches should use new output
directories. The default remains Python, with no extension import unless enabled.

The patch was applied and exercised only in a temporary copy during tests.
Direct integration and the isolated runner share the same native decision
helper. Compare full native/Python detector schedules and TrackEval metrics
before interpreting speed or accuracy differences. Engine thresholds are set
at construction; create a new pipeline when changing its configuration.
