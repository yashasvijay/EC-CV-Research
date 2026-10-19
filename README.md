# Adaptive Streaming Tracking on Apple Silicon

A producer–consumer runtime for tracking-by-detection: the main thread propagates sparse Lucas–Kanade (LK) tracks at source cadence, while a background worker executes YOLO11n ONNX inference and prepares source-time corrections. Bounded queues replace stale pending detector work; the tracking thread owns live state and publishes immutable outputs. Cropped ROI feature selection avoids repeated full-image corner extraction.

The project studies throughput, latency and tracking fidelity under constrained inference capacity. It includes four swappable schedulers, MOT17 adapters, causal/deadline-valid TrackEval exports, stage profiling, and a versioned SLO verifier. The runtime is implemented; full tracking qualification remains open.

## Historical SLO verification table (contract v3)

The saved **MOT17-02-FRCNN, FixedKPolicy(3), 600-frame, 30 Hz paced replay** maintained tracks on every frame. The original contract-v2 artifacts and their full raw logs are committed under `results/slos/mot17-02-fixed3-populated/`; [contract-v3 revalidation](results/slos/mot17-02-fixed3-populated/verification-v3/slo_report.json) additionally enforces ≥95% track-frame coverage.

| Metric | Verified measurement | Target | Status |
|---|---:|---:|---|
| Source-capped throughput | **30.0 FPS** | ≥30.0 FPS | PASS |
| Capture-to-output p95 | **26.21 ms** | <33 ms | PASS |
| Capture-to-output p99 | **29.74 ms** | <33 ms | PASS |
| Deadline misses | **0.67%** (4/600) | <1.0% | PASS |
| Track-frame fraction | **100%** (600/600) | ≥95% | PASS |
| Useful fusion ratio | **70.6%** (125/177 background runs) | ≥10% | PASS |

`useful_fusion_ratio` counts completed background detector runs with **at least one accepted correction**, divided by all completed background detector runs. It is not the fraction of individual boxes accepted. Bootstrap and empty/rejected results do not enter its numerator.

The first track appeared after **97.04 ms**; synchronous bootstrap cost is included in the deadline population. The worker completed 177 background runs and 125 yielded useful fusion. Mean CPU was 143.66% (100% = one logical core), sampled peak RSS 341.59 MiB, and mean detector service 86.57 ms. This is one populated paced-replay result, not repeated-run or physical-camera certification. HOTA/MOTA remain unevaluated, and power was not measured.

[Original SLO report](results/slos/mot17-02-fixed3-populated/slo_report.json) · [Raw frame ledger](results/slos/mot17-02-fixed3-populated/runtime/frames.jsonl) · [Evidence and scope](docs/verified-performance.md)

The older 30 FPS / 5.61 ms p99 run published no tracks and fails current gates. Its artifacts remain separate and are not the current tracking result. The populated measurement predates subsequent concurrency changes. Contract v4 additionally requires per-track detection-age p95 <300 ms; the historical export lacks this field and cannot pass v4. No new hardware performance is claimed for this refactor.

## Localized ROI optimization

The original code passed an entire 1080p image to `goodFeaturesToTrack` for each track, despite using a bounding-box output mask. The shared selector now crops the image first, uses a crop-local mask, restores global point coordinates, and enforces strict box bounds.

Native feature-call attribution fell from **114.62 to 1.31 ms/frame** (approximately **98.86%**). The separate paired selector benchmark, including mask construction and coordinate offsets, measured **147.80 to 2.10 ms/frame: 98.58%**, or roughly 98.5%. These are different measurement boundaries. Both describe feature selection rather than total pipeline speedup; corner populations changed, and post-change fidelity remains unqualified.

[ROI results and repeat variation](reports/roi-features/results.md) · [Blameless gating COE](docs/POSTMORTEM_GATING.md)

## Quick start

Use Python 3.13 and a development environment separate from saved benchmark environments:

```bash
python3.13 -m venv .venv-dev
source .venv-dev/bin/activate
pip install -e '.[dev,eval]'
export EVAL_PYTHON="$PWD/.venv-dev/bin/python"
make test
make lint
```

`pip install -e .` installs only runtime dependencies. `make install` adds development tools. Native C++ gating is optional and is not required by the async runtime. MOT17 images/annotations and model weights must be acquired separately; do not assume a clean clone contains them.

Run a new SLO experiment (idle preflight is enforced; existing output directories are never overwritten):

```bash
python -m scripts.verify_slos \
  --sequence data/raw/MOT17/train/MOT17-02-FRCNN \
  --model models/yolo11n.onnx --policy fixed_k --k 3 \
  --output results/slos/mot17-02-fixed3-repeat
```

Recheck the saved telemetry without inference:

```bash
python -m scripts.verify_slos \
  --run-dir results/slos/mot17-02-fixed3-populated/runtime \
  --output results/slos/mot17-02-fixed3-populated-recheck
```

The verifier writes `slo_report.json` and `slo_report.md`, plus runtime telemetry for a new run. Exit 0 means all timing/integrity gates pass, 1 means a measured gate fails, and 2 means invalid input or deferred execution. A timing PASS is never tracking release approval. See the [machine-readable contract and ADRs](docs/DESIGN.md).

## Architecture and evaluation

| Component | Responsibility |
|---|---|
| `src/async_pipeline.py` | Main-thread ownership, propagation, nonblocking admission and guarded fusion |
| `src/runtime/` | Bounded queues, detector worker, leased history and motion replay |
| `src/schedulers/` | Fixed-K, Kalman uncertainty, adaptive heuristic, every-frame oracle reference |
| `src/features.py` | Cropped ROI feature selection shared by tracker and replay worker |
| `src/detector.py` | Person-only YOLO ONNX preprocessing, execution and NMS |
| `src/pipeline.py`, `src/sort_tracker.py` | Synchronous controls and eight-state SORT variant |
| `scripts/run_async_evaluation.py` | Full-sequence causal and deadline-valid accuracy evaluation |
| `scripts/ablation_policies.py`, `scripts/plot_pareto.py` | Policy sweep and accuracy/performance figures |

The `OraclePolicy` reference requests every frame through the same queues; it has no theoretical optimality claim. The earlier synchronous 48-configuration grid is complete. Its [Pareto plot](reports/ablations/pareto_frontier.png) is historical accuracy-versus-inference evidence, not an async policy frontier. The populated result covers FixedKPolicy(3); it does not provide an evaluated four-policy frontier.

The original synchronous MOT17-02 comparison measured 3.84→4.21 FPS, 600→484 detector calls and HOTA 25.82→25.94. Preserve those [historical results](docs/results-mot17-02.md) separately from the ROI profile and populated async run.

## Documentation and validation

- [Architecture](docs/architecture.md), [implemented async API](docs/async-runtime.md), [scheduler contracts](docs/schedulers.md).
- [Design/SLO contract](docs/DESIGN.md), [performance evidence](docs/verified-performance.md), [technical report](reports/final_report_draft.md).
- [Development, CI and evaluation commands](docs/engineering.md), [profiling](docs/pipeline-profiling.md), [historical reproducibility](docs/reproducibility.md).

Ruff and strict mypy cover the source tree; pytest includes deterministic async lifecycle/overload/fusion tests, ROI containment, scheduler boundaries, synthetic ONNX/TrackEval integration, and strict SLO failure cases. Optional native tests require the separate extension build. The publication validation records the current full-suite outcome; older reports retain their original test counts. CI configuration and local checks are distinct from a remotely completed GitHub Actions run.

Audit check (2026-10-18): `make lint` passes Ruff and strict mypy across **27 source modules**. The [current validation record](reports/audit-hardening/validation.json) records full-suite and targeted regression outcomes. These software checks do not qualify tracking accuracy.

## Cold-start correction

The runtime bootstraps supported tracks synchronously when no live tracks exist, with at least 15 source frames between attempts by default. Empty scenes continue through the background scheduler during this cooldown instead of blocking on every frame. Bootstrap cost remains measured. Batched LK and the effective replay cap protect the main-thread lag window. Contract version 3 requires ≥95% track-frame coverage and useful fusion ratio ≥10%. The populated hardware result above verifies the prior bootstrap/replay correction; the latency injection tool below maps synthetic overload separately.

## Detector-latency operating envelope

```bash
python -m scripts.sweep_detector_latency \
  --latencies-ms 40 80 160 320 --frames 600 --repeats 1 \
  --output results/detector-latency-envelope
```

The sweep runs deterministic moving textured boxes with exact synthetic detections and injected minimum detector-call delays. It writes per-case raw runtime/SLO artifacts plus summary JSON/CSV for coverage, useful fusion ratio, deadlines, latency and synchronous fallback frequency. Each case uses a fresh process, seeded order and idle-host preflight. Sleep injection models service delay, not slower-hardware CPU/power behavior or learned detector accuracy. This tool is implemented and tested; a full four-delay measured envelope is not claimed here.
