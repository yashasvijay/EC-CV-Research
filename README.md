# Adaptive Streaming Tracking on Apple Silicon

A producer–consumer runtime for tracking-by-detection: the main thread propagates sparse Lucas–Kanade (LK) tracks at source cadence, while a background worker executes YOLO11n ONNX inference and prepares source-time corrections. Bounded queues replace stale pending detector work; the tracking thread owns live state and publishes immutable outputs. Cropped ROI feature selection avoids repeated full-image corner extraction.

The project studies throughput, latency and tracking fidelity under constrained inference capacity. It includes four swappable schedulers, MOT17 adapters, causal/deadline-valid TrackEval exports, stage profiling, and a versioned SLO verifier. The runtime is implemented; full tracking qualification remains open.

## SLO verification table

The saved **MOT17-02-FRCNN, FixedKPolicy(3), 600-frame, 30 Hz paced replay** passed its original version-1 run-level timing contract:

| Metric | Verified measurement | Target | Timing status |
|---|---:|---:|---|
| Source-capped throughput | **30.0 FPS** | ≥30.0 FPS | PASS |
| Capture-to-output p95 | **5.56 ms** | <33 ms | PASS |
| Capture-to-output p99 | **5.61 ms** | <33 ms | PASS |
| Deadline misses | **0.17%** (1/600; fraction 0.001667) | <1.0% | PASS |

**This run published zero tracks in all 600 frames and fails the current version-2 verifier.** Detector freshness was null; none of the completed correction proposals was fused into a live track. These numbers demonstrate timely frame publication under this workload, not successful 30 FPS tracking. HOTA/MOTA/IDF1 remain unevaluated. Physical-camera latency, repeated-run robustness, power savings and tracking fidelity are not established.

The detector completed 123 calls; 77 pending requests were superseded. Result rejection counts were 88 for target lag, 28 for replay budget, five for deadline headroom and one for commit budget. Detector service averaged 82.82 ms. Process CPU averaged 110.67% (100% = one logical core); sampled peak RSS was 328.94 MiB. Idle preflight averaged 3.60% host CPU; thermal conditions were uncontrolled. Throughput is capped at source rate by the verifier; the runtime's raw short-window estimate was 30.0419 FPS.

[Original SLO report](results/slos/mot17-02-fixed3/slo_report.json) · [Frame ledger](results/slos/mot17-02-fixed3/runtime/frames.jsonl) · [Evidence and limitations](docs/verified-performance.md)

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
  --run-dir results/slos/mot17-02-fixed3/runtime \
  --output results/slos/mot17-02-fixed3-recheck
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

The `OraclePolicy` reference requests every frame through the same queues; it has no theoretical optimality claim. The earlier synchronous 48-configuration grid is complete. Its [Pareto plot](reports/ablations/pareto_frontier.png) is historical accuracy-versus-inference evidence, not an async policy frontier. The full real four-policy ROI sweep remains pending.

The original synchronous MOT17-02 comparison measured 3.84→4.21 FPS, 600→484 detector calls and HOTA 25.82→25.94. Preserve those [historical results](docs/results-mot17-02.md) separately from the ROI profile and empty-output async run.

## Documentation and validation

- [Architecture](docs/architecture.md), [implemented async API](docs/async-runtime.md), [scheduler contracts](docs/schedulers.md).
- [Design/SLO contract](docs/DESIGN.md), [performance evidence](docs/verified-performance.md), [technical report](reports/final_report_draft.md).
- [Development, CI and evaluation commands](docs/engineering.md), [profiling](docs/pipeline-profiling.md), [historical reproducibility](docs/reproducibility.md).

Ruff and strict mypy cover the source tree; pytest includes deterministic async lifecycle/overload/fusion tests, ROI containment, scheduler boundaries, synthetic ONNX/TrackEval integration, and strict SLO failure cases. Optional native tests require the separate extension build. The publication validation records the current full-suite outcome; older reports retain their original test counts. CI configuration and local checks are distinct from a remotely completed GitHub Actions run.

Publication check (2026-10-18): `make test` passed **137 tests**; `make lint` passed Ruff and strict mypy across **26 source modules**. See [validation record](reports/publication-validation/validation.json). These software checks do not qualify tracking accuracy.

## Cold-start correction

The runtime now bootstraps supported tracks synchronously when no live tracks exist and batches LK across track points during propagation/replay/commit. Bootstrap cost remains measured. The effective replay cap reserves time inside the commit-lag window. Contract version 2 requires populated outputs and a useful background-fusion ratio of at least 10%; bootstrap-only and empty-output runs fail. Regression checks pass, but a corrected real-hardware benchmark and HOTA/MOTA evaluation remain pending.
