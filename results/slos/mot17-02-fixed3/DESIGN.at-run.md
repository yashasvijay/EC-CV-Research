# Streaming tracking system specification

Status: target specification; performance qualification pending. Revision: 1, 2026-10-18.

## Scope and service boundary

The runtime accepts monotonically numbered BGR frames, propagates sparse Lucas–Kanade (LK) features, consults a scheduler, and publishes immutable tracks. YOLO11n ONNX inference and historical correction preparation run in one background worker. The main thread owns live track state. The deployment target is Apple Silicon ARM64, 1920×1080, 30 source frames/s, COCO pedestrian detections, one configured OpenCV/ORT thread per component. Python thread count does not imply a single process-wide CPU core.

Camera-rate throughput and latency are **targets**, not demonstrated capabilities. The ROI synchronous profile measured 8.06 FPS and 143.44 ms pipeline p99 over 50 frames with instrumentation; it is not an asynchronous end-to-end measurement. Full-sequence async qualification remains necessary. See [measured ROI results](../reports/roi-features/results.md).

## SLO contract

The following JSON block is the sole executable threshold definition. `scripts/verify_slos.py` reads it directly; changes require reviewing this specification and the verification tests together.

<!-- SLO_CONTRACT_START -->
```json
{
  "version": 1,
  "camera_fps": 30.0,
  "throughput_min_fps": 30.0,
  "p95_e2e_max_ms": 33.0,
  "p99_e2e_max_ms": 33.0,
  "deadline_ms": 33.0,
  "deadline_miss_max_fraction": 0.01,
  "minimum_frames": 600
}
```
<!-- SLO_CONTRACT_END -->

| SLI | Acceptance rule | Measurement |
|---|---|---|
| Throughput | ≥30 FPS | Published source frames / max(observed wall duration, N/30); caps startup-biased estimates at source rate |
| End-to-end p95 | <33 ms | Linear NumPy percentile of nominal capture-to-publication latency |
| End-to-end p99 | <33 ms | Same capture-to-publication population, including late frames |
| Deadline misses | <1% | Missing source frames plus publications with latency **≥33 ms**, divided by all expected frames |
| Measurement integrity | PASS | Complete contiguous ledger from frame 1, ≥600 frames, successful worker shutdown, matching 30 FPS / 33 ms runtime configuration and consistent timestamps |

Equality fails each strict upper bound: 33 ms is late and 1% misses fails. A 30 FPS cadence is 33.333… ms; the 33 ms budget intentionally leaves a small margin. Runtime `deadline_missed` currently uses `>`; the verifier recomputes `>=` from timestamps to enforce this stricter contract. It never substitutes frame-work latency for capture-to-output latency.

A paced replay assigns capture times from nominal release slots before processing. It retains all input frames; accumulated processing backlog increases end-to-end latency. This includes image decoding relative to release but excludes camera exposure, hardware acquisition, network transport and downstream rendering. Production camera qualification must supply monotonic acquisition timestamps at the same boundary; paced replay alone cannot certify a physical camera system.

Warmup initializes the detector before the measurement window. No tracking bootstrap frames or slow-tail samples are removed. The minimum 600-frame window is an engineering gate, not a statistically guaranteed population-tail bound. Qualify both MOT17-02 and MOT17-04 separately and use at least five fresh-process repetitions before a release claim; retain every run and do not pool sequences to hide a failure. A passing single run is labeled a run-level result only. No confidence interval or hard real-time guarantee follows from empirical p99.

Accuracy is a separate release prerequisite: evaluate causal and deadline-valid predictions with full-source TrackEval HOTA/MOTA/IDF1/IDSW against a matched full-inference baseline. An empty tracker may satisfy latency SLOs and still be unusable. No numeric accuracy-regression allowance has been approved; SLO PASS alone is not release approval. Log time to first track, frames with tracks, detector calls, freshness, queues, CPU and RAM alongside the decision. Power remains unmeasured unless collected independently.

## Architecture and ownership

1. The source adapter preserves original frame IDs, dimensions, nominal rate and source timestamps.
2. The main thread converts grayscale, propagates live tracks, validates/fuses available proposals, evaluates `BaseScheduler`, and attempts nonblocking admission.
3. Bounded request/result queues transport owned payloads. Detector inference and source-time association/replay execute in the worker.
4. The runtime publishes causal outputs immediately; a late detector result never rewrites an earlier output.
5. Export and TrackEval run after verified shutdown, outside the measured loop.

Policy intent is distinct from admitted and completed work. Fixed-K, Kalman covariance, adaptive confidence/motion, and every-frame `OraclePolicy` reference use identical queue, replay and covariance settings. The oracle reference requests every frame; it has no ground-truth access or theoretical optimality claim.

## ADR-001: Latest-pending replacement under backpressure

**Status:** accepted in the current runtime. **Decision:** each queue has capacity one by default. A nonblocking insertion replaces its oldest pending payload when full. The in-flight native inference is not cancelled. Release the displaced request/result's history lease, record replacement/disposal counters and preserve source IDs. Queue operations, not approximate `qsize()` snapshots, determine admission.

**Rationale:** recent observations are more useful than an arbitrarily old FIFO backlog. Bounded queues limit retained images and observation age. The main tracking loop never waits for detector capacity. This policy drops detector requests/results, not source tracking frames. If grayscale history cannot retain an admitted request safely, reject admission and continue publishing tracking output.

**Consequences:** a policy request is not an inference guarantee. Effective inference reduction must use completed calls, with admitted/replaced/started/shutdown-discarded counts reported separately. Starvation and long detector freshness are possible; capacity bounds do not prove latency bounds. Worker startup and bounded shutdown joins occur outside the frame loop; an uninterruptible native call may outlive the join timeout and must be reported as incomplete shutdown.

**Alternatives considered:** unbounded FIFO hides overload as growing staleness and memory; bounded blocking FIFO stalls the tracking thread; drop-newest protects old work at the cost of freshness; cancelling an in-flight thread/native call is unsupported safely. Process isolation permits stronger cancellation at added copy/startup cost and remains a future alternative.

## ADR-002: Source-time replay and guarded stale-detection fusion

**Status:** accepted as a translation approximation, pending full-sequence fidelity validation. Associate boxes against the request's source-frame track snapshot. Seed features only inside cropped active bounding-box ROIs, then replay through a contiguous, leased grayscale history. Prepare corrections off-thread; perform only a bounded short final alignment and commit on the state-owning thread.

Reject wrong epochs, expired or out-of-order observations, excessive target lag, missing history/support, retired identities and changed anchor revisions. Transport a matched source-box residual onto the current propagated box. A new birth requires current feature support and deduplication. Historical detector misses do not immediately erase current tracks. Retain source/arrival/target frame IDs, rejection reasons and detector freshness.

Default limits come from `RuntimeConfig`: 16 history frames /64 MiB, 12 replay frames, 2 final commit frames, 500 ms observation age, 100 ms replay budget and 3 ms commit headroom. BGR payloads and scratch allocations are additional memory. These are implementation limits, not the 33 ms SLO guarantee. A native LK call can overrun soft headroom.

**Alternatives considered:** applying stale boxes directly produces time-misaligned anchors; discarding every delayed result wastes asynchronous detections; full replay on the main thread creates latency spikes; a separate fusion worker may improve detector service but adds ownership/coordination cost; out-of-sequence Kalman smoothing is a future alternative, not implemented equivalence. Translation replay is unreliable under scale changes, occlusion and illumination failure; strict rejection favors causal consistency over accepting unsupported corrections.

## Verification and artifacts

```bash
.venv/bin/python -m scripts.verify_slos \
  --sequence data/raw/MOT17/train/MOT17-02-FRCNN \
  --model models/yolo11n.onnx --policy fixed_k --k 3 \
  --output results/slos/mot17-02-fixed3
```

The command checks idle conditions before model execution, runs the real runtime at contract cadence, and writes `runtime/{summary.json,frames.jsonl,events.jsonl,tracks.txt,deadline_tracks.txt}`, `slo_report.json`, `slo_report.md`, and source/model/contract provenance. Non-idle preflight defers execution. Existing outputs are never overwritten. For saved async outputs, use `--run-dir PATH --output NEW_REPORT_DIR`; this checks integrity and thresholds without rerunning inference. Exit codes: 0 all gates PASS; 1 measured/integrity FAIL; 2 configuration/input/deferred preflight error. Missing, truncated or malformed telemetry cannot PASS.
