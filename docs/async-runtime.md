# Asynchronous LK tracking runtime

> Current scope (2026-10-18): The populated MOT17-02 replay measured 30.0 FPS / 29.74 ms p99 with tracks on all 600 frames and useful fusion in 125/177 background runs. Accuracy and repeated-run qualification are separate. See [verified performance](verified-performance.md).

`src/async_pipeline.py` now implements `AsyncTrackerRuntime`. Helper modules in `src/runtime/` separate packet contracts, bounded grayscale history, queue transport, motion replay, and `DetectionWorker`. Synchronous controls remain available for historical comparisons. The initial 48-configuration grid is complete; source hashes identify each experiment version.

## Minimal usage

Run in a separate process after completing the existing timing experiment:

```python
import cv2
from src.async_pipeline import AsyncTrackerRuntime, RuntimeConfig
from src.datasets.mot17 import MOT17Sequence
from src.detector import YOLODetector
from src.schedulers import FixedKPolicy

cv2.setNumThreads(1)
sequence = MOT17Sequence("data/raw/MOT17/train/MOT17-02-FRCNN")
runtime = AsyncTrackerRuntime(
    detector_factory=lambda: YOLODetector("models/yolo11n.onnx", threads=1),
    scheduler=FixedKPolicy(k=3),
    config=RuntimeConfig(deadline_ms=33, frame_rate=sequence.info.frame_rate),
    metadata={"sequence": sequence.metadata, "model": "models/yolo11n.onnx"},
)
with runtime:
    for output in runtime.run(sequence.images(), paced=True):
        # Display/consume output.tracks without modifying runtime.tracker.
        pass
runtime.export("results/async-fixed3")
```

This example is an API demonstration. The current saved 30 FPS result is populated; the earlier empty-output result is archived. See [verified performance](verified-performance.md). Model initialization runs in the detector worker before readiness. For benchmark warmup, provide a factory that initializes and warms the detector before returning it. Record the model checksum, package versions, provider/thread configuration and experiment conditions in `metadata`; the runtime does not infer a checksum from an arbitrary callable.

For a camera callback or an existing frame iterator, call `start()`, then `process_frame(frame, frame_idx=..., capture_time_ns=..., source_time_ns=...)`, and finally `close()`. `step` aliases `process_frame`; `stop` aliases `close`. Capture time must use the runtime monotonic clock. Dataset/source time may use another clock but must increase consistently within the stream; source time is used for observation aging and covariance prediction, never subtracted from a capture timestamp. Omitted capture/source timestamps use host receipt time. A source-frame gap is counted; a large gap invalidates LK support. Dimension changes and restarts require a new runtime/epoch.

`run(paced=True)` uses nominal release times to simulate source cadence. Slow processing accumulates deadline misses while retaining every input frame. It is paced replay, not a camera driver or a capture-buffer management implementation. Reading/decode through the iterator can block outside `process_frame`; paced capture-to-output metrics include that delay relative to the nominal release. An external camera may drop frames before this runtime receives them; preserve its frame IDs so gaps enter the source coverage metric.

## Detection and correction ownership

The main caller owns live tracks and scheduler state. The detector factory and background calls execute in one non-daemon `DetectionWorker`; synchronous bootstrap calls use the same detector under an inference lock. Request and result transport uses bounded `queue.Queue` instances with nonblocking insertion/removal. One producer writes each queue. On overflow, the oldest pending payload is superseded; the in-flight inference is not cancelled. Queue-depth readings are observational snapshots, not synchronization predicates.

Detector payloads copy the BGR frame and source-state snapshot. Grayscale history has explicit frame and byte limits; source requests and pending corrections hold counted leases. Pinned frames cannot be evicted. If all space is pinned, the runtime continues propagating/publishing the input frame and rejects detector admission that needs unavailable history. BGR requests, track snapshots, points, native scratch storage, and the current LK image copy occupy additional bounded memory beyond the grayscale-history byte cap. No zero-copy or allocation-free claim is made.

The worker performs historical association using source-frame boxes/classes and vectorized IoU/Hungarian matching. It seeds features on source ROIs and forward/backward replays them through a leased, contiguous history path. Replay depth, points, track count, wall budget, and source age are capped. Proposal preparation shares the detector worker in this first implementation; a separate fusion worker remains a future throughput experiment.

At current-frame commit, the runtime verifies epoch, observation order, expiry, replay-target lag, current history support, and track anchor revision. Matched tracks receive the source-frame box residual transported onto their current propagated box. This uses the explicit translation approximation; it is not an exact solution under scale change or occlusion. New tracks require current feature support and are deduplicated against live tracks. Retired tracks cannot be resurrected by a historical match. Unsupported births and stale corrections are rejected with a recorded reason. Detector misses at an old source frame do not immediately erase current tracks; track retirement uses age since the last positive anchor.

Feature selection and multi-frame historical replay occur off-thread. A bounded short replay from proposal target to the current frame may occur during commit. Fusion admission uses a configurable headroom floor and recent observed p95 commit cost; work preparation checks the deadline before mutating live state. This is soft budgeting: a single OpenCV call or final bounded commit can still overrun. Actual frame/deadline telemetry records that overrun. Steady-state `process_frame` does not wait for inference. The explicit exception is synchronous bootstrap when no supported live tracks remain: detector-lock wait, inference and ROI seeding are measured in `bootstrap_ms` and the frame latency. Worker joins, file export and full historical replay remain outside the frame loop.

Optional covariance state is a parallel eight-state box Kalman model. Its process prediction uses elapsed source seconds converted into the existing filter's frame units. The worker updates at detection source time and replays covariance forward; the main thread advances it over the final commit interval. LK points do not reduce covariance without a calibrated observation-noise model. This uncertainty is a scheduling signal, not a demonstrated statistical error bound. `covariance=False` rejects `KalmanUncertaintyPolicy` rather than silently substituting another policy. Configure all compared policies with the same tracker/covariance settings.

## Lifecycle and safety

`start()` waits for factory readiness with a timeout outside the frame loop. Runtime instances are single-use. Calls that mutate live state must come from the owning thread. Worker errors are exposed through `health()`, raised on subsequent frame processing, and preserved in partial summaries. Input/scheduler/processing errors fail the runtime rather than allow a partially changed state to continue unnoticed; callers still close it in `finally` or through the context manager.

`close()` stops admission, discards queued work with accounting, signals the worker, and joins with a bounded timeout. It returns `ShutdownReport(stopped, worker_alive, worker_error)`. An ORT/native inference call already in progress cannot be forcibly interrupted by a Python event. If the report says the worker is still alive, retain the runtime and call `close()` again after it exits; do not claim shutdown succeeded or free pinned buffers. The context manager raises if shutdown remains incomplete. Strict forced-cancellation requirements need process isolation, which this implementation does not provide.

No end-of-stream draining retroactively improves prior outputs. The runtime keeps immutable output snapshots and exports them only after verified shutdown. It never updates previously published MOT rows using a later detector result.

## Telemetry and export

Each `FrameOutput` exposes immutable tracks and a read-only telemetry mapping containing work/capture-to-output latency, deadline outcome, LK/fusion/scheduler-admission costs, queue depths, detector service/preparation time, source age/freshness, trigger/admission reason, gaps, and sampled process CPU/RSS. CPU sampling runs every 250 ms rather than in every decision; telemetry includes sample status. Process CPU can exceed 100%, where 100% means one logical core.

`summary()` reports p50/p95/p99, deadline-miss rates for published and source frames, coverage, throughput, detector service/queue wait, freshness, resource summaries, worker/runtime counters and errors. Source-frame deadline rate counts missing input IDs as failures. Throughput spans worker readiness through the last published output and excludes shutdown; process CPU covers readiness through summary/shutdown and has its own duration. HOTA/MOTA/IDF1/IDSW/power remain null until independently evaluated.

Frame/event/latency history is bounded by `telemetry_frames` (default 4096). Quantiles cover the retained window and are labelled accordingly; total frame and deadline counters cover the run. `ledger_complete` and `event_ledger_complete` disclose evicted/lost telemetry. Full-sequence fidelity evaluation is invalid when its required frame/output ledger was truncated. `scripts/run_async_evaluation.py` connects complete exports to causal and deadline-valid TrackEval scoring; `scripts/ablation_policies.py` runs the four-policy sweep. Live capture-buffer management remains separate work.

After shutdown, `export(new_directory)` writes:

- `summary.json`, `frames.jsonl`, and `events.jsonl`.
- `tracks.txt`: causal published predictions in one-based MOT coordinates.
- `deadline_tracks.txt`: only predictions available by their frame deadline.

The original full source-frame ledger and ground truth must still include missing/late frames during TrackEval evaluation. Use `scripts/run_async_evaluation.py` to supply evaluator metadata, validate the complete ledger, and score both causal and deadline-valid predictions. Unit/synthetic tests are not MOT accuracy, throughput, thermal, power, or real-time validation.

## Tests

```bash
.venv/bin/python -m pytest -q tests/test_async_pipeline.py
```

Tests use event-controlled mock detectors and a fake clock for deterministic overload/deadline behavior. They cover lifecycle and timeout recovery, blocked inference while frames continue, queue replacement, bounded pinned history, motion-aligned births and matched corrections, retirement, duplicate/expired/old-epoch results, zero-texture rejection, buffer ownership, causal exports, policy integration, and worker error propagation. A small constant ONNX graph checks actual wrapper/worker interoperability without executing the real YOLO model or interfering with the running grid search.

## Cold-start and batched replay correction

Synchronous bootstrap defaults to enabled at source frame 1 or after all live tracks retire. It uses the same ready detector under an inference mutex, avoiding concurrent calls or a second model instance. Supported features establish tracks/anchors and covariance immediately; empty/textureless detections do not create invented tracks. A bootstrap frame does not also enqueue duplicate background work. Its source IDs remain one-based; the cost is measured and may miss the deadline.

`src/motion.py` batches all supported track points for main propagation, worker replay and final commit alignment. Python OpenCV does not accept prebuilt pyramid tuples; the forward/backward native calls share one internal pyramid computation per direction across the track batch. Each track retains its own validity, median displacement and reliability reduction. `src/runtime/motion.py` retains the single-track compatibility wrapper and ROI seeding adapter.

The effective preparation cap is the lesser of `replay_budget_ms` and the commit-lag time window minus a frame-pickup reserve: 33.33 ms at the default 30 Hz / two-frame commit lag. Association, seeding and historical replay enter that timer, checked before/after frame steps. Native calls remain uninterruptible and stale proposals remain rejected. Track source-age limits have not been loosened.

Telemetry separates bootstrap calls/successes, accepted background-update flags, useful fusion counts and completed background inferences. The contract-v3 verifier requires ≥95% track-frame coverage and useful fusion ratio ≥0.1; the numerator counts background runs accepting ≥1 correction, not individual boxes. Synchronous bootstrap is excluded. The preserved zero-track timing run fails those new gates. The populated run verifies timing after bootstrap/replay correction; HOTA/MOTA are not evaluated.

## Empty scenes and latency injection

`bootstrap_min_interval_frames` defaults to 15 source frames and is a positive integer. Attempt time, empty results and suppressed retries are logged; the cooldown applies after successful attempts too, bounding synchronous recovery frequency. Background scheduling remains available while a synchronous retry is suppressed. `bootstrap_attempt_fraction` measures synchronous attempts per source frame, not background requests.

The synthetic sweep is documented in [the latency-injection guide](detector-latency-sweep.md). `useful_fusion_ratio` is a run-level background acceptance metric; historical `fused_detection_ratio` names remain unchanged only in archived measurements.
