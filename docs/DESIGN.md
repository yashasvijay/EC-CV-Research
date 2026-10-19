# Streaming tracking system specification

Status: populated 600-frame replay verified; current contract requires sustained coverage and useful background fusion. Revision: 3, 2026-10-18.

## Scope and service boundary

The runtime accepts monotonically numbered BGR frames, propagates sparse Lucas–Kanade (LK) features, consults a scheduler, and publishes immutable tracks. YOLO11n ONNX inference and historical correction preparation run in one background worker. The main thread owns live track state. The deployment target is Apple Silicon ARM64, 1920×1080, 30 source frames/s, COCO pedestrian detections, one configured OpenCV/ORT thread per component. Python thread count does not imply a single process-wide CPU core.

The populated MOT17-02 FixedKPolicy(3) run maintained tracks on 600/600 frames at **30.0 FPS, 26.21 ms p95, 29.74 ms p99 and 0.67% deadline misses**. Its useful fusion ratio was **70.6% (125/177 background runs)**. [Original report](../results/slos/mot17-02-fixed3-populated/slo_report.json) and [current-contract recheck](../results/slos/mot17-02-fixed3-populated/verification-v3/slo_report.json) retain the exact values and raw-ledger provenance. This is populated paced replay, not HOTA/MOTA or physical-camera validation. The earlier empty-output 5.61 ms run is historical and fails current tracking gates. [Measurement scope](verified-performance.md).

## SLO contract

The following JSON block is the sole executable threshold definition. `scripts/verify_slos.py` reads it directly; changes require reviewing this specification and the verification tests together.

<!-- SLO_CONTRACT_START -->
```json
{
  "version": 4,
  "camera_fps": 30.0,
  "throughput_min_fps": 30.0,
  "p95_e2e_max_ms": 33.0,
  "p99_e2e_max_ms": 33.0,
  "p95_detection_age_max_ms": 300.0,
  "deadline_ms": 33.0,
  "deadline_miss_max_fraction": 0.01,
  "minimum_frames": 600,
  "minimum_useful_fusion_ratio": 0.1,
  "minimum_track_frame_fraction": 0.95
}
```
<!-- SLO_CONTRACT_END -->

| SLI | Acceptance rule | Measurement |
|---|---|---|
| Throughput | ≥30 FPS | Published source frames / max(observed wall duration, N/30); caps startup-biased estimates at source rate |
| End-to-end p95 | <33 ms | Linear NumPy percentile of nominal capture-to-publication latency |
| End-to-end p99 | <33 ms | Same capture-to-publication population, including late frames |
| Track detection-age p95 | <300 ms | Per-frame maximum active-track source anchor age plus capture-to-output latency; missing samples fail |
| Deadline misses | <1% | Missing source frames plus publications with latency **≥33 ms**, divided by all expected frames |
| Track-frame fraction | ≥95% | Frames publishing ≥1 live track / all expected source frames |
| Useful fusion ratio | ≥10% | Completed background runs accepting ≥1 correction / completed background runs; excludes bootstrap |
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

## ADR-001: Idle-slot admission under backpressure

**Status:** accepted; supersedes latest-pending admission. **Decision:** reserve the single worker slot atomically before enqueueing. Both queued requests and pending results make the worker unavailable. Busy policy triggers are deferred, with no payload copy or queue replacement. The defensive latest-pending mailbox retains disposal accounting, but normal producer admission does not supersede requests. In-flight native inference cannot be safely cancelled.

**Rationale:** idle admission avoids redundant snapshots and makes detector ownership explicit. Bounded queues limit retained images and observations. Cold-start inference is allowed only on an idle detector, with cooldown; an empty tracker adopts a valid in-flight result through normal fusion. If history cannot retain a request, release the reservation and continue publishing tracking output.

**Consequences:** a policy request is not an inference guarantee. Effective inference reduction must use completed calls, with admitted/replaced/started/shutdown-discarded counts reported separately. Starvation and long detector freshness are possible; capacity bounds do not prove latency bounds. Worker startup and bounded shutdown joins occur outside the frame loop; an uninterruptible native call may outlive the join timeout and must be reported as incomplete shutdown.

**Alternatives considered:** unbounded FIFO hides overload as growing staleness and memory; bounded blocking FIFO stalls the tracking thread; drop-newest protects old work at the cost of freshness; cancelling an in-flight thread/native call is unsupported safely. Process isolation permits stronger cancellation at added copy/startup cost and remains a future alternative.

## ADR-002: Source-time replay and guarded stale-detection fusion

**Status:** accepted as a translation approximation, pending full-sequence fidelity validation. Associate boxes against the request's source-frame track snapshot. For unmatched detections, seed features inside cropped bounding-box ROIs and replay through contiguous, leased grayscale history. Matched proposals skip historical LK, transport the source residual onto the current box, and reseed on the current frame. Prepare corrections off-thread; perform only a bounded short final alignment and commit on the state-owning thread.

Reject wrong epochs, expired or out-of-order observations, excessive birth target lag, missing birth history/support, retired identities and changed anchor revisions. Transport a matched source-box residual onto the current propagated box. A new birth requires current feature support and deduplication. Historical detector misses do not immediately erase current tracks. Retain source/arrival/target frame IDs, rejection reasons and detector freshness.

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

## Current verification record

The populated result is recorded with raw logs in `results/slos/mot17-02-fixed3-populated/`. It passes the version-3 timing, ≥95% populated-frame and useful-fusion-run checks; it has not passed version 4’s per-track freshness gate. The prior empty-output report remains archived and fails those tracking gates. SLO acceptance does not evaluate identity or box accuracy. The at-run version-2 design is preserved separately from this revision.

## Tracking-validity gate (contract version 3)

A run must publish live tracks on at least 95% of expected source frames and apply at least one correction in at least 10% of completed background detector runs. `useful_fusion_ratio = useful_fusion_runs / worker_counts.completed`; a run counts once regardless of how many boxes were accepted. Zero completed runs gives ratio zero. Bootstrap, empty and rejected background results do not enter the numerator. Saved version-2 artifacts used `fused_with_tracks` for the same run-level count; the verifier accepts that alias only when it agrees with the frame ledger. These are engineering floors, not accuracy guarantees. Version 3 replaces version 2's insufficient `track_frames > 0` gate.

Synchronous bootstrap is enabled by default when no live tracks remain, with `bootstrap_min_interval_frames=15` between attempts, including empty/failed-to-seed attempts. Between attempts the main thread keeps propagating/publishing and uses the background scheduler; it does not synchronously re-infer every frame. Cooldown suppression and remaining-frame counters are exposed. A value of one explicitly restores per-frame retries for diagnostic tests. It uses the same detector under an inference lock, updates supported ROI tracks at the current source frame, and includes lock wait, inference and reseeding in `bootstrap_ms`, frame-work and capture-to-output latency. It may miss the deadline; bootstrap is not a hidden warmup. Source frame IDs remain one-based. `synchronous_bootstrap=False` is an explicit worker-only diagnostic mode, not the verifier default.

Main-thread propagation, worker replay and final correction alignment batch all track points into one forward/backward LK pair per source step. The Python OpenCV binding rejects prebuilt pyramid tuples, so native LK constructs shared pyramids internally once per batched direction. No redundant `buildOpticalFlowPyramid` prepass is performed. `effective_replay_budget_ms` is capped by the commit-lag window minus a pickup reserve (default 33.33 ms rather than 100 ms); this is soft admission, not a guarantee that a single native call completes within budget. Age, source-order, epoch and target-lag rejection remain enforced.

### Idle admission and correction isolation (contract v4)

The producer reserves an idle detector slot before enqueueing. Pending results also block new admission until consumed. Empty-track recovery adopts the next valid background result; synchronous bootstrap is permitted only when the worker is idle and never waits on its inference lock. Bootstrap cooldown remains in force.

Only unmatched detections replay historical optical flow. Matched corrections retain revision checks, apply the source residual to the live box, and reseed features inside that box on the current frame. A failed birth replay does not invalidate independent matched corrections.

The v4 freshness gate requires p95 maximum active-track detection age below 300 ms. Each sample combines source-time age since the track's accepted anchor with capture-to-output latency. Empty-track frames are excluded from age quantiles and remain subject to the 95% track coverage gate. Old exports lacking this field cannot pass v4. Historical v3 results do not establish compliance with this expanded contract.

After verified shutdown, packet accounting must satisfy admitted = started + superseded and published = fused + rejected. Superseded includes every admitted request cancelled before inference starts, including expiration and shutdown. Published results discarded during shutdown or mailbox replacement count as rejected. Live balances may include outstanding packets. Correction acceptance uses accepted correction proposals divided by consumed proposals; useful fusion uses background runs with at least one accepted correction divided by completed background runs. IDF1 and IDSW remain null until ground-truth evaluation, independently of runtime SLOs.
