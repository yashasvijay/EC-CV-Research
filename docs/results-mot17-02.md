# MOT17-02-FRCNN: initial real-sequence experiment

> Current scope (2026-10-18): Historical pre-ROI synchronous accuracy results. These are not the 30 FPS empty-output async timing run. See [verified performance](verified-performance.md).

## Protocol

The experiment processed all 600 frames at 1920x1080 from the MOT17 training
sequence, whose recorded frame rate is 30 FPS. YOLO11n was exported as batch-one
FP32 ONNX with 640x640 input and used COCO person class 0. ONNX Runtime used
CPUExecutionProvider, with one inference thread and one OpenCV thread.

Both modes used the same Lucas–Kanade tracker and class-aware Hungarian IoU
association. Baseline detected every frame. Adaptive used maximum interval 10,
minimum confidence 0.6, and displacement threshold 0.15. Detector warmup was
10 executions before each measured run. Baseline and adaptive ran sequentially,
with one timing repetition each, on macOS ARM64 with eight logical CPU cores.

The eight-state SORT variant was not used in this comparison. Scores are for
this private detector protocol; the sequence's FRCNN suffix identifies the
image copy and does not mean MOT17 public FRCNN detections were used.

Offline TrackEval 1.3.0 evaluated pedestrian tracks with MOT17 distractor
preprocessing. HOTA was averaged over localization thresholds. Ground truth
was never passed into inference or scheduling. Model/package/provenance hashes
and numerical results are preserved in [the result JSON](results-mot17-02.json).

## Results

| Metric | Always detect + LK | Adaptive + LK |
|---|---:|---:|
| Frames processed | 600 | 600 |
| Throughput | 3.8436 FPS | 4.2110 FPS |
| Detector calls | 600 | 484 |
| Inference reduction | 0% | 19.3333% |
| Mean pipeline latency | 252.56 ms | 230.31 ms |
| p95 pipeline latency | 325.35 ms | 400.62 ms |
| p99 pipeline latency | 446.63 ms | 611.07 ms |
| Mean process CPU | 96.22% | 92.23% |
| Sampled peak RSS | 258.50 MiB | 275.59 MiB |
| HOTA | 25.8232 | 25.9405 |
| MOTA | 20.6232 | 20.8277 |
| IDF1 | 25.4788 | 25.8475 |
| ID switches | 136 | 112 |

Throughput increased by 9.5575% in this timing repetition. Adaptive skipped
116 detector executions. Of 484 detector calls, the first was triggered by no
tracks and the remaining 483 by low confidence. No maximum-interval or
displacement trigger was recorded. This suggests examining confidence behavior
on crowded scenes before choosing new thresholds; it does not establish an
optimal gate.

Throughput includes frame decoding and output instrumentation. Pipeline latency
excludes decoding/export. CPU 100% represents one logical core; RSS is sampled
process resident memory. The adaptive mean latency improved while p95/p99
latencies and peak RSS increased. Tracking metrics use percentage points,
except ID switches, which are counts.

## Interpretation and limits

The experiment supports a modest reduction in detector calls with similar
tracking scores on this sequence. The absolute tracking scores leave substantial
room for improvement. A single sequence and one timing repetition do not
establish statistical significance, generalization, or deployment readiness.
No power sensor or external meter was used, so energy savings are unmeasured.
CPU model, thermal state, and background load were not separately recorded;
future timing trials should capture these along with a balanced run order.

The next validation steps are repeated timing trials, additional sequences,
a real fixed-K ablation against the same tracker, and held-out evaluation after
development-only parameter tuning. Real-time 30 FPS edge operation remains a
research target. The 26 automated tests reported separately validate correctness
and synthetic integration, rather than providing additional real-sequence trials.
