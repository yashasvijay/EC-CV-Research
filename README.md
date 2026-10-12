# Adaptive Neural Gating for Real-Time Computer Vision on Edge Hardware

Phase 1, Weeks 1–2: CPU ONNX detection with Lucas–Kanade adaptive gating,
an eight-state SORT variant, strict MOT17 loading, automated TrackEval metrics,
and fixed-interval inference ablations.


This project studies whether tracking confidence and motion signals can reduce
neural detector inference while preserving multi-object tracking accuracy.
The current implementation is a Python CPU prototype, with dedicated native
extension directories reserved for later profiling and optimization.

## Current evidence

On the full 600-frame **MOT17-02-FRCNN** training sequence, a single sequential
baseline/adaptive experiment using YOLO11n ONNX and one CPU thread measured:

| Metric | Always detect + LK | Adaptive + LK |
|---|---:|---:|
| Throughput | 3.84 FPS | 4.21 FPS |
| Detector calls | 600 | 484 |
| HOTA | 25.82 | 25.94 |
| MOTA | 20.62 | 20.83 |
| IDF1 | 25.48 | 25.85 |
| ID switches | 136 | 112 |
| Mean process CPU | 96.22% | 92.23% |
| Sampled peak RSS | 258.50 MiB | 275.59 MiB |
| p95 pipeline latency | 325.35 ms | 400.62 ms |

Adaptive scheduling reduced inference by **19.3%** and increased throughput by
**9.6%** in this run. RAM and tail latency increased. These results describe
one sequence and one timing repetition on a macOS ARM CPU host; they do not
establish real-time 30 FPS edge deployment, energy savings, or a general gain
across datasets. The project title describes the research objective.
HOTA/MOTA/IDF1 use percentage units; CPU 100% represents one logical core.

- [Architecture and tracker semantics](docs/architecture.md)
- [Reproducing the real experiment](docs/reproducibility.md)
- [Experiment protocol and full results](docs/results-mot17-02.md)
- [Machine-readable measured results](docs/results-mot17-02.json)

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── requirements-eval.txt            # isolated TrackEval dependencies
├── .gitignore
├── docs/                           # architecture, reproduction and results
├── configs/phase1.json              # protocol reference
├── src/
│   ├── __init__.py
│   ├── detector.py                  # YOLOv8/11 ONNX + letterbox + class-aware NMS
│   ├── datasets/mot17.py            # seqinfo.ini and strict frame loading
│   ├── sort_tracker.py              # eight-state Kalman + Hungarian matching
│   ├── tracker.py                   # LK + forward/backward checks + Hungarian IDs
│   ├── pipeline.py                  # gating and stage timers
│   └── benchmark.py                 # psutil, JSONL and aggregate JSON
├── scripts/
│   ├── __init__.py
│   ├── run_phase1.py                # one sequence and one mode per process
│   ├── evaluate_tracking.py         # TrackEval -> summary.json
│   ├── ablation_fixed_interval.py    # K=1,2,5,10,15, repeated runs
│   └── compare.py                   # repeated-run summaries
├── tests/
│   ├── test_phase1.py                # synthetic motion, gating, ONNX and logging
│   └── test_phase1_week2.py          # dataset, Kalman, actual TrackEval and sweep
├── csrc/
│   ├── README.md
│   ├── include/
│   └── src/
├── data/
│   ├── raw/                        # licensed MOT17/VisDrone inputs
│   └── processed/                  # future canonical dataset adapters
├── models/
├── results/
├── reports/
└── notebooks/
```

Planned additions (not implemented): `src/datasets/visdrone.py`,
`scripts/grid_search.py`, `scripts/profile.py`, `scripts/quantize.py`,
`csrc/CMakeLists.txt`, `csrc/bindings.cpp`, `csrc/include/gating.hpp`,
`csrc/src/gating.cpp`, `reports/ablations/`, and `reports/paper/`.

## Setup (Python 3.11+; validated locally with Python 3.13)

```bash
cd EC-CV-Research
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest onnx
python3 -m venv .venv-eval
.venv-eval/bin/python -m pip install -r requirements-eval.txt
python -m pytest -q
```

Export a **detection** checkpoint in a separate environment so PyTorch and GUI
OpenCV dependencies do not contaminate the benchmark environment:

```bash
python3 -m venv /tmp/eccv-export-env
/tmp/eccv-export-env/bin/python -m pip install ultralytics onnx
/tmp/eccv-export-env/bin/python -c 'from ultralytics import YOLO; YOLO("yolo11n.pt").export(format="onnx", imgsz=640, batch=1, dynamic=False, half=False, nms=False, simplify=False, opset=17, device="cpu")'
mv yolo11n.onnx models/yolo11n.onnx
python -m pip freeze > reports/environment.txt
```

`yolov8n.pt` is also supported with the same export options. The wrapper expects
float32 NCHW input and raw `[1, 4+num_classes, N]` output. It rejects embedded-NMS,
segmentation, pose, YOLOv5 and other incompatible outputs. Default COCO class 0
means person. A custom model needs its own `--num-classes` and `--classes`.

## Baseline versus adaptive

Download a MOT17 training sequence from https://motchallenge.net/data/MOT17/
and place it under `data/raw/MOT17/train/`. Preserve `seqinfo.ini`, `gt/gt.txt`,
and `img1/`. Use one detector suffix per physical sequence to avoid counting
MOT17's duplicated image sequences three times. This pipeline runs its own
private detector and does not consume MOT17 public detections.

```bash
source .venv/bin/activate
SEQ=data/raw/MOT17/train/MOT17-02-FRCNN/img1
python -m scripts.run_phase1 --model models/yolo11n.onnx --source "$SEQ" --mode baseline --output results/baseline-1 --threads 1
python -m scripts.run_phase1 --model models/yolo11n.onnx --source "$SEQ" --mode adaptive --output results/adaptive-1 --threads 1 --max-interval 10 --min-confidence 0.6 --max-displacement 0.15
python -m scripts.compare results/baseline-1 results/adaptive-1
```

A video path can replace `--source "$SEQ"`. Image sequences must use zero-padded
filenames and contain every frame; exported frame IDs are one-based sequence
positions. Use full sequences for accuracy evaluation. `--max-frames 300` is
useful only for a smoke test. Outputs refuse to overwrite an existing directory.

For performance, run at least five repetitions per mode in fresh processes;
alternate baseline/adaptive order, use distinct output directories, keep the
same frame range, threads, resolution and weights, and record temperature,
power mode, hardware, OS and background load. Report run-level mean and SD
with per-run latency quantiles; frames are correlated and are not independent
experimental replicates. Reserve held-out sequences for evaluation; tune gates
only on development sequences. The CLI records flags, packages and model hash.

## Week 2: MOT17 and eight-state SORT

The dataset adapter accepts `data/raw/MOT17/train/<sequence>/` with `seqinfo.ini`,
`img1/` (or the declared `imDir`) and optional `gt/gt.txt`. It checks numeric
frame IDs 1 through `seqLength`, missing/duplicate/extra frames, and decoded
image dimensions. It loads images lazily and exposes FPS, dimensions, original
IDs, and timestamps `(frame_id - 1) / frame_rate`. Ground truth is required only
by evaluation, never by tracking. `seqinfo.ini` does not supply calibrated
camera intrinsics or extrinsics; those are not invented.

```python
from src.datasets.mot17 import MOT17Dataset

for sequence in MOT17Dataset("data/raw/MOT17", split="train"):
    print(sequence.metadata)
    for frame in sequence:
        image, frame_id = frame.image, frame.frame_id
```

The dataset collection defaults to the FRCNN suffix to avoid counting the same
video three times; `detector_suffix="DPM"` or `"SDP"` selects another copy.
The runner accepts a sequence directory or its `img1/` path and records metadata.

The original [SORT implementation](https://github.com/abewley/sort) uses seven
states. Our requested **eight-state SORT variant** uses
`[cx, cy, area, aspect_ratio, vx, vy, varea, vaspect_ratio]`, constant velocity,
a Joseph-form covariance update, and class-aware Hungarian IoU matching. This
is an explicit variant, not an exact reproduction of the published baseline.
Kalman time is measured in video frames, including skipped inference frames.

`SortTracker.step(detections)` accepts the detector's `(N,6)` array
`[x1,y1,x2,y2,score,class]` and returns objects exposing `id`, `box`, and `score`.
Call it exactly once per video frame. `step(None)` means inference was skipped
and emits confirmed predictions. An empty `(0,6)` array means the detector ran
but found no objects; unmatched tracks are suppressed until reacquired.
`max_age=30` counts frames since the last match; `min_hits=1` confirms tracks at
birth. Higher `min_hits` counts matched detector observations, so confirmation
is delayed more for larger K. Neither this tracker nor LK performs re-ID.

```bash
source .venv/bin/activate
SEQ=data/raw/MOT17/train/MOT17-02-FRCNN
python -m scripts.run_phase1 --model models/yolo11n.onnx --source "$SEQ" --tracker sort --mode baseline --output results/sort-baseline --threads 1
python -m scripts.run_phase1 --model models/yolo11n.onnx --source "$SEQ" --tracker sort --mode fixed --max-interval 5 --output results/sort-k5 --threads 1
```

Existing LK baseline/adaptive commands retain their behavior. LK also supports
`--mode fixed` for controlled future comparisons. SORT currently supports
baseline and fixed scheduling; `--tracker sort --mode adaptive` is rejected
because LK confidence is not a Kalman uncertainty estimate.

## Automated accuracy evaluation

Install `requirements-eval.txt` in `.venv-eval` as shown above. The pinned
[TrackEval package](https://pypi.org/project/trackeval/1.3.0/) supports modern
NumPy; the previous NumPy 1.23 workaround for older source is no longer needed.
Keep evaluation isolated so its plotting and OpenCV packages do not change
benchmark timings. `EVAL_PYTHON=/path/to/python` overrides the evaluation
interpreter used by integration tests.

```bash
.venv-eval/bin/python -m scripts.evaluate_tracking --sequence "$SEQ" --run-dir results/sort-baseline
.venv-eval/bin/python -m scripts.evaluate_tracking --sequence "$SEQ" --run-dir results/sort-k5
```

The evaluator stages `tracks.txt` into TrackEval's expected layout, enables
MOT17 pedestrian/distractor preprocessing, and runs HOTA, CLEAR, and Identity
metrics. It atomically adds **HOTA, MOTA, IDF1, IDSW** to the existing
`summary.json`, preserving latency/FPS/CPU/RAM measurements. HOTA is the mean
over localization thresholds. HOTA/MOTA/IDF1 are percentage points (MOTA may
be negative); IDSW is an integer count. It saves evaluator version, GT and
prediction SHA-256 hashes, timestamp, and metric units. Detailed TrackEval
reports go under the run's `trackeval/` directory. Evaluation time is excluded
from measured throughput. Exceptions leave the summary unchanged.

Evaluation requires a complete full-sequence run, a contiguous `frames.jsonl`
ledger, matching source metadata, and COCO person-only `--classes 0`. Missing
prediction rows represent empty frames and are valid, including an entirely
empty file. Duplicate frame/ID pairs, invalid boxes, out-of-range frame IDs,
and truncated runs are rejected. Exported MOT coordinates are one-based;
internal boxes are zero-based. Existing Week 1 full-sequence runs with the same
source, frame ledger, and class metadata can also be evaluated.

## Fixed-interval ablation

```bash
python -m scripts.ablation_fixed_interval \
  --sequence "$SEQ" --model models/yolo11n.onnx \
  --output results/fixed-interval-week2 --repeats 5 --threads 1 \
  --eval-python .venv-eval/bin/python
```

Default K values are **1, 2, 5, 10, 15**. Detection runs exactly on frames
`1, 1+K, 1+2K, ...`, even when there are no tracks; no failure-triggered extra
inferences occur. All other tracker parameters remain constant across K.
`max_age` must be at least `max(K)-1` so skipped frames do not cause forced
expiry. K=1 is the always-detect control using the same tracker.

Each repetition shuffles K order with a recorded seed and runs each benchmark
in a fresh process. Evaluation also runs in a separate process. Each run logs
per-frame latency, FPS, CPU/RAM, detections, and all tracking metrics. Results:

- `manifest.json`: configuration, planned order, completed jobs, failure status.
- `runs.json`: individual measurements and their source run paths.
- `tradeoff.csv` and `tradeoff.json`: per-K HOTA, MOTA, IDF1, IDSW, inference
  reduction, FPS, CPU and RAM, with run-level mean and sample SD. SD is null
  for one repetition. These files describe only completed jobs; always check
  the manifest for full sweep completion.
- `kXX-rYY/`: benchmark outputs plus evaluation reports; sibling `.log` files
  retain subprocess output for diagnosis.

Inference reduction is `100 * (1 - detector_calls / frames)`, relative to
K=1. Finite-sequence rounding is included, so it need not equal `100*(1-1/K)`.
FPS is end-to-end measured throughput, not the mean of reciprocal latencies.
Repeated runs on one sequence measure timing variation, not generalization;
run held-out sequences separately and use TrackEval's sequence aggregation
for corpus scores rather than naively averaging sequence metrics.

VisDrone requires its own annotation/class adapter and protocol and remains
outside this Week 2 implementation. Synthetic fixtures verify implementation
correctness; the separately documented MOT17-02 experiment provides real-sequence
accuracy measurements. Edge energy remains unmeasured.

## Instrumentation and limitations

- `frames.jsonl`: per-frame pipeline/flow/detector/association/decode latency,
  trigger, pre-detection confidence, normalized displacement, tracks, RSS, CPU.
- `summary.json`: throughput (frames / measured wall time), pipeline-only FPS,
  mean/p50/p95/p99 latency, process CPU seconds and mean usage, sampled peak RSS,
  inference fraction and run metadata. Failed runs are marked partial.
- Pipeline timing excludes decode and export. Wall throughput includes input,
  MOT/JSONL output and instrumentation, but excludes model loading and detector
  warmup. Tracker initialization remains measured. `frame_work_ms` ends before
  resource sampling and JSONL writing. JSONL streams; latency/RSS arrays use
  O(number of frames) memory for exact quantiles.
- CPU is process usage: 100% = one logical core; multithreaded use can exceed
  100%. Nonblocking samples are refreshed no faster than 100 ms; null means
  no sample yet, and `cpu_sample_updated` distinguishes held values. Aggregate
  CPU uses CPU-time / wall-time, not an average of noisy per-frame percentages.
- RSS is sampled process resident memory, not allocator peak or whole-device RAM.
  Power remains null: measure watts/joules with the target board's sensor or an
  external meter. CPU utilization cannot establish energy savings.
- For `--tracker lk`, both baseline/adaptive modes run LK and identical
  association. For `--tracker sort`, baseline/fixed modes share the eight-state
  Kalman tracker. In each pair, the scheduling comparison holds the tracker fixed.
- Confidence multiplies point survival, forward/backward consistency and motion
  coherence since the last detection. It is a heuristic, not a probability.
  Bounding boxes use median translation only: scale change, rotation, coherent
  drift and occlusion can defeat the gate. Detection has no re-identification.
- In LK adaptive mode, low-texture/no-track scenes force inference. Refresh occurs within the maximum
  interval, limiting but not eliminating delayed discovery of new objects.
  Association retains missing IDs for two detector updates, suppressing their
  boxes until matched again. ID switches remain possible.

Week 2 code supports MOT17 loading, an eight-state SORT variant, automatic
metrics and fixed-K ablations. The first real MOT17-02 experiment is documented
above; repeated timing trials, multi-sequence validation, and the real fixed-K
sweep remain to be completed. Weeks 3–4 add Farneback and
validation-only Pareto search; Weeks 5–6 add profiled native work/INT8; Weeks 7–8
freeze experiments and report accuracy-throughput-energy tradeoffs.

References: [Ultralytics export](https://docs.ultralytics.com/modes/export/),
[ONNX Runtime API](https://onnxruntime.ai/docs/api/python/api_summary.html),
[OpenCV LK](https://docs.opencv.org/4.7.0/dc/d6b/group__video__track.html),
[psutil](https://psutil.readthedocs.io/stable/),
[TrackEval](https://github.com/JonathonLuiten/TrackEval).
