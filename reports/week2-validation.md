# Phase 1, Week 2 validation

Date: 2026-10-12

Command: `.venv/bin/python -m pytest -q`

Result: **26 passed in 26.40 seconds** (6 Week 1 checks, 20 Week 2 checks including parametrized cases).

## Verified behavior

- MOT17 metadata, frame order, timestamps, lazy decoding, missing/duplicate frames,
  and decoded dimensions. Inference does not require or consume ground truth.
- Eight-state Kalman constant-velocity tracking, prediction across skipped
  detector frames, aspect-ratio velocity, covariance symmetry/PSD, positive boxes.
- Hungarian association with reordered detections, class gating, ID persistence,
  expiry, and separate semantics for skipped inference versus empty detections.
- Strict schedules for K = 1, 2, 5, 10, 15.
- Actual TrackEval 1.3.0 evaluation on generated 16-frame MOT17 fixtures:
  - Perfect predictions: HOTA 100, MOTA 100, IDF1 100, IDSW 0.
  - One mid-sequence identity switch: HOTA approximately 70.7107,
    MOTA 93.75, IDF1 50, IDSW 1.
  - Empty predictions: HOTA 0, MOTA 0, IDF1 0, IDSW 0.
  - MOT17 distractor preprocessing removes matching distractor predictions.
- Incomplete runs are rejected without changing summary.json; duplicate
  frame/track IDs are rejected. Existing throughput fields survive evaluation.
- Full ablation CLI with real ONNX Runtime and a constant-output synthetic model:
  all five K values run in fresh processes, evaluate in a separate environment,
  save complete manifests and CSV/JSON tradeoff rows, and report the exact
  finite-sequence inference reduction. All five synthetic HOTA values are 100.
- `.venv-eval/bin/python -m pip check`: no broken requirements.

## Scope

These are deterministic synthetic correctness/integration checks, not MOT17
accuracy, edge throughput, or energy measurements. Real MOT17 frames and ground
truth were absent from data/raw during this validation. The existing YOLO model
is available, but the full ablation test intentionally uses a known-output ONNX
fixture so expected coordinates and metrics are independently checkable.

The implementation is an eight-state SORT variant, with aspect-ratio velocity,
not an exact seven-state reproduction of the original SORT. The same tracker
configuration is held constant across all K values.

Resolved package versions are recorded separately in
week2-benchmark-environment.txt and week2-evaluation-environment.txt. Evaluation
installation did not change the existing benchmark environment. The integration
sweep exposed and fixed Python symlink resolution bypassing the evaluation venv.
