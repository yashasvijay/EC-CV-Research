# Week 1 validation

2026-10-02: `python -m pytest -q`: **6 passed**.

Checks cover translated textured patches, ID preservation after detector update,
texture-loss refresh, maximum detection interval, always-detect baseline,
missed-track suppression, JSON/resource logging, and real ONNX Runtime decoding
and NMS using a synthetic constant-output ONNX graph. The ONNX test also runs
both CLI modes over five synthetic PNG frames, validates complete-run metadata,
inference fractions (baseline 1.0, adaptive 0.2), and stable MOT IDs.

This validates implementation plumbing, not learned YOLO predictions or real
accuracy/performance. No YOLO checkpoint, MOT17/VisDrone dataset, MOTA/HOTA
results, or device energy measurements were available for this run. TrackEval
commands in README have not been executed. Resolved dependencies are recorded
in environment.txt.
