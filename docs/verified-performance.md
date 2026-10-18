# Verified performance and qualification scope

Updated 2026-10-18. The authoritative measurements are the immutable saved run and its frame ledger; headline numbers must retain the workload and qualification below.

## Async timing result

Run: `results/slos/mot17-02-fixed3/`, MOT17-02-FRCNN frames 1–600, 1920×1080, 30 Hz paced replay, FixedKPolicy(3), one OpenCV/ORT thread, ten detector warmups, CPUExecutionProvider, Apple Silicon/macOS. Package versions recorded for this run: NumPy 2.5.4, OpenCV headless 4.14.0.94, ONNX Runtime 1.31.0, psutil 7.2.2. These differ from earlier synchronous profiling environments.

| SLI | Exact measured value | Display | Contract |
|---|---:|---:|---|
| Throughput, source-capped | 30.0 FPS | 30.0 FPS | ≥30 FPS |
| Capture-to-output p95 | 5.5648979 ms | 5.56 ms | <33 ms |
| Capture-to-output p99 | 5.60960342 ms | 5.61 ms | <33 ms |
| Deadline-miss fraction | 0.0016666667 | 0.17% | <0.01 (1%) |

All timing/integrity gates passed. The runtime published a complete 600-frame ledger, but **every frame contained zero tracks**. Both `tracks.txt` and `deadline_tracks.txt` are empty. Detector freshness is null. This is a valid frame-publication timing measurement and an unsuccessful demonstration of useful tracking. It must not be marketed as tracking-fidelity validation or a 30 FPS populated-tracker benchmark.

The worker completed 123 detections and published 122 proposals. All 122 were rejected: target lag 88, replay budget 28, deadline headroom five, commit budget one. Four candidate births lacked support; these worker counts do not independently establish a root cause for all failed fusions. There were 200 scheduled/admitted requests and 77 superseded pending requests. The immediate follow-up is to diagnose replay/commit acceptance and bootstrap, then repeat latency and TrackEval with populated tracks. Do not silently relax stale-data or deadline safeguards to obtain a favorable number.

Detector mean service time was 82.82 ms and mean queue wait 48.52 ms. Process CPU averaged 110.67%, sampled peak RSS 328.94 MiB. The verifier caps throughput at nominal source cadence; raw runtime throughput was 30.0419 FPS. Input is paced file replay, not a physical camera. Decode/backlog are included relative to nominal release; exposure, capture transport and rendering are excluded. There is one measured run, no thermal control, no power measurement and no evaluated tracking accuracy.

## Feature-selection result

The original 50-frame cProfile/timer attribution measured 114.6196 ms/frame in native feature selection; the ROI profile measured 1.3052 ms/frame. The relative reduction from the unrounded measurements is about 98.86%. The paired, mask/offset-inclusive comparison measured 147.7969→2.0976 ms/frame, a 98.58% reduction (~98.5%) across three repeats with substantial variation. Do not attach the paired percentage to the native-call timing boundary as an exact calculation. [Full evidence](../reports/roi-features/results.md).

The ROI synchronous profile measured 8.06 FPS and 143.44 ms pipeline p99 while tracking active objects. It cannot be causally compared to the 5.61 ms async p99 as an end-to-end tracking speedup: the latter had no live tracks.

## Historical grid and publication provenance

The completed 48-configuration grid and Pareto figure use the earlier synchronous tracker. The figure plots HOTA against inference reduction, not async latency. Policy-wide accuracy/latency results after ROI optimization remain pending.

`DESIGN.at-run.md` in the saved SLO directory preserves the exact specification bytes matching the original report's `design_sha256`. The current design text adds this qualification without rewriting the historical report. The report also records model/source hashes, package versions, runtime input hashes, preflight samples and the verifier hash. The publication recheck recomputes timing SLIs from saved timestamps; it performs no new inference. Historical artifacts and their original status are preserved.

## Verification commands

Use the quick start in [README](../README.md). For a new run choose a new output directory. `python -m scripts.verify_slos --run-dir results/slos/mot17-02-fixed3/runtime --output NEW_DIRECTORY` verifies saved timing only. Full causal/deadline TrackEval uses `scripts.run_async_evaluation` with a complete MOT17 sequence; the four-policy sweep uses `scripts.ablation_policies`. See [engineering commands](engineering.md).
