# Verified performance and qualification scope

Updated 2026-10-18. The authoritative measurements are the immutable saved run and its frame ledger; headline numbers must retain the workload and qualification below.

## Populated async result

The authoritative run is `results/slos/mot17-02-fixed3-populated/`: MOT17-02-FRCNN frames 1–600, 1920×1080, 30 Hz paced replay, FixedKPolicy(3), one configured OpenCV/ORT thread and ten detector warmups. All 600 frames published tracks. The original report and raw files are committed in full; `DESIGN.at-run.md` preserves its exact v2 specification, and `verification-v3/` rechecks the unchanged raw ledger against the stronger current contract.

| SLI | Exact measurement | Display |
|---|---:|---:|
| Source-capped throughput | 30.0 FPS | 30.0 FPS |
| Capture-to-output p95 | 26.206049999999998 ms | 26.21 ms |
| Capture-to-output p99 | 29.741468329999996 ms | 29.74 ms |
| Deadline misses | 4/600 = 0.0066666667 | 0.67% |
| Track-frame fraction | 600/600 | 100% |
| Useful fusion ratio | 125/177 = 0.7062146893 | 70.6% |

Useful fusion counts background runs with ≥1 accepted correction, not accepted boxes. Of 177 completed background runs, 125 were useful; synchronous bootstrap is excluded. First-track latency was 97.038333 ms and is included in frame/deadline measurements. CPU averaged 143.6575% of one logical core and sampled peak RSS was 341.59375 MiB. Detector mean service was 86.5707 ms. Raw throughput was 30.0217 FPS and was capped at source cadence by the verifier.

The preflight passed, but thermal conditions were uncontrolled. This is one file-replay run, not a physical-camera, repeated-run or power result. HOTA/MOTA/IDF1/IDSW are unevaluated. This measurement predates the newly added bootstrap cooldown and does not measure a speedup from that change.

The prior `mot17-02-fixed3` run had no tracks and its 5.61 ms p99 is not current tracking evidence. It remains immutable and fails current gates. The populated result resolves the stale bootstrap/timing disclaimer; accuracy and generalization remain distinct measurement gaps.

## Feature-selection result

The original 50-frame cProfile/timer attribution measured 114.6196 ms/frame in native feature selection; the ROI profile measured 1.3052 ms/frame. The relative reduction from the unrounded measurements is about 98.86%. The paired, mask/offset-inclusive comparison measured 147.7969→2.0976 ms/frame, a 98.58% reduction (~98.5%) across three repeats with substantial variation. Do not attach the paired percentage to the native-call timing boundary as an exact calculation. [Full evidence](../reports/roi-features/results.md).

The ROI synchronous profile measured 8.06 FPS and 143.44 ms pipeline p99. The populated async run has a different capture-to-output timing boundary, package versions and concurrency; report both workloads rather than claiming a controlled end-to-end speedup.

## Historical grid and publication provenance

The completed 48-configuration grid and Pareto figure use the earlier synchronous tracker. The figure plots HOTA against inference reduction, not async latency. Policy-wide accuracy/latency results after ROI optimization remain pending.

`DESIGN.at-run.md` in the saved SLO directory preserves the exact specification bytes matching the original report's `design_sha256`. The current design text adds this qualification without rewriting the historical report. The report also records model/source hashes, package versions, runtime input hashes, preflight samples and the verifier hash. The publication recheck recomputes timing SLIs from saved timestamps; it performs no new inference. Historical artifacts and their original status are preserved.

## Verification commands

Use the quick start in [README](../README.md). For a new run choose a new output directory. `python -m scripts.verify_slos --run-dir results/slos/mot17-02-fixed3/runtime --output NEW_DIRECTORY` verifies saved timing only. Full causal/deadline TrackEval uses `scripts.run_async_evaluation` with a complete MOT17 sequence; the four-policy sweep uses `scripts.ablation_policies`. See [engineering commands](engineering.md).

## Contract-v3 enforcement

Current acceptance requires `track_frame_fraction >= 0.95` and `useful_fusion_ratio >= 0.1`, recomputed from full source-frame telemetry. The historical empty run fails; the populated run passes. Missing frames enter the coverage denominator and also fail integrity. A positive track count on one frame cannot qualify a sequence. Neither useful fusion nor occupancy measures ground-truth fidelity.
