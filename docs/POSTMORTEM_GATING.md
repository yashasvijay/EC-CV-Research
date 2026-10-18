# Correction of Errors: detector gating and feature-selection cost

Status: feature-selection remediation measured; populated async timing verified after bootstrap/replay correction; accuracy qualification remains separate. Written 2026-10-18. This is a research performance incident, not a documented customer outage. Individual fault is not assigned.

## Summary

The initial MOT17-02 adaptive run invoked detection on 484 of 600 frames (80.67%), saving only 19.33% of detector calls. Throughput rose from 3.84 to 4.21 FPS. Optimizing a small native gating calculation could not address most frame time. Exclusive profiling subsequently isolated repeated full-image feature selection: 114.62 ms/frame, or 50.38% of measured synchronous pipeline time.

The original implementation used a bounding-box mask but passed the entire 1080p image to `goodFeaturesToTrack`. Restricting allowed output pixels did not restrict all corner-response computation. The remediation crops each active track ROI, applies a crop-local mask, selects corners locally, restores global coordinates and filters strict box bounds. Shared reseeding now serves synchronous and asynchronous paths.

## Impact

The research pipeline did not meet its camera-rate goal. On MOT17-02, the gate reduced calls without approaching the proposed 50–60% reduction target. This consumed experiment time and initially directed native optimization toward a low-cost decision stage. No measured power saving or external customer impact is established.

The original 50-frame profile had a 227.51 ms mean pipeline latency. The updated instrumented profile measured 118.78 ms mean, 143.44 ms p99 and 8.06 observed FPS. This is a material improvement and still far from 30 FPS. These are synchronous attribution runs. A later async replay passed latency-only gates at 30.0 FPS and 5.61 ms p99 but emitted zero tracks on every frame; it does not prove useful asynchronous tracking performance.

## Timeline and facts

Exact incident onset and intermediate wall-clock timestamps were not recorded; the sequence below is evidence-ordered rather than a reconstructed operational timeline.

| Event | Evidence |
|---|---|
| Initial real MOT17-02 comparison | [saved results](results-mot17-02.json): 484/600 adaptive detector calls, baseline 3.8436 FPS, adaptive 4.2110 FPS |
| MOT17-04 validation | Initial reported detection fraction ~0.701; it is a different sequence and must not be described as the ~80% result |
| Baseline profiling | [original summary](../reports/profiling/summary.json): 50 frames, 507 feature-selection calls, 114.6196 ms/frame feature cost |
| Localized reseeding and bounds tests | `src/features.py`, tracker/worker integration and `tests/test_features.py` |
| Paired comparison | [paired summary](../reports/roi-features/paired/summary.json): three 50-frame repetitions, identical saved boxes and alternating backend order |
| Updated profile | [ROI profile](../reports/roi-features/profile/summary.json): 1.3052 ms/frame feature cost, 1.10% of pipeline |
| Run-level SLO mechanism | [design contract](DESIGN.md) and `scripts/verify_slos.py`; a 600-frame saved run passes timing but has zero populated outputs; useful-tracking qualification remains open |

## Detection and diagnosis

Call-count reduction was used as an early optimization proxy before stage-level costs were available. Aggregate FPS alone could not distinguish detector execution, optical flow, feature selection, association or bookkeeping. The profiler combined `cProfile` with exclusive phase timers; nested parent spans were not summed as independent costs. It found feature selection exceeded YOLO's 79.67 ms/frame in the original profile. This justified moving work to cropped ROIs before further optimizing gate arithmetic.

The observed gate frequency is established; its complete causal breakdown is not. The gate fires for missing tracks, interval expiry, low cumulative LK confidence or normalized displacement. Multiplying confidence by successive flow-quality estimates can cross a fixed threshold frequently, especially with weak features or occlusion. That mechanism is visible in code, but no cited historical trigger histogram establishes its share of the 484 calls. It would be incorrect to attribute all calls to confidence drift or all throughput loss to the gate. The costly reseeding path and the frequent triggering are related workload effects, not proven identical root causes.

## Five Whys

1. **Why was detector-call reduction modest?** The configured policy triggered on 80.67% of MOT17-02 frames. The dominant reason requires reason-count analysis; it is not recoverable from the call fraction alone.
2. **Why did fewer calls fail to deliver camera-rate throughput?** Inference scheduling left expensive LK/reseeding work in the frame path, and refreshes repeatedly invoked feature selection.
3. **Why was feature selection expensive despite ROI masks?** Each per-track call still received the full 1920×1080 image. A mask restricts eligible corners, while cropping restricts the image region on which OpenCV computes responses.
4. **Why was the computation boundary missed?** The design treated a spatial selection mask as a computation bound without a paired measurement of the actual call region.
5. **Why did optimization initially favor native gating?** The research process did not yet require phase attribution and an end-to-end latency contract before prioritizing native work. This is a missing verification mechanism, not an individual error.

## Resolution and validation

The shared selector clips boxes to image bounds, handles degenerate/off-image boxes, uses a crop-sized mask, restores pixel coordinates and excludes points outside fractional bounds. Tests exercise image/mask dimensions, active-track containment, invalid inputs, textureless ROIs and async parity. Existing tracks are not reseeded when marked missed.

The paired selector cost fell from 147.797 to 2.098 ms/frame: **98.58% reduction**, including mask/offset overhead. Repeat means were 117.159/189.146/137.086 ms for full-frame masking and 1.404/3.402/1.486 ms for cropped ROIs. Variation is substantial. The profiler's 114.62→1.31 ms comparison measures native feature calls and includes profiling overhead; it is a different timing method, collected at different times. Neither number is a universal device speedup.

A preflight passed six CPU samples averaging 4.43%, but thermal conditions were uncontrolled and host load could change during collection. Selected corner populations changed (82,926 versus 83,406 over the paired workload). HOTA/MOTA have not been reevaluated for this change. No fidelity-equivalence claim is supported yet.

## What worked / what needs improvement

Preserved raw ledgers, source/model hashes and exclusive timing made the bottleneck reproducible. A paired benchmark with identical boxes isolated the selector change. Waiting for idle preflight avoided knowingly measuring under heavy load. Remaining gaps are repeated full-sequence accuracy validation, trigger-reason attribution, populated-tracker async camera-rate qualification and controlled thermal/power measurements. The development test suite verifies mechanics, not these research outcomes.

## Corrective and preventive actions

Owners are roles awaiting individual assignment; due points are review milestones, not invented calendar commitments.

| Action | Owner role | Due point | Status / acceptance |
|---|---|---|---|
| Crop and locally mask reseeding | Tracking engineer | Before next ablation | Implemented; bounds tests and paired latency evidence saved |
| Preserve exclusive stage profiling | Systems engineer | Every hot-path change | Implemented profiler; retain measured baseline and provenance |
| Add run-level SLO gate | Evaluation lead | Before latency qualification | Populated replay verified (30.0 FPS / 29.74 ms p99 / 0.67% misses); contract v3 requires ≥95% live-frame coverage |
| Analyze trigger reason frequencies | Evaluation lead | Before gate retuning | Open; export reason counts with source sequence and settings |
| Evaluate causal and deadline-valid HOTA/MOTA | Evaluation lead | Before release claim | Harness implemented; real ROI async sweep pending |
| Repeat full-sequence hardware runs | Systems engineer | Before release claim | Open; ≥5 repetitions/sequence and preserve all failures |
| Measure thermal/power conditions | Systems engineer | Before energy-efficiency claim | Open; CPU utilization alone is insufficient |

Close this COE only when accuracy and async SLO follow-ups have evidence or an explicitly reviewed disposition. The local feature-cost remediation does not close the full performance objective.

## Format reference

This document follows the COE emphasis on impact, evidence, contributing mechanisms, blameless Five Whys and verifiable preventive actions described in [AWS: creating a Correction of Errors document](https://aws.amazon.com/blogs/mt/creating-a-correction-of-errors-document/) and [AWS: why develop a COE](https://aws.amazon.com/blogs/mt/why-you-should-develop-a-correction-of-error-coe/). It is a project adaptation, not an internal Amazon template or incident record.

## Follow-up: empty-output async timing result

The [saved FixedKPolicy(3) run](verified-performance.md) completed 123 inferences but accepted no published correction. Rejection reasons are recorded; target lag and replay budget dominate this run. Its light publication loop met timing targets because no live tracks incurred propagation work. This is a separate fusion/bootstrap finding, not a regression in measured ROI extraction cost. Keep the COE open until a populated-tracker benchmark and full-sequence fidelity checks are reviewed.

## Populated-run resolution

The [populated artifact](../results/slos/mot17-02-fixed3-populated/slo_report.json) supersedes the empty-output timing evidence: 600/600 frames contain tracks, 125/177 background runs accept corrections, p95 is 26.21 ms and p99 is 29.74 ms. The bootstrap/replay defect is corrected for this measured workload. Audit follow-up adds a 15-frame synchronous retry cooldown and ≥95% coverage gate to prevent empty-scene blocking and weak single-frame acceptance. Full accuracy and repeated deployment qualification remain separate evidence requirements.
