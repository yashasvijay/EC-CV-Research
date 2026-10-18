# Localized feature selection: measured results

MOT17-02-FRCNN, frames 1–50, 1920 × 1080, one OpenCV/ORT thread. The paired selector benchmark and updated synchronous baseline profile each ran once, sequentially.

## Paired selector timing

Three repetitions used identical saved frame/box workloads with alternating backend order. These wall-clock measurements include mask construction and coordinate offsets, exclude decoding/logging, and do not use cProfile.

| Selector | Mean ms/frame | p95 ms | p99 ms | Repeat means ms | Repeat SD ms |
|---|---:|---:|---:|---|---:|
| full_frame_mask | 147.797 | 222.000 | 622.327 | 117.159, 189.146, 137.086 | 37.170 |
| cropped_roi_mask | 2.098 | 2.466 | 5.064 | 1.404, 3.402, 1.486 | 1.131 |

Observed reseeding reduction: **98.58%** (70.46×). Each backend made 1,521 calls. Substantial repeat variation limits precision; these three repetitions do not establish a stable hardware-wide speedup. Corner totals differ: 82,926 versus 83,406.

## Updated instrumented pipeline

| Stage | ms/frame | Pipeline share |
|---|---:|---:|
| YOLO ONNX session.run | 82.894 | 69.788% |
| Detector preprocessing/postprocessing | 1.539 | 1.296% |
| OpenCV LK (forward + backward) | 29.786 | 25.077% |
| Flow filtering/motion/state/copy | 1.505 | 1.267% |
| OpenCV goodFeaturesToTrack | 1.305 | 1.099% |
| Scalar IoU calls | 0.853 | 0.718% |
| Hungarian assignment solve | 0.005 | 0.005% |
| Update loops/masks/lifecycle/reseed setup | 0.604 | 0.509% |
| Grayscale conversion | 0.252 | 0.212% |
| Pipeline orchestration/timing overhead | 0.035 | 0.030% |

Pipeline mean/p95/p99: 118.779/140.745/143.444 ms. Observed throughput including export/logging: 8.063 FPS. Process CPU: 98.31% (100% is one logical core); sampled peak RSS: 247.17 MiB. Both 33 ms and 50 ms pipeline-work deadline miss rates remain 100%.

Historical feature-selection attribution was 114.620 ms/frame (50.38%); updated attribution is 1.305 ms/frame (1.10%). Historical pipeline mean was 227.514 ms and throughput 4.258 FPS. These profiles include cProfile and timer overhead and were collected at different times; their comparison is descriptive, not a paired end-to-end causal estimate. Native OpenCV feature-selection attribution excludes Python crop/mask preparation, so it is distinct from the paired selector timing above.

## Conditions and provenance

The preflight passed after a 20-second cooldown: six two-second CPU samples were 5.2, 3.3, 1.9, 4.6, 7.8 and 3.8%, averaging 4.43%. No competing experiment process was found. Load checks do not establish controlled thermal conditions, and background load may change during measurement. No thermal or power measurements were taken.

Cropping changes border derivatives and corner populations. HOTA, MOTA, IDF1 and ID switches have not been reevaluated; no tracking-fidelity equivalence is claimed. The original profile and completed grid-search artifacts passed SHA-256 preservation checks.

Model SHA-256: `3586b83d3d87a0465a7df8a85b9c0fb7c8fbac4efacd1dbb5e1ae535707dbc4b`.
Saved box workload SHA-256: `53a31319b1016cc112dd4f080347481ff17746384792472422efce69a467b9a3`.
Benchmark source SHA-256: `7332c95c6871c9c78ec06e7e4ceb98651705d57ec67188747e604e1fc461e681`.

Current profile source hashes:

- `scripts/profile_pipeline.py`: `1c99bcf178633dac3a142944f2372b0f86f7014e734364550ac60ba296c3edd9`
- `src/pipeline.py`: `37a02f07065cdc988e5d03272d7155219ce41db5d9c4815fdd02389c1d677ebb`
- `src/tracker.py`: `589858194a92ba8fa72de180c39b052a77e15a5bd4f6563b6cfc89fb89284e88`
- `src/features.py`: `ec54b2648b6a7736089b9413d5f8c9b55dd48237e5536e00b160f4b132bd22c3`
- `src/detector.py`: `6204fad6aa356b8af57a1912860ecd221a90865a2a66c48e11252560ec6e9b96`
- `src/benchmark.py`: `0d2543caaa17d731c6946fa373acd50e0c0d8f40e0b280a37d74bec1305c3dd9`
- `src/datasets/mot17.py`: `137b959dcc296a7a3e9e8a1f15261c0eaac8027af262a481039e257b8b11c3e0`

Artifacts: [paired summary](paired/summary.json), [paired rows](paired/frames.jsonl), [profile summary](profile/summary.json), [profile dump](profile/baseline_profile.prof), [validation](validation.json), [load checks](preflight.jsonl).
