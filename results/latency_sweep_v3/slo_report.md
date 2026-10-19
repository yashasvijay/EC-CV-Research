# Async runtime SLO verification

Status: **PASS**

| Metric | Status | Measured | Rule |
|---|---|---:|---|
| track_frame_fraction | PASS | 1.0 | >= 0.95 |
| useful_fusion_ratio | PASS | 0.9949494949494949 | >= 0.1 |
| p95_detection_age_ms | PASS | 182.931975 | < 300.0 |
| measurement_integrity | PASS | — |  [] |
| throughput_fps | PASS | 30.0 | >= 30.0 |
| p95_e2e_ms | PASS | 20.910568749999996 | < 33.0 |
| p99_e2e_ms | PASS | 23.57004507999999 | < 33.0 |
| deadline_miss_fraction | PASS | 0.006666666666666667 | < 0.01 |

Run-level timing and useful-fusion checks; accuracy and physical-camera qualification remain separate

Latency uses nominal capture-to-publication timestamps, including backlog. No accuracy, thermal, power or hard real-time guarantee is inferred.
