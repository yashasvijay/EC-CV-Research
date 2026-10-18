# Async runtime SLO verification

Status: **PASS**

| Metric | Status | Measured | Rule |
|---|---|---:|---|
| track_frame_fraction | PASS | 1.0 | >= 0.95 |
| useful_fusion_ratio | PASS | 0.3583815028901734 | >= 0.1 |
| measurement_integrity | PASS | — |  [] |
| throughput_fps | PASS | 30.0 | >= 30.0 |
| p95_e2e_ms | PASS | 26.621411050000003 | < 33.0 |
| p99_e2e_ms | PASS | 29.698941249999994 | < 33.0 |
| deadline_miss_fraction | PASS | 0.005 | < 0.01 |

Run-level timing and useful-fusion checks; accuracy and physical-camera qualification remain separate

Latency uses nominal capture-to-publication timestamps, including backlog. No accuracy, thermal, power or hard real-time guarantee is inferred.
