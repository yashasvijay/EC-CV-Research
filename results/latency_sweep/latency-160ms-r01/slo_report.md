# Async runtime SLO verification

Status: **FAIL**

| Metric | Status | Measured | Rule |
|---|---|---:|---|
| track_frame_fraction | PASS | 1.0 | >= 0.95 |
| useful_fusion_ratio | FAIL | 0.0 | >= 0.1 |
| measurement_integrity | PASS | — |  [] |
| throughput_fps | FAIL | 29.806608205765446 | >= 30.0 |
| p95_e2e_ms | FAIL | 173.38263364999992 | < 33.0 |
| p99_e2e_ms | FAIL | 211.56872342 | < 33.0 |
| deadline_miss_fraction | FAIL | 0.175 | < 0.01 |

Run-level timing and useful-fusion checks; accuracy and physical-camera qualification remain separate

Latency uses nominal capture-to-publication timestamps, including backlog. No accuracy, thermal, power or hard real-time guarantee is inferred.
