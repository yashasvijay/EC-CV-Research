# Async runtime SLO verification

Status: **PASS**

| Metric | Status | Measured | Rule |
|---|---|---:|---|
| measurement_integrity | PASS | — |  [] |
| throughput_fps | PASS | 30.0 | >= 30.0 |
| p95_e2e_ms | PASS | 5.5648979 | < 33.0 |
| p99_e2e_ms | PASS | 5.60960342 | < 33.0 |
| deadline_miss_fraction | PASS | 0.0016666666666666668 | < 0.01 |

run-level latency only; tracking accuracy and production camera qualification are separate

Latency uses nominal capture-to-publication timestamps, including backlog. No accuracy, thermal, power or hard real-time guarantee is inferred.
