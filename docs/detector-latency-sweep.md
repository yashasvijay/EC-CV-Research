# Detector service-latency injection

Run serial fresh-process cases at 40, 80, 160 and 320 ms:

```bash
python -m scripts.sweep_detector_latency \
  --latencies-ms 40 80 160 320 --frames 600 --repeats 1 \
  --width 1920 --height 1080 --tracks 8 --k 3 --threads 1 \
  --output results/detector-latency-envelope
```

The workload is deterministic moving textured rectangles with exact synthetic source boxes, not MOT17 or learned YOLO predictions. Each detector call waits for the remainder of the requested minimum service time; scheduling/sleep overshoot is measured in observed detector service telemetry. Bootstrap uses the same delay. Source image generation is included relative to nominal camera release. This isolates scheduling/replay/backpressure under service delay, not lower-clock hardware's CPU contention, cache, power or thermal behavior.

The parent shuffles latency/repetition order with seed 42, requires idle-host preflight before each case, and launches each case in a new Python process. Defaults use a single repetition; use more repetitions to quantify variability. The script never overwrites completed or partial output directories. A busy-host check leaves a deferred manifest and starts no new case. Already completed cases remain available; choose a new output for a fresh sweep.

Artifacts:

- `manifest.json`: planned order, completed cases and complete/deferred/failed status.
- `summary.json` and `summary.csv`: case-level latency, useful fusion ratio, track-frame fraction, deadline misses, CPU/RAM and fallback frequency. A sweep can complete with failed SLO cases; that is operating-envelope evidence, not a harness error.
- Each case contains `runtime/summary.json`, frame/event ledgers, causal/deadline track files, an SLO JSON/Markdown report and `case.json`.
- Per-case subprocess logs and preflight samples remain beside case directories.

`useful_fusion_ratio` counts background detector **runs with at least one accepted correction**, divided by completed background runs. It does not count individual boxes. `fallback_attempt_fraction` is synchronous bootstrap/recovery attempts divided by source frames; attempts/second uses measured runtime wall duration. The 15-frame default cooldown bounds retry frequency even when no objects can be seeded. Case metadata retains seed, image/track dimensions, runtime configuration and source/driver hashes.

The sweep is implemented and covered by synthetic integration tests. No full four-delay performance result is asserted until the corresponding complete manifest and raw artifacts exist. Short test cases deliberately fail the 600-frame qualification minimum. The separate populated MOT17 hardware run is documented in [verified performance](verified-performance.md).
