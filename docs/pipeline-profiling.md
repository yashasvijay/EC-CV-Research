# Profile the synchronous tracking baseline

> Current scope (2026-10-18): Original and ROI synchronous profiles are preserved under reports/profiling and reports/roi-features/profile. Native feature attribution fell from 114.62 to 1.31 ms/frame; the paired selector reduction is 98.58%. See [verified performance](verified-performance.md).

The standalone profiler executes the existing `Pipeline` in baseline mode. Instrumentation is installed in its own process and restored on exit; production source files and the grid-search process remain unchanged.

The historical run is already saved. For a new attribution run, choose a new directory and run from the repository root:

```bash
.venv/bin/python -m scripts.profile_pipeline \
  --model models/yolo11n.onnx \
  --source data/raw/MOT17/train/MOT17-02-FRCNN \
  --frames 50 --threads 1 --warmup 10 \
  --output reports/profiling-repeat
```

Defaults match this command. `--source` accepts a MOT17 sequence directory or its `img1` directory. The requested segment must exist in full; `--start-frame` selects its first source frame and starts with an empty tracker. Output directories must be new.

Outputs:

- `baseline_profile.prof`: serialized cProfile data, readable with the same Python runtime using `pstats` or a compatible viewer.
- `breakdown.txt`: total exclusive stage partition in milliseconds/frame, percentages, call counts, inclusive parent spans, tail latencies, CPU and sampled RSS.
- `cprofile_top.txt`: top 30 functions by cumulative time. Cumulative function times are nested and must not be summed.
- `summary.json`: machine-readable per-stage totals, mean cost/frame, mean inclusive cost/call, p50/p95/p99 of per-frame exclusive costs, provenance, instrumentation mode, and resource summaries.
- `frames.jsonl`: per-frame inclusive/exclusive timers, call counts, baseline metrics and resources.
- `tracks.txt`: source-frame-indexed MOT-format predictions for output-parity inspection.

The stage partition separates ONNX `session.run` from detector preprocessing/postprocessing; native forward/backward LK calls from filtering/state propagation; `goodFeaturesToTrack` from feature masks and reseeding setup; scalar IoU calls and Hungarian solve from the remaining update loops. The update residual includes crop/mask preparation and Python loop control. In the historical full-image implementation it included full-image mask creation. Grayscale conversion and orchestration are separate. Percentages use measured pipeline time, excluding decode, export and resource logging. Per-frame stage quantiles include zero-cost frames for stages not called; call means use their actual call count.

Only measured decode/pipeline intervals enter cProfile. Model construction, ten detector warmup calls, output export, resource logging and report generation are excluded from the serialized profile. Throughput from `Benchmark` includes export/logging and the instrumented loop. Work-budget exceedance rates at 33/50 ms describe the synchronous pipeline, not capture-to-publication streaming deadlines.

These are attribution measurements with timer and cProfile overhead. In particular, per-IoU instrumentation and deterministic profiling can inflate Python-heavy work. Python's [profiling documentation](https://docs.python.org/3.13/library/profile.html) distinguishes function profiling from performance benchmarking. Do not present this output as uninstrumented throughput or native/Python speedup evidence. A lower-overhead companion run is available in a separate output directory:

```bash
.venv/bin/python -m scripts.profile_pipeline --frames 50 --threads 1 \
  --timing-only --output reports/profiling-timers-only
```

It retains phase timers but produces no `.prof` file. Compare repeated runs under equivalent idle conditions; this companion still includes timers and does not measure timer-free production throughput. Use an unmodified baseline run to establish that separately.

By default, a grid-search manifest marked `running` prevents model construction or inference. A stale manifest is treated conservatively; verify its actual process rather than editing the manifest merely to bypass the guard. `--allow-concurrent` explicitly enables a diagnostic and records the active manifests in metadata, but concurrent measurements are unsuitable for the requested uncontended baseline. `--search-root` changes where manifests are checked.

Ensure no competing inference/profiling workload is active and record relevant hardware/power/thermal conditions before the real run. HOTA/MOTA/IDF1/IDSW remain null: a 50-frame attribution prefix does not constitute full-sequence fidelity validation. Preserve partial outputs if a run fails or is interrupted; never use them as the completed 50-frame result.
