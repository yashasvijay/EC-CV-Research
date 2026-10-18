# Adaptive gating grid search

> Current scope (2026-10-18): The original 48-configuration synchronous grid is complete (48/48). Its source hashes describe the historical tracker; use a new output for a changed implementation. See [verified performance](verified-performance.md).

Run the search on a development sequence, then verify selected configurations
on held-out sequences. Lower confidence thresholds can permit longer tracking
without detection, but neither higher intervals nor relaxed motion thresholds
guarantee 50–60% inference reduction. The weakest LK track can trigger a whole
frame refresh. Trigger counts are retained to diagnose this behavior.

## Full requested grid

```bash
source .venv/bin/activate
python -m scripts.grid_search \
  --sequence data/raw/MOT17/train/MOT17-04-FRCNN \
  --model models/yolo11n.onnx \
  --baseline-run results/baseline-04 \
  --output results/grid_search \
  --threads 1 \
  --eval-python .venv-eval/bin/python
```

The defaults sweep all **48** combinations:

- Maximum detection interval: `[10, 15, 20, 30]`.
- Minimum tracking confidence: `[0.5, 0.6, 0.7]`.
- Maximum normalized displacement: `[0.15, 0.20, 0.25, 0.30]`.

Each run uses the same adaptive LK pipeline in a fresh process, executes
sequentially, and records the full benchmark output. Execution order is shuffled
with seed 42. The default uses one repetition; `--repeats 3` produces 144 runs
and exports means and sample SD across repetitions. Frame correlations are not
treated as independent statistical replicates.

At roughly five to six minutes per MOT17-04 run, the 48-run search may take
several hours. It prints each configuration and a progress heartbeat every
30 seconds. Do not run other benchmark jobs concurrently. The script creation
and synthetic checks do not execute or establish results for the real grid.

MOT17 source directories or their `img1/` directories enable TrackEval
automatically when ground truth is present. Videos and ordinary image folders
can also be used; without ground truth, accuracy metrics remain null and only
the efficiency frontier is meaningful. `--evaluation on` requires full MOT17
frames/GT. `--evaluation off --max-frames 100` enables a short smoke test, whose
latency or reduction must not be presented as a full-sequence result.

An optional baseline must match the model hash, source, frame range, LK tracker,
class selection, warmup and thread count. If accuracy has not yet been evaluated,
the baseline is evaluated once before the sweep and its existing summary is
enriched. Offline evaluation does not contribute to benchmark latency or FPS.

## Stability limits

There is no universal optimal point without a quality tolerance. For example,
adding `--max-hota-drop 1.0` would require HOTA no more than one percentage point
below the supplied baseline; this is an example constraint to choose explicitly,
not an established acceptance threshold for the project. IDF1, MOTA and ID
switches are exported as additional evidence and are not constrained by that
HOTA-only flag. Inspect them before accepting a candidate.

HOTA/MOTA/IDF1 use percentage points; IDSW is a count. A negative HOTA drop means
improved HOTA. Configurations with 50–60% measured inference reduction are
flagged with `in_target_reduction_band`, whether or not they meet a quality limit.

## Outputs

Under `results/grid_search/`:

| File | Contents |
|---|---|
| `manifest.json` | Configuration, source/model/code fingerprints, planned jobs, completed runs and status |
| `summary.json` | Per-run records, aggregated configurations, objectives, baseline and all frontiers |
| `summary.csv` | One row per completed configuration, including latency quantiles and frontier flags |
| `pareto_frontier.json` | Efficiency, accuracy/efficiency, and HOTA-constrained frontiers |
| `pareto_frontier.csv` | Union of efficiency and quality frontier points with flags |
| `runs/job-XXX-attempt-YY/` | Complete per-frame logs, predictions, summary and evaluation reports |
| `job-XXX-attempt-YY.log` | Captured benchmark/evaluator stdout and stderr |

The **efficiency frontier** maximizes inference reduction and minimizes mean
pipeline latency. The **quality frontier** adds maximizing HOTA. A point is
excluded only when another point is at least as good in every objective and
strictly better in one. Exact ties are retained. The HOTA-constrained frontier
first filters by the chosen baseline loss limit, then applies efficiency
objectives. Unevaluated accuracy is never filled with a synthetic score.

Inference reduction is `100 * (1 - detector_calls / processed_frames)`, relative
to full inference. Latency fields are mean/p50/p95/p99 pipeline milliseconds;
throughput includes decoding and output instrumentation. Process CPU and sampled
RSS retain the existing logger's definitions. One repetition yields null sample
SD. Repeated quantiles are averaged per-run quantiles, not pooled frame quantiles.

## Interruptions and resume

Stop with Ctrl+C, then rerun the identical command with `--resume`. Successfully
recorded jobs are skipped. Failed or interrupted jobs get a new attempt directory,
preserving the earlier files. Existing output directories require `--resume`
or a different output path; they are never silently overwritten.

Resume requires the same model, grid, source metadata, code fingerprints,
package versions, interpreter version and baseline summary. Source fingerprints
include file names, sizes, modification times, sequence metadata and GT contents;
they are a fast change check, not cryptographic hashes of every image pixel.
If these inputs change, start a new search directory. Completed summaries are
also hashed and checked before resume.

All reports update after each completed run. After failure they describe only
completed configurations; inspect `status`, `planned_runs` and `completed_runs`
before treating the export as a complete search. Candidate performance during a
partial search remains provisional.
