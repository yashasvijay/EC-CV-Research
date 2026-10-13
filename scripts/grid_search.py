"""Resumable adaptive-LK search with measured efficiency and tracking-quality frontiers.

Run as python -m scripts.grid_search. Each benchmark runs sequentially in a
fresh process. TrackEval is automatic for full MOT17 sequences with ground truth.
"""
import argparse
import csv
import hashlib
from importlib.metadata import version
from itertools import product
import json
import math
import platform
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import numpy as np

from scripts.evaluate_tracking import atomic_json
from src.datasets.mot17 import MOT17Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTERVALS = [10, 15, 20, 30]
DEFAULT_CONFIDENCES = [0.5, 0.6, 0.7]
DEFAULT_DISPLACEMENTS = [0.15, 0.20, 0.25, 0.30]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pareto_frontier(rows, objectives):
    """Retain non-dominated rows, including exact ties; omit missing metrics.

    objectives maps metric names to 'min' or 'max'. A dominating candidate is
    at least as good in every objective and strictly better in at least one.
    """
    if not objectives or any(direction not in ("min", "max") for direction in objectives.values()):
        raise ValueError("Use nonempty min/max objectives")
    valid = [row for row in rows if all(row.get(key) is not None and math.isfinite(row[key])
                                       for key in objectives)]
    def values(row):
        return tuple(row[key] * (1 if direction == "min" else -1)
                     for key, direction in objectives.items())
    points = [values(row) for row in valid]
    return [row for i, row in enumerate(valid) if not any(
        all(a <= b for a, b in zip(other, points[i])) and
        any(a < b for a, b in zip(other, points[i]))
        for j, other in enumerate(points) if i != j
    )]


def run_command(command, log_path, run_path=None):
    """Capture child output and show a 30-second progress heartbeat."""
    with Path(log_path).open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    return_code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    frame_count = 0
                    ledger = Path(run_path) / "frames.jsonl" if run_path else None
                    if ledger and ledger.is_file():
                        with ledger.open() as stream:
                            frame_count = sum(1 for line in stream if line.endswith("\n"))
                    print(f"  still running; {frame_count} logged frames; log={log_path}", flush=True)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if return_code:
        raise RuntimeError(f"Command failed (exit {return_code}); inspect {log_path}")


def row_from_summary(job, summary, run_path):
    if summary.get("status") != "complete" or summary.get("frames", 0) < 1:
        raise ValueError(f"Incomplete/empty benchmark: {run_path}")
    fraction = summary["detection_fraction"]
    if not 0 <= fraction <= 1:
        raise ValueError("Invalid detection fraction")
    detector_calls = sum(value for key, value in summary["triggers"].items() if key != "skipped")
    latency = summary["pipeline_latency_ms"]
    return job | {
        "run_dir": str(run_path), "frames": summary["frames"], "detector_calls": detector_calls,
        "detection_fraction": fraction, "inference_reduction_percent": 100 * (1 - fraction),
        "throughput_fps": summary["throughput_fps"], "pipeline_fps": summary["pipeline_fps"],
        **{f"pipeline_latency_{key}_ms": latency[key] for key in ("mean", "p50", "p95", "p99")},
        "process_cpu_percent_mean": summary["process_cpu_percent_mean"],
        "sampled_peak_rss_mib": summary["sampled_peak_rss_mib"],
        **{key: summary.get(key) for key in ("HOTA", "MOTA", "IDF1", "IDSW")},
        "accuracy_status": summary["accuracy_status"], "trigger_counts": summary["triggers"],
        "summary_sha256": sha256(Path(run_path) / "summary.json"),
    }


def aggregate_rows(rows, baseline_hota=None, max_hota_drop=None):
    grouped = {}
    for row in rows:
        key = tuple(row[name] for name in ("max_interval", "min_confidence", "max_displacement"))
        grouped.setdefault(key, []).append(row)
    aggregates = []
    metrics = ("detector_calls", "inference_reduction_percent", "throughput_fps", "pipeline_fps",
               "pipeline_latency_mean_ms", "pipeline_latency_p50_ms", "pipeline_latency_p95_ms",
               "pipeline_latency_p99_ms", "process_cpu_percent_mean", "sampled_peak_rss_mib",
               "HOTA", "MOTA", "IDF1", "IDSW")
    for index, (key, runs) in enumerate(sorted(grouped.items()), 1):
        item = dict(zip(("max_interval", "min_confidence", "max_displacement"), key))
        item.update(configuration_id=f"config-{index:03d}", repetitions=len(runs),
                    frames=runs[0]["frames"], run_ids=[row["job_id"] for row in runs])
        for metric in metrics:
            values = [row[metric] for row in runs]
            measured = all(value is not None for value in values)
            item[metric] = float(np.mean(values)) if measured else None
            item[metric + "_sd"] = float(np.std(values, ddof=1)) if measured and len(values) > 1 else None
        item["hota_drop_pp"] = baseline_hota - item["HOTA"] if baseline_hota is not None and item["HOTA"] is not None else None
        item["within_hota_limit"] = (item["hota_drop_pp"] <= max_hota_drop
                                     if max_hota_drop is not None and item["hota_drop_pp"] is not None else None)
        item["in_target_reduction_band"] = 50 <= item["inference_reduction_percent"] <= 60
        aggregates.append(item)
    efficiency = {"inference_reduction_percent": "max", "pipeline_latency_mean_ms": "min"}
    quality = efficiency | {"HOTA": "max"}
    efficiency_ids = {row["configuration_id"] for row in pareto_frontier(aggregates, efficiency)}
    quality_ids = {row["configuration_id"] for row in pareto_frontier(aggregates, quality)}
    eligible = [row for row in aggregates if row["within_hota_limit"] is True]
    constrained_ids = {row["configuration_id"] for row in pareto_frontier(eligible, efficiency)}
    for row in aggregates:
        row.update(pareto_efficiency=row["configuration_id"] in efficiency_ids,
                   pareto_quality=row["configuration_id"] in quality_ids,
                   pareto_within_hota_limit=row["configuration_id"] in constrained_ids)
    return aggregates


def atomic_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=Path(path).parent, newline="", delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                                 for key, value in row.items()})
        temporary.replace(path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def export_results(output, manifest):
    baseline = manifest["baseline"]
    aggregates = aggregate_rows(manifest["runs"], baseline.get("HOTA") if baseline else None,
                                manifest["config"]["max_hota_drop"])
    frontiers = {
        "efficiency": [row for row in aggregates if row["pareto_efficiency"]],
        "quality": [row for row in aggregates if row["pareto_quality"]],
        "within_hota_limit": [row for row in aggregates if row["pareto_within_hota_limit"]],
    }
    summary = {
        "status": manifest["status"], "planned_runs": len(manifest["jobs"]),
        "completed_runs": len(manifest["runs"]), "config": manifest["config"],
        "baseline": baseline, "configurations": aggregates, "runs": manifest["runs"],
        "pareto_frontiers": frontiers,
        "objectives": {"efficiency": {"inference_reduction_percent": "max", "pipeline_latency_mean_ms": "min"},
                       "quality": {"inference_reduction_percent": "max", "pipeline_latency_mean_ms": "min", "HOTA": "max"}},
        "notes": ["Partial searches describe completed runs only; check status and counts.",
                  "SD is across repetitions, not across correlated frames; one repetition gives null SD.",
                  "HOTA/MOTA/IDF1 are percentage points; IDSW is a count; unevaluated accuracy stays null.",
                  "The frontier contains tradeoffs, not a uniquely optimal configuration.",
                  "Tune on development sequences and verify selected settings on held-out sequences."],
    }
    atomic_json(output / "summary.json", summary)
    atomic_csv(output / "summary.csv", aggregates)
    atomic_json(output / "pareto_frontier.json", frontiers)
    atomic_csv(output / "pareto_frontier.csv", [row for row in aggregates if row["pareto_efficiency"] or row["pareto_quality"]])


def source_fingerprint(source, sequence):
    files = sequence.frame_paths if sequence else sorted(source.iterdir()) if source.is_dir() else [source]
    digest = hashlib.sha256()
    for path in files:
        if path.is_file():
            stat = path.stat()
            digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    if sequence:
        digest.update((sequence.path / "seqinfo.ini").read_bytes())
        if sequence.ground_truth_path.is_file():
            digest.update(sequence.ground_truth_path.read_bytes())
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", "--source", dest="sequence", required=True,
                        help="MOT17 sequence directory, img1 directory, or video")
    parser.add_argument("--model", required=True, help="raw detection ONNX model")
    parser.add_argument("--output", default="results/grid_search")
    parser.add_argument("--max-intervals", nargs="+", type=int, default=DEFAULT_INTERVALS)
    parser.add_argument("--min-confidences", nargs="+", type=float, default=DEFAULT_CONFIDENCES)
    parser.add_argument("--max-displacements", nargs="+", type=float, default=DEFAULT_DISPLACEMENTS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=0, help="0=full source; truncated runs require evaluation off")
    parser.add_argument("--evaluation", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--eval-python", default=str(ROOT / ".venv-eval/bin/python"))
    parser.add_argument("--baseline-run", help="optional matching full-inference run, evaluated if needed")
    parser.add_argument("--max-hota-drop", type=float, help="allowed absolute HOTA percentage-point loss vs baseline")
    parser.add_argument("--resume", action="store_true", help="resume the same configuration without overwriting runs")
    args = parser.parse_args(argv)
    if (args.threads < 1 or args.warmup < 0 or args.repeats < 1 or args.max_frames < 0
            or min(args.max_intervals) < 1
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in args.min_confidences)
            or any(not math.isfinite(v) or v <= 0 for v in args.max_displacements)):
        parser.error("Invalid intervals, thresholds, frame limit, threads or repetitions")
    for values in (args.max_intervals, args.min_confidences, args.max_displacements):
        if len(set(values)) != len(values):
            parser.error("Grid values must be unique")
    if args.max_hota_drop is not None and (not math.isfinite(args.max_hota_drop) or args.max_hota_drop < 0 or not args.baseline_run):
        parser.error("A finite nonnegative --max-hota-drop requires --baseline-run")
    source, model = Path(args.sequence).resolve(), Path(args.model).resolve()
    if not source.exists() or not model.is_file():
        parser.error("Source and ONNX model must exist")
    sequence_path = source if (source / "seqinfo.ini").is_file() else source.parent
    sequence = MOT17Sequence(sequence_path) if (sequence_path / "seqinfo.ini").is_file() else None
    can_evaluate = sequence is not None and sequence.ground_truth_path.is_file()
    evaluation = args.evaluation == "on" or (args.evaluation == "auto" and can_evaluate)
    if evaluation and (not can_evaluate or args.max_frames):
        parser.error("Tracking evaluation requires full MOT17 frames and ground truth; use --evaluation off for smoke tests/videos")
    if args.max_hota_drop is not None and not evaluation:
        parser.error("A tracking quality constraint requires evaluation")
    eval_python = str(Path(args.eval_python).absolute())  # retain venv symlink path
    evaluation_runtime = None
    if evaluation:
        probe = subprocess.run([eval_python, "-c",
                                "import json, platform, trackeval; from importlib.metadata import version; "
                                "print(json.dumps({'python': platform.python_version(), "
                                "'packages': {name: version(name) for name in ('trackeval', 'numpy', 'scipy')}}))"],
                               check=True, capture_output=True, text=True)
        evaluation_runtime = json.loads(probe.stdout)
    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        parser.error(f"Output already exists: {output}; choose a new directory or --resume")
    if args.resume and not (output / "manifest.json").is_file():
        parser.error("--resume requires an existing search manifest")
    model_hash = sha256(model)
    baseline = None
    if args.baseline_run:
        baseline_path = Path(args.baseline_run).resolve()
        baseline = json.loads((baseline_path / "summary.json").read_text())
        metadata = baseline["metadata"]
        recorded_source = Path(metadata["source"])
        if not recorded_source.is_absolute():
            recorded_source = ROOT / recorded_source
        expected_frames = min(len(sequence), args.max_frames) if sequence and args.max_frames else len(sequence) if sequence else None
        if (baseline.get("status") != "complete" or metadata["mode"] != "baseline"
                or metadata.get("tracker", "lk") != "lk" or metadata["model_sha256"] != model_hash
                or metadata["threads"] != args.threads or metadata["warmup"] != args.warmup
                or metadata["classes"] != [0] or baseline["detection_fraction"] != 1.0
                or recorded_source.resolve() not in ({sequence.path, sequence.image_dir} if sequence else {source})
                or expected_frames is not None and baseline["frames"] != expected_frames):
            parser.error("Baseline must match the full source, model, LK tracker, classes and threads")
        if evaluation and baseline.get("accuracy_status") != "evaluated":
            run_command([eval_python, "-m", "scripts.evaluate_tracking", "--sequence", str(sequence.path),
                         "--run-dir", str(baseline_path)], baseline_path / "grid-evaluation.log")
            baseline = json.loads((baseline_path / "summary.json").read_text())
        if evaluation and baseline["evaluation"]["gt_sha256"] != sha256(sequence.ground_truth_path):
            parser.error("Baseline evaluation ground truth has changed")
        baseline = {key: baseline.get(key) for key in ("frames", "throughput_fps", "HOTA", "MOTA", "IDF1", "IDSW")} | {
            "run_dir": str(baseline_path), "summary_sha256": sha256(baseline_path / "summary.json")}
    config = {
        "source": str(source), "model": str(model), "model_sha256": model_hash,
        "source_fingerprint": source_fingerprint(source, sequence),
        "max_intervals": args.max_intervals, "min_confidences": args.min_confidences,
        "max_displacements": args.max_displacements, "repeats": args.repeats,
        "threads": args.threads, "warmup": args.warmup, "seed": args.seed,
        "max_frames": args.max_frames, "evaluation": evaluation,
        "platform": platform.platform(), "python": platform.python_version(),
        "packages": {name: version(name) for name in
                     ("numpy", "opencv-python-headless", "onnxruntime", "scipy", "psutil")},
        "eval_python": eval_python if evaluation else None, "max_hota_drop": args.max_hota_drop,
        "evaluation_runtime": evaluation_runtime,
        "code_sha256": {name: sha256(ROOT / name) for name in
                        ("scripts/grid_search.py", "scripts/run_phase1.py", "scripts/evaluate_tracking.py",
                         "src/detector.py", "src/tracker.py", "src/pipeline.py", "src/benchmark.py", "src/datasets/mot17.py")},
    }
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["config"] != config or manifest["baseline"] != baseline:
            parser.error("Resume configuration, source, model, code, or baseline differs from the saved search")
        for row in manifest["runs"]:
            if sha256(Path(row["run_dir"]) / "summary.json") != row["summary_sha256"]:
                parser.error("A completed run summary has changed; use a new output directory")
    else:
        output.mkdir(parents=True)
        jobs, rng = [], random.Random(args.seed)
        for repeat in range(1, args.repeats + 1):
            combinations = list(product(args.max_intervals, args.min_confidences, args.max_displacements))
            rng.shuffle(combinations)
            for interval, confidence, displacement in combinations:
                jobs.append({"job_id": f"job-{len(jobs)+1:03d}", "repeat": repeat,
                             "max_interval": interval, "min_confidence": confidence,
                             "max_displacement": displacement})
        manifest = {"status": "running", "config": config, "baseline": baseline, "jobs": jobs, "runs": []}
    manifest.update(status="running", error=None)
    atomic_json(output / "manifest.json", manifest)
    export_results(output, manifest)
    completed = {row["job_id"] for row in manifest["runs"]}
    print(f"Search: {len(manifest['jobs'])} runs, {len(completed)} already complete; evaluation={evaluation}", flush=True)
    try:
        for job in manifest["jobs"]:
            if job["job_id"] in completed:
                continue
            attempt = 1
            run = output / "runs" / f"{job['job_id']}-attempt-{attempt:02d}"
            while run.exists():
                attempt += 1
                run = output / "runs" / f"{job['job_id']}-attempt-{attempt:02d}"
            log_path = output / f"{run.name}.log"
            print(f"[{len(manifest['runs'])+1}/{len(manifest['jobs'])}] {job}", flush=True)
            command = [sys.executable, "-m", "scripts.run_phase1", "--model", str(model), "--source", str(source),
                       "--mode", "adaptive", "--tracker", "lk", "--output", str(run),
                       "--threads", str(args.threads), "--warmup", str(args.warmup),
                       "--max-frames", str(args.max_frames), "--max-interval", str(job["max_interval"]),
                       "--min-confidence", str(job["min_confidence"]), "--max-displacement", str(job["max_displacement"])]
            run_command(command, log_path, run)
            if evaluation:
                run_command([eval_python, "-m", "scripts.evaluate_tracking", "--sequence", str(sequence.path),
                             "--run-dir", str(run)], log_path)
            summary = json.loads((run / "summary.json").read_text())
            if sequence and summary["frames"] != (min(len(sequence), args.max_frames) if args.max_frames else len(sequence)):
                raise ValueError("Benchmark did not cover the expected frame range")
            if baseline and summary["frames"] != baseline["frames"]:
                raise ValueError("Baseline and candidate frame ranges differ")
            manifest["runs"].append(row_from_summary(job, summary, run))
            atomic_json(output / "manifest.json", manifest)
            export_results(output, manifest)
            print(f"  reduction={100*(1-summary['detection_fraction']):.2f}%, "
                  f"latency={summary['pipeline_latency_ms']['mean']:.2f} ms, "
                  f"FPS={summary['throughput_fps']:.2f}, HOTA={summary.get('HOTA')}", flush=True)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        atomic_json(output / "manifest.json", manifest)
        export_results(output, manifest)
    print(f"Results: {output / 'summary.json'} and {output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
