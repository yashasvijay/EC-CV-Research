"""Fresh-process fixed-K inference sweep, evaluated against the same full sequence."""

import argparse
import csv
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from src.datasets.mot17 import MOT17Sequence
from scripts.evaluate_tracking import atomic_json


def summarize(rows):
    curve = []
    for interval in sorted({row["K"] for row in rows}):
        runs = [row for row in rows if row["K"] == interval]
        item = {"K": interval, "runs": len(runs)}
        for key in (
            "HOTA",
            "MOTA",
            "IDF1",
            "IDSW",
            "inference_reduction_percent",
            "FPS",
            "process_cpu_percent_mean",
            "sampled_peak_rss_mib",
        ):
            values = [row[key] for row in runs]
            item[key] = float(np.mean(values))
            item[key + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        curve.append(item)
    return curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, help="new sweep directory")
    parser.add_argument("--intervals", nargs="+", type=int, default=[1, 2, 5, 10, 15])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-python",
        default=sys.executable,
        help="Python with requirements-eval.txt installed (e.g. .venv-eval/bin/python)",
    )
    args = parser.parse_args()
    intervals = args.intervals
    if (
        len(set(intervals)) != len(intervals)
        or min(intervals) < 1
        or 1 not in intervals
        or args.repeats < 1
        or args.threads < 1
        or args.warmup < 0
        or args.min_hits < 1
        or not 0 < args.match_iou <= 1
    ):
        parser.error("Use unique positive intervals including K=1 and valid run parameters")
    if args.max_age < max(intervals) - 1:
        parser.error("max-age must cover skipped frames for every K; keep it fixed across the sweep")
    sequence = MOT17Sequence(args.sequence)
    if not sequence.ground_truth_path.is_file():
        parser.error("MOT17 ground truth is required for the tradeoff curve")
    model = Path(args.model).resolve()
    if not model.is_file():
        parser.error(f"Missing model: {model}")
    # Fail before an expensive sweep if evaluation is not installed.
    # Do not resolve the venv's Python symlink: that would bypass its packages.
    eval_python = str(Path(args.eval_python).absolute())
    subprocess.run([eval_python, "-c", "import trackeval"], check=True)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    plan = []
    for repeat in range(1, args.repeats + 1):
        order = intervals.copy()
        rng.shuffle(order)
        plan.extend({"repeat": repeat, "K": interval} for interval in order)
    manifest = {
        "status": "running",
        "config": vars(args),
        "sequence": sequence.metadata,
        "order": plan,
        "completed": [],
    }
    atomic_json(output / "manifest.json", manifest)
    rows = []
    try:
        for job in plan:
            interval, repeat = job["K"], job["repeat"]
            run = output / f"k{interval:02d}-r{repeat:02d}"
            command = [
                sys.executable,
                "-m",
                "scripts.run_phase1",
                "--model",
                str(model),
                "--source",
                str(sequence.path),
                "--output",
                str(run),
                "--tracker",
                "sort",
                "--mode",
                "fixed",
                "--max-interval",
                str(interval),
                "--threads",
                str(args.threads),
                "--warmup",
                str(args.warmup),
                "--max-age",
                str(args.max_age),
                "--min-hits",
                str(args.min_hits),
                "--match-iou",
                str(args.match_iou),
                "--classes",
                "0",
            ]
            print(f"Running K={interval}, repetition {repeat}", flush=True)
            with (output / f"{run.name}.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                subprocess.run(
                    [
                        eval_python,
                        "-m",
                        "scripts.evaluate_tracking",
                        "--sequence",
                        str(sequence.path),
                        "--run-dir",
                        str(run),
                    ],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            summary = json.loads((run / "summary.json").read_text())
            row = job | {
                key: summary[key]
                for key in (
                    "HOTA",
                    "MOTA",
                    "IDF1",
                    "IDSW",
                    "process_cpu_percent_mean",
                    "sampled_peak_rss_mib",
                )
            }
            row.update(
                FPS=summary["throughput_fps"],
                inference_reduction_percent=100 * (1 - summary["detection_fraction"]),
                run_dir=str(run),
            )
            rows.append(row)
            atomic_json(output / "runs.json", rows)
            curve = summarize(rows)
            atomic_json(output / "tradeoff.json", curve)
            with (output / "tradeoff.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
                writer.writeheader()
                writer.writerows(curve)
            manifest["completed"].append(job)
            atomic_json(output / "manifest.json", manifest)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        atomic_json(output / "manifest.json", manifest)
    print(f"Tradeoff curve: {output / 'tradeoff.csv'}")


if __name__ == "__main__":
    main()
