"""Serial, fresh-process full-sequence async policy ablation."""

import argparse
import csv
import json
from pathlib import Path
import random
import subprocess
import sys
from scripts.evaluate_tracking import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequences", nargs="+", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--eval-python", default=".venv-eval/bin/python")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--oracle-schedule-dir", help="Optional JSON frame-list file named <sequence>.json")
    a = p.parse_args()
    if a.repeats < 1:
        p.error("repeats must be positive")
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=False)
    plan = [
        {"sequence": s, "policy": policy, "repeat": r}
        for s in a.sequences
        for r in range(1, a.repeats + 1)
        for policy in ["fixed_k", "kalman", "heuristic", "oracle"]
    ]
    random.Random(a.seed).shuffle(plan)
    manifest = {"status": "running", "config": vars(a), "plan": plan, "completed": []}
    rows = []
    atomic_json(root / "manifest.json", manifest)
    try:
        for job in plan:
            name = f"{Path(job['sequence']).name}-{job['policy']}-r{job['repeat']}"
            run = root / name
            cmd = [
                sys.executable,
                "-m",
                "scripts.run_async_evaluation",
                "--sequence",
                job["sequence"],
                "--model",
                a.model,
                "--policy",
                job["policy"],
                "--output",
                str(run),
                "--eval-python",
                a.eval_python,
            ]
            if job["policy"] == "oracle" and a.oracle_schedule_dir:
                cmd += [
                    "--oracle-schedule",
                    str(Path(a.oracle_schedule_dir) / (Path(job["sequence"]).name + ".json")),
                ]
            print("Running " + name, flush=True)
            with (root / (name + ".log")).open("w") as log:
                subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
            s = json.loads((run / "summary.json").read_text())
            wc = s["worker_counts"]
            row = {
                **job,
                "sequence": Path(job["sequence"]).name,
                "run_dir": str(run),
                "configuration_id": name,
                **{
                    k: s[k]
                    for k in [
                        "HOTA",
                        "MOTA",
                        "IDF1",
                        "IDSW",
                        "throughput_fps",
                        "deadline_miss_rate_source",
                        "process_cpu_percent_mean",
                        "sampled_peak_rss_mib",
                    ]
                },
                "work_p95_ms": s["frame_latency_ms"]["p95"],
                "work_p99_ms": s["frame_latency_ms"]["p99"],
                "capture_p95_ms": s["capture_to_output_latency_ms"]["p95"],
                "capture_p99_ms": s["capture_to_output_latency_ms"]["p99"],
                "worker_counts": wc,
                "scheduler": s["scheduler"],
                "deadline_accuracy": s["deadline_accuracy"],
            }
            rows.append(row)
            manifest["completed"].append(name)
            atomic_json(
                root / "summary.json",
                {"status": "running", "runs": rows, "planned_runs": len(plan), "completed_runs": len(rows)},
            )
            atomic_json(root / "manifest.json", manifest)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        atomic_json(root / "manifest.json", manifest)
        atomic_json(
            root / "summary.json",
            {
                "status": manifest["status"],
                "runs": rows,
                "planned_runs": len(plan),
                "completed_runs": len(rows),
            },
        )
        if rows:
            with (root / "summary.csv").open("w") as f:
                fields = [k for k, v in rows[0].items() if not isinstance(v, dict)]
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)


if __name__ == "__main__":
    main()
