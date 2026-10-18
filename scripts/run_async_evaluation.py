"""Full-sequence paced async policy evaluation; exports causal and deadline accuracy."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import cv2
from src.async_pipeline import AsyncTrackerRuntime, RuntimeConfig
from src.datasets.mot17 import MOT17Sequence
from src.detector import YOLODetector
from src.schedulers import FixedKPolicy, KalmanUncertaintyPolicy, AdaptiveHeuristicPolicy, OraclePolicy
from scripts.evaluate_tracking import atomic_json


def policy_for(name, length, k=3, threshold=0.01, schedule=None):
    if name == "fixed_k":
        return FixedKPolicy(k)
    if name == "kalman":
        return KalmanUncertaintyPolicy(threshold)
    if name == "heuristic":
        return AdaptiveHeuristicPolicy()
    if name == "oracle":
        frames = list(range(1, length + 1)) if schedule is None else schedule
        if any(f > length for f in frames):
            raise ValueError("Oracle frame exceeds sequence")
        return OraclePolicy(
            frames,
            offline=True,
            provenance=(
                "every-frame request reference; not optimized or a proven upper bound"
                if schedule is None
                else "user supplied offline schedule"
            ),
        )
    raise ValueError(f"Unknown policy: {name}")


def validate_export(root, length):
    summary = json.loads((root / "summary.json").read_text())
    rows = [json.loads(x) for x in (root / "frames.jsonl").read_text().splitlines()]
    if (
        summary["status"] != "complete"
        or not summary["ledger_complete"]
        or summary["frames"] != length
        or [r["frame"] for r in rows] != list(range(1, length + 1))
    ):
        raise ValueError("Incomplete causal frame ledger")
    if summary["worker_error"] or summary["main_error"]:
        raise ValueError("Runtime error")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--policy", choices=["fixed_k", "kalman", "heuristic", "oracle"], required=True)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--deadline-ms", type=float, default=33)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--oracle-schedule")
    p.add_argument("--eval-python", default=".venv-eval/bin/python")
    a = p.parse_args()
    root = Path(a.output)
    if root.exists():
        p.error("Output exists; preserve prior/partial run")
    seq = MOT17Sequence(a.sequence)
    model = Path(a.model).resolve()
    if a.warmup < 0 or a.threads < 1:
        p.error("Invalid warmup/threads")
    subprocess.run([a.eval_python, "-c", "import trackeval"], check=True)
    schedule = None if a.oracle_schedule is None else json.loads(Path(a.oracle_schedule).read_text())
    policy = policy_for(a.policy, len(seq), a.k, a.threshold, schedule)
    cv2.setNumThreads(a.threads)
    files = sorted(set(Path("src").rglob("*.py")) | {Path(__file__)})
    metadata = {
        "sequence": seq.metadata,
        "source": str(seq.path),
        "classes": [0],
        "model": str(model),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "code_sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        "ingestion": "paced replay at seqinfo camera rate; no input drops; backlog counted in capture latency",
        "warmup": a.warmup,
        "threads": a.threads,
        "repeat_count": 1,
    }
    warm = seq.read_frame(1).image

    def factory():
        detector = YOLODetector(model, threads=a.threads)
        for _ in range(a.warmup):
            detector(warm)
        return detector

    config = RuntimeConfig(
        frame_rate=seq.info.frame_rate,
        deadline_ms=a.deadline_ms,
        telemetry_frames=max(4096, len(seq) * 20),
        offline=a.policy == "oracle",
    )
    runtime = AsyncTrackerRuntime(factory, policy, config, metadata=metadata)
    try:
        runtime.start(timeout_s=60)
        for output in runtime.run(seq.images(), paced=True):
            if output.frame_idx % 100 == 0:
                print(f"{a.policy}: {output.frame_idx}/{len(seq)}", flush=True)
    finally:
        shutdown = runtime.close(timeout_s=30)
        if not shutdown.stopped:
            raise RuntimeError("Worker did not stop")
        runtime.export(root)
    summary = validate_export(root, len(seq))
    # Reuse evaluator without changing async telemetry or published predictions.
    # A separate deadline run retains the full ledger/GT, including late empty frames.
    deadline = root / "deadline-evaluation"
    deadline.mkdir()
    for name in ["summary.json", "frames.jsonl"]:
        (deadline / name).write_bytes((root / name).read_bytes())
    (deadline / "tracks.txt").write_bytes((root / "deadline_tracks.txt").read_bytes())
    for run in [root, deadline]:
        subprocess.run(
            [
                a.eval_python,
                "-m",
                "scripts.evaluate_tracking",
                "--sequence",
                str(seq.path),
                "--run-dir",
                str(run),
            ],
            check=True,
        )
    summary = json.loads((root / "summary.json").read_text())
    ds = json.loads((deadline / "summary.json").read_text())
    summary["deadline_accuracy"] = {key: ds[key] for key in ["HOTA", "MOTA", "IDF1", "IDSW"]}
    atomic_json(root / "summary.json", summary)
    atomic_json(
        root / "validation.json",
        {
            "status": "passed",
            "full_frames": len(seq),
            "ledger_complete": True,
            "worker_stopped": True,
            "causal_and_deadline_evaluated": True,
            "runtime_config": asdict(config),
        },
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in [
                    "HOTA",
                    "MOTA",
                    "throughput_fps",
                    "capture_to_output_latency_ms",
                    "deadline_miss_rate_source",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
