"""Paced synthetic latency-injection sweep; not a YOLO speed or MOT accuracy benchmark."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
import platform
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np

from scripts.evaluate_tracking import atomic_json
from scripts.verify_slos import ROOT, assess, digest, idle_check, load_contract, write_report
from src.async_pipeline import AsyncTrackerRuntime, RuntimeConfig
from src.schedulers import FixedKPolicy


class SyntheticScene:
    """Deterministic moving textured boxes with source ID encoded outside their ROIs."""

    def __init__(self, width: int, height: int, tracks: int, seed: int):
        self.width, self.height, self.tracks = width, height, tracks
        cols = math.ceil(math.sqrt(tracks))
        rows = math.ceil(tracks / cols)
        cell_width, cell_height = width // cols, height // rows
        if min(cell_width, cell_height) < 32:
            raise ValueError("Use larger images or fewer tracks (minimum 32-pixel cells)")
        rng = np.random.default_rng(seed)
        self.patches = [
            rng.integers(0, 256, (cell_height // 2, cell_width // 2, 3), dtype=np.uint8)
            for _ in range(tracks)
        ]
        self.origins = [
            (i % cols * cell_width + cell_width // 4, i // cols * cell_height + cell_height // 4)
            for i in range(tracks)
        ]

    def boxes(self, index: int) -> np.ndarray:
        """Return exact synthetic source boxes; this reference has no learned detector error."""
        shift = int(3 * math.sin(index / 15))
        rows = []
        for (x, y), patch in zip(self.origins, self.patches):
            h, w = patch.shape[:2]
            rows.append([x + shift, y, x + shift + w, y + h, 0.9, 0])
        return np.asarray(rows, dtype=np.float32)

    def frames(self, count: int):
        """Generate source images; generation time is included relative to nominal release."""
        for i in range(count):
            image = np.zeros((self.height, self.width, 3), np.uint8)
            for box, patch in zip(self.boxes(i), self.patches):
                x, y, x2, y2 = box[:4].astype(int)
                image[y:y2, x:x2] = patch
            image[0, 0] = [i & 255, (i >> 8) & 255, (i >> 16) & 255]
            yield image


class LatencyDetector:
    """Impose a minimum total detector-call wall time, including bootstrap calls."""

    def __init__(self, scene: SyntheticScene, latency_ms: float):
        self.scene, self.latency_ms = scene, latency_ms

    def __call__(self, image: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        pixel = image[0, 0].astype(int)
        index = int(pixel[0] + (pixel[1] << 8) + (pixel[2] << 16))
        detections = self.scene.boxes(index)
        time.sleep(max(0.0, self.latency_ms / 1000 - (time.perf_counter() - started)))
        return detections


def run_case(
    output: Path,
    latency_ms: float,
    *,
    frames: int,
    width: int,
    height: int,
    tracks: int,
    seed: int,
    k: int,
    threads: int,
    design: Path,
) -> dict[str, Any]:
    """Run one fresh-process case and retain raw telemetry even when its SLOs fail."""
    if output.exists():
        raise ValueError("Case output already exists")
    contract = load_contract(design)
    scene = SyntheticScene(width, height, tracks, seed)
    cv2.setNumThreads(threads)
    config = RuntimeConfig(
        frame_rate=contract["camera_fps"],
        deadline_ms=contract["deadline_ms"],
        telemetry_frames=max(4096, frames * 20),
    )
    metadata = {
        "workload": "synthetic moving textured boxes; exact source detections",
        "scope": "sleep simulates detector service delay; not CPU/power scaling, learned accuracy or hardware equivalence",
        "injected_detector_min_ms": latency_ms,
        "source_frames": frames,
        "width": width,
        "height": height,
        "synthetic_objects": tracks,
        "seed": seed,
        "threads": threads,
        "bootstrap_injection": True,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {"opencv": cv2.__version__, "numpy": np.__version__},
        "code_sha256": {str(f.relative_to(ROOT)): digest(f) for f in ROOT.joinpath("src").rglob("*.py")},
        "driver_sha256": digest(Path(__file__)),
        "config": asdict(config),
    }
    runtime = AsyncTrackerRuntime(
        lambda: LatencyDetector(scene, latency_ms), FixedKPolicy(k), config, metadata=metadata
    )
    output.mkdir(parents=True)
    try:
        runtime.start()
        for _ in runtime.run(scene.frames(frames), paced=True):
            pass
    finally:
        shutdown = runtime.close(timeout_s=30)
        if not shutdown.stopped:
            raise RuntimeError("Detector worker did not stop")
        runtime.export(output / "runtime")
    summary = json.loads((output / "runtime/summary.json").read_text())
    rows = [json.loads(line) for line in (output / "runtime/frames.jsonl").read_text().splitlines()]
    report = assess(summary, rows, contract)
    report.update(contract=contract, design_sha256=digest(design), workload=metadata)
    write_report(output, report)
    calls = summary["counts"].get("bootstrap_calls", 0)
    row = {
        "injected_detector_min_ms": latency_ms,
        "frames": frames,
        "slo_status": report["status"],
        "track_frame_fraction": report["checks"]["track_frame_fraction"].get("measured"),
        "useful_fusion_ratio": report["checks"]["useful_fusion_ratio"].get("measured"),
        "deadline_miss_fraction": report["checks"]["deadline_miss_fraction"].get("measured"),
        "throughput_fps": report["checks"]["throughput_fps"].get("measured"),
        "p95_e2e_ms": report["checks"]["p95_e2e_ms"].get("measured"),
        "p99_e2e_ms": report["checks"]["p99_e2e_ms"].get("measured"),
        "fallback_attempts": calls,
        "fallback_attempt_fraction": calls / frames,
        "fallback_attempts_per_second": calls / summary["wall_seconds"],
        "bootstrap_cooldown_frames": summary["counts"].get("bootstrap_cooldown_frames", 0),
        "observed_service_mean_ms": (summary.get("detector_service_ms") or {}).get("mean"),
        "requests_superseded": summary["counts"].get("requests_superseded", 0),
        "process_cpu_percent_mean": summary["process_cpu_percent_mean"],
        "sampled_peak_rss_mib": summary["sampled_peak_rss_mib"],
        "run_dir": str(output),
    }
    atomic_json(output / "case.json", row)
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latencies-ms", nargs="+", type=float, default=[40, 80, 160, 320])
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--tracks", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--design", type=Path, default=ROOT / "docs/DESIGN.md")
    parser.add_argument("--case-latency-ms", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if (
            min(args.frames, args.width, args.height, args.tracks, args.repeats, args.k, args.threads) < 1
            or args.frames >= 2**24
        ):
            raise ValueError("Invalid frame/dimension/track/repeat parameters")
        latencies = args.latencies_ms if args.case_latency_ms is None else [args.case_latency_ms]
        if (
            not latencies
            or len(set(latencies)) != len(latencies)
            or any(not math.isfinite(v) or v < 0 for v in latencies)
        ):
            raise ValueError("Use unique finite nonnegative latency values")
        load_contract(args.design)
        SyntheticScene(args.width, args.height, args.tracks, args.seed)
        if args.output.exists():
            raise ValueError("Output exists; partial/completed runs are preserved")
        if args.case_latency_ms is not None:
            run_case(
                args.output,
                args.case_latency_ms,
                frames=args.frames,
                width=args.width,
                height=args.height,
                tracks=args.tracks,
                seed=args.seed,
                k=args.k,
                threads=args.threads,
                design=args.design,
            )
            return 0
        args.output.mkdir(parents=True)
        plan = [(delay, repeat) for repeat in range(1, args.repeats + 1) for delay in latencies]
        random.Random(args.seed).shuffle(plan)
        manifest = {
            "status": "running",
            "workload": "synthetic latency injection; not real detector hardware",
            "plan": plan,
            "completed": [],
            "driver_sha256": digest(Path(__file__)),
        }
        atomic_json(args.output / "manifest.json", manifest)
        rows = []
        try:
            for delay, repeat in plan:
                preflight = idle_check()
                name = f"latency-{delay:g}ms-r{repeat:02d}"
                atomic_json(args.output / (name + "-preflight.json"), preflight)
                if not preflight["passed"]:
                    manifest.update(status="deferred", reason="Host is not idle; no new case started")
                    return 2
                command = [
                    sys.executable,
                    "-m",
                    "scripts.sweep_detector_latency",
                    "--case-latency-ms",
                    str(delay),
                    "--output",
                    str(args.output / name),
                    "--frames",
                    str(args.frames),
                    "--width",
                    str(args.width),
                    "--height",
                    str(args.height),
                    "--tracks",
                    str(args.tracks),
                    "--seed",
                    str(args.seed),
                    "--k",
                    str(args.k),
                    "--threads",
                    str(args.threads),
                    "--design",
                    str(args.design),
                ]
                with (args.output / (name + ".log")).open("w") as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                rows.append(json.loads((args.output / name / "case.json").read_text()) | {"repeat": repeat})
                manifest["completed"].append(name)
            manifest["status"] = "complete"
        except BaseException as exc:
            manifest.update(status="failed", error=str(exc))
            raise
        finally:
            atomic_json(args.output / "manifest.json", manifest)
            atomic_json(
                args.output / "summary.json",
                {
                    "status": manifest["status"],
                    "runs": rows,
                    "definition": "useful_fusion_ratio counts background runs with >=1 accepted correction, not individual boxes",
                    "scope": "synthetic exact detections plus wall-delay injection; not HOTA/MOTA, energy or hardware speed equivalence",
                },
            )
            if rows:
                with (args.output / "summary.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
