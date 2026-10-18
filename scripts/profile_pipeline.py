"""Profile the unchanged synchronous LK baseline in an isolated process."""

import argparse
from collections import defaultdict
from contextlib import contextmanager, ExitStack
import cProfile
from functools import wraps
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import pstats
from time import perf_counter
from unittest.mock import patch

import cv2
import numpy as np

from src.benchmark import Benchmark
from src.datasets.mot17 import MOT17Sequence
from src.detector import YOLODetector
from src.pipeline import Pipeline
from src import tracker as tracker_module

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "yolo_inference": "YOLO ONNX session.run",
    "detector_other": "Detector preprocessing/postprocessing",
    "lk_flow": "OpenCV LK (forward + backward)",
    "flow_other": "Flow filtering/motion/state/copy",
    "feature_selection": "OpenCV goodFeaturesToTrack",
    "iou": "Scalar IoU calls",
    "hungarian": "Hungarian assignment solve",
    "update_other": "Update loops/masks/lifecycle/reseed setup",
    "grayscale": "Grayscale conversion",
    "pipeline_other": "Pipeline orchestration/timing overhead",
}


class StageTimer:
    def __init__(self, clock=perf_counter):
        self.clock = clock
        self.stack = []
        self.reset()

    def reset(self):
        if self.stack:
            raise RuntimeError("Cannot reset active timing spans")
        self.inclusive = defaultdict(float)
        self.exclusive = defaultdict(float)
        self.calls = defaultdict(int)

    @contextmanager
    def span(self, name):
        entry = [name, self.clock(), 0.0]
        self.stack.append(entry)
        try:
            yield
        finally:
            elapsed = self.clock() - entry[1]
            if self.stack.pop() is not entry:
                raise RuntimeError("Timing spans exited out of order")
            self.inclusive[name] += elapsed * 1000
            self.exclusive[name] += (elapsed - entry[2]) * 1000
            self.calls[name] += 1
            if self.stack:
                self.stack[-1][2] += elapsed

    def wrap(self, name, function):
        @wraps(function)
        def measured(*args, **kwargs):
            with self.span(name):
                return function(*args, **kwargs)

        return measured


class SessionProxy:
    def __init__(self, session, timer):
        self.original = session
        self.run = timer.wrap("yolo_inference", session.run)

    def __getattr__(self, name):
        return getattr(self.original, name)


class TimedDetector:
    def __init__(self, detector, timer):
        self.detector, self.timer = detector, timer

    def __call__(self, frame):
        with self.timer.span("detector_other"):
            return self.detector(frame)


@contextmanager
def instrument(pipeline, detector, timer):
    # Patches exist only in this single-threaded profiling process.
    with ExitStack() as stack:
        for owner, attribute, name in (
            (cv2, "calcOpticalFlowPyrLK", "lk_flow"),
            (cv2, "goodFeaturesToTrack", "feature_selection"),
            (cv2, "cvtColor", "grayscale"),
            (tracker_module, "box_iou", "iou"),
            (tracker_module, "linear_sum_assignment", "hungarian"),
            (pipeline.tracker, "predict", "flow_other"),
            (pipeline.tracker, "update", "update_other"),
            (pipeline, "step", "pipeline_other"),
        ):
            stack.enter_context(patch.object(owner, attribute, timer.wrap(name, getattr(owner, attribute))))
        stack.enter_context(patch.object(detector, "session", SessionProxy(detector.session, timer)))
        stack.enter_context(patch.object(pipeline, "detector", TimedDetector(detector, timer)))
        yield


def running_searches(root):
    active = []
    for path in Path(root).glob("**/manifest.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if data.get("status") == "running" and "max_intervals" in data.get("config", {}):
            active.append(str(path.resolve()))
    return active


def quantiles(values):
    return {
        label: float(np.percentile(values, value)) for label, value in (("p50", 50), ("p95", 95), ("p99", 99))
    }


def summarize_stages(rows):
    pipeline_total = sum(r["inclusive_stage_ms"]["pipeline_other"] for r in rows)
    output = {}
    for name in STAGES:
        values = [r["exclusive_stage_ms"].get(name, 0.0) for r in rows]
        inclusive = sum(r["inclusive_stage_ms"].get(name, 0.0) for r in rows)
        calls = sum(r["stage_calls"].get(name, 0) for r in rows)
        total = sum(values)
        output[name] = {
            "label": STAGES[name],
            "calls": calls,
            "total_exclusive_ms": total,
            "mean_exclusive_ms_per_frame": total / len(rows),
            "pipeline_percent": 100 * total / pipeline_total if pipeline_total else 0.0,
            "mean_inclusive_ms_per_call": inclusive / calls if calls else None,
            "mean_inclusive_ms_per_frame": inclusive / len(rows),
            "exclusive_per_frame_quantiles_ms": quantiles(values),
        }
    measured = sum(v["total_exclusive_ms"] for v in output.values())
    if not np.isclose(measured, pipeline_total, rtol=1e-9, atol=1e-6):
        raise RuntimeError("Exclusive stage accounting does not reconcile")
    return output


def format_breakdown(summary, stages):
    lines = [
        f"Synchronous baseline profile: {summary['frames']} frames",
        f"Status: {summary['status']} | Timing mode: {summary['metadata']['timing_mode']}",
        "Exclusive costs below partition measured pipeline time (decode/export excluded).",
        "Stage                                           ms/frame     share       calls",
        "-" * 80,
    ]
    for stage in sorted(stages.values(), key=lambda s: -s["total_exclusive_ms"]):
        lines.append(
            f"{stage['label']:<47} {stage['mean_exclusive_ms_per_frame']:9.3f}"
            f" {stage['pipeline_percent']:8.2f}% {stage['calls']:11d}"
        )
    latency = summary["pipeline_latency_ms"]
    lines.extend(
        [
            "-" * 80,
            f"Pipeline mean/p95/p99: {latency['mean']:.3f} / {latency['p95']:.3f} / {latency['p99']:.3f} ms",
            "Inclusive parent spans (nested; do not add these to the partition):",
        ]
    )
    for name in ("detector_other", "flow_other", "update_other"):
        value = stages[name]["mean_inclusive_ms_per_frame"]
        lines.append(f"  {name}: {value:.3f} ms/frame")
    lines.extend(
        [
            f"Decoded-frame work mean: {summary['frame_work_latency_ms']['mean']:.3f} ms",
            f"Observed throughput (includes export/logging): {summary['throughput_fps']:.3f} FPS",
            f"CPU: {summary['process_cpu_percent_mean']:.2f}% (100% = one logical core)",
            f"Sampled peak RSS: {summary['sampled_peak_rss_mib']:.2f} MiB",
            "Costs include timer overhead and, by default, cProfile overhead; these are attribution measurements.",
            "IoU-loop control, full-image masks and feature-reseed setup remain in update_other.",
            "HOTA/MOTA/IDF1/IDSW are unevaluated for this short prefix; no fidelity claim is made.",
        ]
    )
    if summary["metadata"]["active_grid_searches"]:
        lines.append(
            "CONCURRENT DIAGNOSTIC: active-search manifests were found; timing may be affected by other work."
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolo11n.onnx")
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/MOT17/train/MOT17-02-FRCNN")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/profiling")
    parser.add_argument("--frames", "--max-frames", dest="frames", type=int, default=50)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--allow-concurrent",
        action="store_true",
        help="explicitly permit diagnostic timing while a grid search is marked running",
    )
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help="companion run with phase timers but no cProfile or .prof dump",
    )
    parser.add_argument("--search-root", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    if min(args.frames, args.start_frame, args.threads) < 1 or args.warmup < 0:
        parser.error("frames/start-frame/threads must be positive; warmup must be nonnegative")
    active = running_searches(args.search_root)
    if active and not args.allow_concurrent:
        parser.error(
            "A grid search is marked running. Wait for completion or use --allow-concurrent "
            "for an explicitly labelled diagnostic. No inference was started."
        )
    if args.output.exists():
        parser.error("Output directory already exists; choose a new directory to preserve earlier profiles")
    source = args.source.resolve()
    if not (source / "seqinfo.ini").is_file():
        source = source.parent
    sequence = MOT17Sequence(source)
    last_frame = args.start_frame + args.frames - 1
    if last_frame > len(sequence):
        parser.error("Requested segment exceeds sequence length")
    cv2.setNumThreads(args.threads)
    detector = YOLODetector(args.model, threads=args.threads)
    first = sequence.read_frame(args.start_frame).image
    for _ in range(args.warmup):
        detector(first)
    pipeline, timer = Pipeline(detector, mode="baseline"), StageTimer()
    profile = cProfile.Profile()
    source_files = (
        "scripts/profile_pipeline.py",
        "src/pipeline.py",
        "src/tracker.py",
        "src/features.py",
        "src/detector.py",
        "src/benchmark.py",
        "src/datasets/mot17.py",
    )
    metadata = {
        "mode": "baseline",
        "tracker": "lk",
        "sequence": sequence.metadata,
        "source_frame_range": [args.start_frame, last_frame],
        "warmup": args.warmup,
        "threads": args.threads,
        "classes": [0],
        "num_classes": 80,
        "model": str(args.model.resolve()),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "code_sha256": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in source_files},
        "packages": {
            p: version(p) for p in ("numpy", "opencv-python-headless", "onnxruntime", "scipy", "psutil")
        },
        "providers": detector.session.get_providers(),
        "active_grid_searches": active,
        "timing_mode": "phase_timers_only" if args.timing_only else "cprofile_plus_phase_timers",
        "timing": "pipeline excludes decode/export/logging; all measured phases include instrumentation overhead",
        "status": "partial",
    }
    benchmark = Benchmark(args.output, metadata)
    rows = []
    error = None
    try:
        with instrument(pipeline, detector, timer), (args.output / "tracks.txt").open("w") as tracks_file:
            for frame_id in range(args.start_frame, last_frame + 1):
                timer.reset()
                if not args.timing_only:
                    profile.enable()
                try:
                    with timer.span("frame_other"):
                        with timer.span("decode"):
                            packet = sequence.read_frame(frame_id)
                        tracks, row = pipeline.step(packet.image)
                finally:
                    profile.disable()
                row.update(
                    source_frame=frame_id,
                    tracks=len(tracks),
                    pipeline_ms=timer.inclusive["pipeline_other"],
                    frame_work_ms=timer.inclusive["frame_other"],
                    decode_ms=timer.inclusive["decode"],
                    inclusive_stage_ms=dict(timer.inclusive),
                    exclusive_stage_ms=dict(timer.exclusive),
                    stage_calls=dict(timer.calls),
                )
                for t in tracks:
                    x1, y1, x2, y2 = t.box
                    tracks_file.write(
                        f"{frame_id},{t.id},{x1 + 1:.3f},{y1 + 1:.3f},"
                        f"{x2 - x1:.3f},{y2 - y1:.3f},{t.score:.5f},-1,-1,-1\n"
                    )
                benchmark.record(row)
                rows.append(row)
                if len(rows) % 10 == 0:
                    print(f"Profiled {len(rows)}/{args.frames} frames", flush=True)
        metadata["status"] = "complete"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        metadata["error"] = error
        raise
    finally:
        profile.disable()
        summary = benchmark.finish()
        if not args.timing_only:
            profile_path = args.output / "baseline_profile.prof"
            profile.dump_stats(str(profile_path))
            summary["profile_file"] = profile_path.name
            stream = io.StringIO()
            pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(30)
            (args.output / "cprofile_top.txt").write_text(stream.getvalue())
        else:
            summary["profile_file"] = None
        if rows:
            stages = summarize_stages(rows)
            summary["stage_breakdown"] = stages
            summary["deadline_miss_rate_pipeline_work"] = {
                str(b): sum(r["pipeline_ms"] > b for r in rows) / len(rows) for b in (33, 50)
            }
            summary["accuracy_status"] = "not_evaluated; short source segment, not a full-sequence result"
            text = format_breakdown(summary, stages)
            (args.output / "breakdown.txt").write_text(text)
            print(text)
        summary["error"] = error
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
