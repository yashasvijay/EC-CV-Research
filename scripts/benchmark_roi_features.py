import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import cv2
import numpy as np
import psutil

from src.datasets.mot17 import MOT17Sequence
from src.features import select_roi_features
from scripts.profile_pipeline import running_searches

ROOT = Path(__file__).resolve().parents[1]


def legacy_features(gray, box, max_points):
    mask = np.zeros_like(gray)
    x1, y1, x2, y2 = box.astype(int)
    mask[y1:y2, x1:x2] = 255
    points = cv2.goodFeaturesToTrack(gray, max_points, 0.01, 5, mask=mask)
    return points if points is not None else np.empty((0, 1, 2), np.float32)


def load_boxes(path, width, height):
    boxes = defaultdict(list)
    for line in path.read_text().splitlines():
        fields = [float(v) for v in line.split(",")]
        if len(fields) != 10 or not np.isfinite(fields).all():
            raise ValueError("Expected finite ten-column MOT predictions")
        frame, _, x, y, w, h, *_ = fields
        if frame != int(frame) or frame < 1 or min(w, h) <= 0:
            raise ValueError("Invalid frame or box in MOT predictions")
        box = np.array([x - 1, y - 1, x - 1 + w, y - 1 + h], np.float32)
        if np.any(box[:2] < 0) or box[2] > width + 0.001 or box[3] > height + 0.001:
            raise ValueError("Comparison workload must contain image-clipped boxes")
        box[[0, 2]] = box[[0, 2]].clip(0, width)
        box[[1, 3]] = box[[1, 3]].clip(0, height)
        boxes[int(frame)].append(box)
    return boxes


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Paired full-frame-mask versus cropped-ROI reseeding attribution"
    )
    parser.add_argument("--sequence", type=Path, default=ROOT / "data/raw/MOT17/train/MOT17-02-FRCNN")
    parser.add_argument("--tracks", type=Path, default=ROOT / "reports/profiling/tracks.txt")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/roi-features/paired")
    args = parser.parse_args(argv)
    if min(args.frames, args.repeats, args.threads) < 1 or args.warmup < 0:
        parser.error("frames/repeats/threads must be positive; warmup must be nonnegative")
    if running_searches(ROOT / "results"):
        parser.error("A grid search is marked running; comparison is deferred")
    if args.output.exists():
        parser.error("Output exists; choose a new directory")
    sequence = MOT17Sequence(args.sequence)
    if args.frames > len(sequence):
        parser.error("Requested frames exceed sequence length")
    boxes = load_boxes(args.tracks, sequence.info.width, sequence.info.height)
    if not any(boxes.get(i) for i in range(1, args.frames + 1)):
        parser.error("No boxes in the selected source interval")
    cv2.setNumThreads(args.threads)
    backends = {"full_frame_mask": legacy_features, "cropped_roi_mask": select_roi_features}
    first_id = next(i for i in range(1, args.frames + 1) if boxes.get(i))
    first = cv2.cvtColor(sequence.read_frame(first_id).image, cv2.COLOR_BGR2GRAY)
    for _ in range(args.warmup):
        for function in backends.values():
            function(first, boxes[first_id][0], 60)
    args.output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    process.cpu_percent(None)
    start, cpu_start = perf_counter(), sum(process.cpu_times()[:2])
    rows, peak_rss = [], 0.0
    status = "partial"
    try:
        for repeat in range(args.repeats):
            for frame_idx in range(1, args.frames + 1):
                gray = cv2.cvtColor(sequence.read_frame(frame_idx).image, cv2.COLOR_BGR2GRAY)
                order = list(backends)
                if (repeat + frame_idx) % 2:
                    order.reverse()
                for name in order:
                    function = backends[name]
                    before = perf_counter()
                    points = [function(gray, box, 60) for box in boxes.get(frame_idx, ())]
                    latency = (perf_counter() - before) * 1000
                    rows.append(
                        {
                            "repeat": repeat + 1,
                            "frame": frame_idx,
                            "backend": name,
                            "seed_ms": latency,
                            "boxes": len(points),
                            "corners": sum(len(p) for p in points),
                        }
                    )
                peak_rss = max(peak_rss, process.memory_info().rss / 1024**2)
            print(f"Completed paired repeat {repeat + 1}/{args.repeats}", flush=True)
        status = "complete"
    finally:
        elapsed = perf_counter() - start
        aggregates = {}
        for name in backends:
            selected = [r for r in rows if r["backend"] == name]
            if not selected:
                continue
            values = [r["seed_ms"] for r in selected]
            means = [
                float(np.mean([r["seed_ms"] for r in selected if r["repeat"] == i]))
                for i in sorted({r["repeat"] for r in selected})
            ]
            aggregates[name] = {
                "mean_ms_per_frame": float(np.mean(values)),
                "p95_ms": float(np.percentile(values, 95)),
                "p99_ms": float(np.percentile(values, 99)),
                "repeat_means_ms": means,
                "repeat_mean_sd_ms": float(np.std(means, ddof=1)) if len(means) > 1 else None,
                "calls": sum(r["boxes"] for r in selected),
                "corners": sum(r["corners"] for r in selected),
            }
        summary = {
            "status": status,
            "frames_per_repeat": args.frames,
            "repeats": args.repeats,
            "sequence": sequence.metadata,
            "threads": args.threads,
            "platform": platform.platform(),
            "packages": {"opencv": cv2.__version__, "numpy": np.__version__},
            "workload_sha256": hashlib.sha256(args.tracks.read_bytes()).hexdigest(),
            "feature_source_sha256": hashlib.sha256((ROOT / "src/features.py").read_bytes()).hexdigest(),
            "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "timing": "paired per-frame reseeding including masks/offsets; decode and logging excluded; no cProfile",
            "order": "alternates by repeat and frame",
            "aggregates": aggregates,
            "wall_seconds": elapsed,
            "sampled_peak_rss_mib": peak_rss,
            "process_cpu_percent_mean": 100 * (sum(process.cpu_times()[:2]) - cpu_start) / elapsed,
            "cpu_window": "includes both backends, decode, and benchmark overhead; 100% = one logical core",
            "HOTA": None,
            "MOTA": None,
            "accuracy_status": "not evaluated; corner populations may differ",
        }
        if status == "complete":
            old = aggregates["full_frame_mask"]["mean_ms_per_frame"]
            new = aggregates["cropped_roi_mask"]["mean_ms_per_frame"]
            summary.update(reseeding_speedup=old / new, latency_reduction_percent=100 * (1 - new / old))
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        with (args.output / "frames.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
