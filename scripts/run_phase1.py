"""Run one mode per process, with identical input frames and detector warmup."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from importlib.metadata import version
import cv2
from src.detector import YOLODetector
from src.pipeline import Pipeline, SortPipeline
from src.benchmark import Benchmark
from src.datasets.mot17 import MOT17Sequence


def frames(source):
    path = Path(source)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not files:
            raise ValueError(f"No images in {path}")
        for p in files:
            frame = cv2.imread(str(p))
            if frame is None:
                raise ValueError(f"Cannot decode {p}")
            yield frame
    else:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open {source}")
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame
        finally:
            capture.release()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True, help="video, image directory, or MOT17 sequence directory")
    p.add_argument("--output", required=True, help="new directory, never overwritten")
    p.add_argument("--mode", choices=["baseline", "adaptive", "fixed"], required=True)
    p.add_argument("--tracker", choices=["lk", "sort"], default="lk")
    p.add_argument("--max-age", type=int, default=30, help="SORT lifetime in video frames")
    p.add_argument("--min-hits", type=int, default=1)
    p.add_argument("--match-iou", type=float, default=0.3)
    p.add_argument("--max-interval", type=int, default=10)
    p.add_argument("--min-confidence", type=float, default=0.6)
    p.add_argument("--max-displacement", type=float, default=0.15)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=0, help="0 means entire sequence")
    p.add_argument("--classes", type=int, nargs="+", default=[0])
    p.add_argument("--num-classes", type=int, default=80)
    args = p.parse_args()
    if args.threads < 1 or args.warmup < 0 or args.max_frames < 0:
        p.error("threads must be positive; warmup/max-frames must be nonnegative")
    if not 0 <= args.min_confidence <= 1 or args.max_displacement <= 0:
        p.error("Invalid gate thresholds")
    if args.max_interval < 1 or args.max_age < 0 or args.min_hits < 1 or not 0 < args.match_iou <= 1:
        p.error("Invalid interval or SORT parameters")
    if args.tracker == "sort" and args.mode == "adaptive":
        p.error("Adaptive confidence gating currently requires --tracker lk")
    source_path = Path(args.source).resolve()
    sequence_path = source_path if (source_path / "seqinfo.ini").is_file() else source_path.parent
    sequence = MOT17Sequence(sequence_path) if (sequence_path / "seqinfo.ini").is_file() else None
    make_frames = sequence.images if sequence else lambda: frames(args.source)
    cv2.setNumThreads(args.threads)
    detector = YOLODetector(
        args.model, classes=args.classes, num_classes=args.num_classes, threads=args.threads
    )
    warm_frames = make_frames()
    try:
        first = next(warm_frames)
    except StopIteration:
        raise ValueError("Source has no decodable frames") from None
    finally:
        warm_frames.close()
    for _ in range(args.warmup):
        detector(first)
    if args.tracker == "sort":
        pipeline = SortPipeline(
            detector, args.mode, args.max_interval, args.max_age, args.min_hits, args.match_iou
        )
    else:
        pipeline = Pipeline(
            detector, args.mode, args.max_interval, args.min_confidence, args.max_displacement
        )
    metadata = vars(args) | {
        "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
        "packages": {
            name: version(name)
            for name in ("numpy", "opencv-python-headless", "onnxruntime", "scipy", "psutil")
        },
        "providers": detector.session.get_providers(),
        "cpu_convention": "100% = one logical core; may exceed 100%",
        "timing": "pipeline excludes decode/export; throughput includes decode, MOT/JSONL writes and instrumentation",
        "tracker_description": (
            "8-state SORT variant" if args.tracker == "sort" else "translation LK + class-aware Hungarian IoU"
        ),
        "sequence": sequence.metadata if sequence else None,
    }
    benchmark = Benchmark(args.output, metadata)
    source = make_frames()
    benchmark.metadata["status"] = "partial"
    try:
        with (Path(args.output) / "tracks.txt").open("w") as mot:
            while not args.max_frames or pipeline.frame_id < args.max_frames:
                start = perf_counter()
                try:
                    frame = next(source)
                except StopIteration:
                    break
                decode_ms = (perf_counter() - start) * 1000
                tracks, row = pipeline.step(frame)
                for t in tracks:
                    x, y, x2, y2 = t.box
                    # MOTChallenge uses one-based image coordinates; width/height unchanged.
                    mot.write(
                        f"{row['frame']},{t.id},{x + 1:.3f},{y + 1:.3f},{x2 - x:.3f},{y2 - y:.3f},{t.score:.5f},-1,-1,-1\n"
                    )
                row.update(
                    decode_ms=decode_ms, tracks=len(tracks), frame_work_ms=(perf_counter() - start) * 1000
                )
                benchmark.record(row)
        benchmark.metadata["status"] = "complete"
    finally:
        source.close()
        summary = benchmark.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
