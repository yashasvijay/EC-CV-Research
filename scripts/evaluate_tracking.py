"""Evaluate one full MOT17 run through TrackEval and atomically enrich summary.json."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from src.datasets.mot17 import MOT17Sequence


def atomic_json(path, value):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_predictions(path, length):
    """Missing rows are legal (no objects); duplicate IDs or bad boxes are not."""
    seen = set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = np.array([float(value) for value in line.split(",")])
        except ValueError as exc:
            raise ValueError(f"Non-numeric prediction at line {line_number}") from exc
        if (
            len(row) != 10
            or not np.isfinite(row).all()
            or row[0] != int(row[0])
            or not 1 <= row[0] <= length
            or row[1] != int(row[1])
            or row[1] < 1
            or row[4] <= 0
            or row[5] <= 0
        ):
            raise ValueError(f"Invalid MOT prediction at line {line_number}")
        key = (int(row[0]), int(row[1]))
        if key in seen:
            raise ValueError(f"Duplicate frame/track ID: {key}")
        seen.add(key)


def evaluate_run(sequence_path, run_dir):
    """Scores use percentage points (MOTA may be negative); IDSW is a count.

    Per-run FPS/latency excludes this offline evaluation. Failed evaluations
    leave summary.json unchanged. MOT17 distractor preprocessing stays enabled.
    """
    try:
        import trackeval
    except ImportError as exc:
        raise RuntimeError("Install evaluation dependencies: pip install -r requirements-eval.txt") from exc
    sequence = MOT17Sequence(sequence_path)
    if not sequence.ground_truth_path.is_file():
        raise FileNotFoundError(f"Ground truth required: {sequence.ground_truth_path}")
    run_dir = Path(run_dir).resolve()
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "complete" or summary.get("frames") != len(sequence):
        raise ValueError("Evaluation requires a complete run over the entire sequence")
    metadata = summary.get("metadata", {})
    recorded_sequence = metadata.get("sequence")
    if recorded_sequence and recorded_sequence.get("name") != sequence.info.name:
        raise ValueError("Run sequence does not match evaluation ground truth")
    source = metadata.get("source")
    if source and Path(source).resolve() not in (sequence.path, sequence.image_dir):
        raise ValueError("Run source does not match the supplied MOT17 sequence")
    if metadata.get("classes") != [0]:
        raise ValueError("MOT17 pedestrian evaluation requires COCO person-only --classes 0")
    # tracks.txt alone cannot distinguish an unprocessed frame from an empty one.
    count = 0
    with (run_dir / "frames.jsonl").open() as stream:
        for count, line in enumerate(stream, 1):
            if json.loads(line).get("frame") != count:
                raise ValueError("Frame ledger is not contiguous from frame 1")
    if count != len(sequence):
        raise ValueError("Frame ledger does not cover the full sequence")
    tracks_path = run_dir / "tracks.txt"
    validate_predictions(tracks_path, len(sequence))
    evaluator_config = trackeval.Evaluator.get_default_eval_config()
    evaluator_config.update(
        {
            "USE_PARALLEL": False,
            "PRINT_RESULTS": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "DISPLAY_LESS_PROGRESS": True,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": False,
            "BREAK_ON_ERROR": True,
            "RETURN_ON_ERROR": False,
            "LOG_ON_ERROR": str(run_dir / "trackeval-error.log"),
        }
    )
    with tempfile.TemporaryDirectory(prefix="mot17-eval-") as temporary:
        staged = Path(temporary) / "tracker" / "data"
        staged.mkdir(parents=True)
        shutil.copyfile(tracks_path, staged / f"{sequence.info.name}.txt")
        config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        config.update(
            {
                "GT_FOLDER": str(sequence.path.parent),
                "TRACKERS_FOLDER": temporary,
                "SKIP_SPLIT_FOL": True,
                "BENCHMARK": "MOT17",
                "SPLIT_TO_EVAL": "train",
                "TRACKERS_TO_EVAL": ["tracker"],
                "CLASSES_TO_EVAL": ["pedestrian"],
                "SEQ_INFO": {sequence.info.name: len(sequence)},
                "DO_PREPROC": True,
                "PRINT_CONFIG": False,
                "OUTPUT_FOLDER": str(run_dir / "trackeval"),
            }
        )
        dataset = trackeval.datasets.MotChallenge2DBox(config)
        metric_config = {"PRINT_CONFIG": False}
        metrics = [
            trackeval.metrics.HOTA(metric_config),
            trackeval.metrics.CLEAR(metric_config),
            trackeval.metrics.Identity(metric_config),
        ]
        results, messages = trackeval.Evaluator(evaluator_config).evaluate([dataset], metrics)
        name = dataset.get_name()
        if messages[name]["tracker"] != "Success":
            raise RuntimeError(f"TrackEval failed: {messages}")
        scores = results[name]["tracker"][sequence.info.name]["pedestrian"]
    extracted = {
        "HOTA": 100 * float(np.mean(scores["HOTA"]["HOTA"])),
        "MOTA": 100 * float(scores["CLEAR"]["MOTA"]),
        "IDF1": 100 * float(scores["Identity"]["IDF1"]),
        "IDSW": int(scores["CLEAR"]["IDSW"]),
    }
    if not all(np.isfinite(value) for value in extracted.values()):
        raise ValueError("TrackEval returned non-finite metrics")
    summary.update(extracted)
    summary.update(
        accuracy_status="evaluated",
        evaluation={
            "evaluator": "TrackEval",
            "version": version("trackeval"),
            "sequence": sequence.info.name,
            "class": "pedestrian",
            "preprocessing": True,
            "metric_units": {"HOTA": "percent", "MOTA": "percent", "IDF1": "percent", "IDSW": "count"},
            "hota_aggregation": "mean over TrackEval localization thresholds",
            "gt_sha256": hashlib.sha256(sequence.ground_truth_path.read_bytes()).hexdigest(),
            "tracks_sha256": hashlib.sha256(tracks_path.read_bytes()).hexdigest(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    atomic_json(summary_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, help="MOT17 sequence containing seqinfo.ini and gt/")
    parser.add_argument(
        "--run-dir", required=True, help="directory with tracks.txt, frames.jsonl, summary.json"
    )
    args = parser.parse_args()
    result = evaluate_run(args.sequence, args.run_dir)
    print(json.dumps({key: result[key] for key in ("HOTA", "MOTA", "IDF1", "IDSW")}, indent=2))


if __name__ == "__main__":
    main()
