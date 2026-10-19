"""Unit and real TrackEval integration checks on synthetic MOT17 fixtures.

Evaluation tests require .venv-eval, or set EVAL_PYTHON to a compatible Python.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from src.datasets.mot17 import MOT17Dataset, MOT17Sequence
from src.pipeline import SortPipeline
from src.sort_tracker import KalmanBox, SortTracker
from scripts.evaluate_tracking import validate_predictions


@pytest.fixture
def sequence(tmp_path):
    path = tmp_path / "MOT17" / "train" / "MOT17-01-FRCNN"
    (path / "img1").mkdir(parents=True)
    (path / "gt").mkdir()
    (path / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT17-01-FRCNN\nimDir=img1\nframeRate=25\n"
        "seqLength=16\nimWidth=64\nimHeight=64\nimExt=.jpg\n"
    )
    for frame in range(1, 17):
        cv2.imwrite(str(path / "img1" / f"{frame:06d}.jpg"), np.zeros((64, 64, 3), np.uint8))
    (path / "gt/gt.txt").write_text("".join(f"{i},1,21,21,20,20,1,1,1\n" for i in range(1, 17)))
    return path


def detection(x=20, y=20, cls=0):
    return np.array([[x, y, x + 20, y + 20, 0.9, cls]], dtype=float)


def test_dataset_order_metadata_and_no_gt_requirement(sequence):
    dataset = MOT17Dataset(sequence.parents[1])
    seq = next(iter(dataset))
    assert len(dataset) == 1 and len(seq) == 16
    assert seq.info.frame_rate == 25
    assert seq.info.width == seq.info.height == 64
    frames = list(seq)
    assert [frame.frame_id for frame in frames] == list(range(1, 17))
    assert frames[-1].timestamp_seconds == pytest.approx(15 / 25)
    assert frames[0].image.shape == (64, 64, 3)
    seq.ground_truth_path.unlink()
    assert len(list(MOT17Sequence(sequence))) == 16


def test_missing_frame_rejected(sequence):
    (sequence / "img1/000005.jpg").unlink()
    with pytest.raises(ValueError, match="missing=\\[5\\]"):
        MOT17Sequence(sequence)


def test_bad_dimensions_rejected(sequence):
    cv2.imwrite(str(sequence / "img1/000001.jpg"), np.zeros((32, 32, 3), np.uint8))
    with pytest.raises(ValueError, match="dimensions"):
        MOT17Sequence(sequence).read_frame(1)


def test_duplicate_frame_rejected(sequence):
    (sequence / "img1/1.jpg").write_bytes((sequence / "img1/000001.jpg").read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        MOT17Sequence(sequence)


def test_kalman_constant_velocity_and_coasting():
    tracker = SortTracker(max_age=10)
    for frame in range(12):
        tracks = tracker.step(detection(20 + frame * 2, 20 + frame))
        assert len(tracks) == 1 and tracks[0].id == 1
    for frame in range(12, 17):
        tracks = tracker.step(None)
        np.testing.assert_allclose(tracks[0].box, detection(20 + frame * 2, 20 + frame)[0, :4], atol=0.15)
    assert tracker.step(detection(54, 37))[0].id == 1
    assert tracker.tracks[0].filter.x.shape == (8,)


def test_kalman_ratio_velocity_and_covariance():
    kalman = KalmanBox([0, 0, 20, 20])
    for step in range(20):
        ratio = 1 + step * 0.03
        width, height = np.sqrt(400 * ratio), np.sqrt(400 / ratio)
        box = np.array([-width / 2, -height / 2, width / 2, height / 2]) + 20
        kalman.predict()
        kalman.update(box)
    assert kalman.x[7] > 0
    assert np.linalg.eigvalsh(kalman.P).min() >= -1e-8
    np.testing.assert_allclose(kalman.P, kalman.P.T, atol=1e-10)
    kalman.x[6:8] = -1e6
    assert np.isfinite(kalman.predict()).all()
    assert np.all(kalman.box[2:] > kalman.box[:2])


def test_skipped_vs_empty_and_expiry():
    tracker = SortTracker(max_age=2)
    assert tracker.step(detection())[0].id == 1
    assert tracker.step(None)[0].id == 1
    assert tracker.step(np.empty((0, 6))) == []
    assert tracker.step(None) == []
    assert tracker.step(detection())[0].id == 2


def test_hungarian_reordered_detections_and_class_gate():
    tracker = SortTracker()
    tracker.step(np.concatenate([detection(0), detection(40)]))
    tracks = tracker.step(np.concatenate([detection(39), detection(1)]))
    assert {track.id: round(track.box[0]) for track in tracks} == {1: 1, 2: 39}
    changed = tracker.step(detection(1, cls=2))
    assert len(changed) == 1 and changed[0].id == 3


@pytest.mark.parametrize("interval", [1, 2, 5, 10, 15])
def test_strict_fixed_schedule(interval):
    pipeline = SortPipeline(lambda frame: detection(), "fixed", interval)
    rows = [pipeline.step(np.zeros((64, 64, 3), np.uint8))[1] for _ in range(31)]
    assert [r["frame"] for r in rows if r["detected"]] == list(range(1, 32, interval))


def eval_python():
    path = Path(
        os.environ.get(
            "EVAL_PYTHON",
            ".venv-eval/bin/python" if Path(".venv-eval/bin/python").is_file() else sys.executable,
        )
    ).absolute()
    assert path.is_file(), "Install requirements-eval.txt in .venv-eval (see README)"
    return str(path)


def make_run(tmp_path, sequence, switched=False, empty=False):
    run = tmp_path / ("empty" if empty else "switched" if switched else "perfect")
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "frames": 16,
                "throughput_fps": 123.4,
                "metadata": {"source": str(sequence), "classes": [0]},
                "HOTA": None,
            }
        )
    )
    (run / "frames.jsonl").write_text("".join(json.dumps({"frame": i}) + "\n" for i in range(1, 17)))
    (run / "tracks.txt").write_text(
        ""
        if empty
        else "".join(f"{i},{2 if switched and i > 8 else 1},21,21,20,20,.9,-1,-1,-1\n" for i in range(1, 17))
    )
    return run


def evaluate(sequence, run, check=True):
    return subprocess.run(
        [
            eval_python(),
            "-m",
            "scripts.evaluate_tracking",
            "--sequence",
            str(sequence),
            "--run-dir",
            str(run),
        ],
        capture_output=True,
        text=True,
        check=check,
    )


@pytest.mark.parametrize("switched,empty", [(False, False), (True, False), (False, True)])
def test_actual_trackeval_metrics(sequence, tmp_path, switched, empty):
    run = make_run(tmp_path, sequence, switched, empty)
    evaluate(sequence, run)
    result = json.loads((run / "summary.json").read_text())
    assert result["throughput_fps"] == 123.4
    assert result["accuracy_status"] == "evaluated"
    assert result["identification_metrics"] == {
        "IDF1": result["IDF1"],
        "IDSW": result["IDSW"],
        "status": "evaluated_by_trackeval",
    }
    assert result["IDSW"] == int(switched)
    assert result["MOTA"] == pytest.approx(0 if empty else 93.75 if switched else 100)
    assert result["IDF1"] == pytest.approx(0 if empty else 50 if switched else 100)
    assert result["HOTA"] == pytest.approx(0 if empty else 100 / np.sqrt(2) if switched else 100)


def test_eval_refuses_partial_run_without_mutating(sequence, tmp_path):
    run = make_run(tmp_path, sequence)
    summary = run / "summary.json"
    value = json.loads(summary.read_text())
    value["frames"] = 15
    summary.write_text(json.dumps(value))
    original = summary.read_bytes()
    result = evaluate(sequence, run, check=False)
    assert result.returncode != 0 and "entire sequence" in result.stderr
    assert summary.read_bytes() == original


def test_actual_trackeval_distractor_preprocessing(sequence, tmp_path):
    run = make_run(tmp_path, sequence)
    with (sequence / "gt/gt.txt").open("a") as stream:
        stream.writelines(f"{i},2,1,1,10,10,1,8,1\n" for i in range(1, 17))
    with (run / "tracks.txt").open("a") as stream:
        stream.writelines(f"{i},2,1,1,10,10,.9,-1,-1,-1\n" for i in range(1, 17))
    evaluate(sequence, run)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["MOTA"] == pytest.approx(100)
    assert summary["HOTA"] == pytest.approx(100)


def test_bad_prediction_duplicate_id(tmp_path):
    path = tmp_path / "tracks.txt"
    path.write_text("1,1,1,1,20,20,.9,-1,-1,-1\n" * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_predictions(path, 1)


def test_full_ablation_cli(sequence, tmp_path):
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    output = np.zeros((1, 84, 1), np.float32)
    output[0, :5, 0] = [30, 30, 20, 20, 0.9]
    node = helper.make_node("Constant", [], ["output"], value=numpy_helper.from_array(output))
    graph = helper.make_graph(
        [node],
        "fixture",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 84, 1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    model_path = tmp_path / "fixture.onnx"
    onnx.save(model, model_path)
    destination = tmp_path / "sweep"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ablation_fixed_interval",
            "--sequence",
            str(sequence),
            "--model",
            str(model_path),
            "--output",
            str(destination),
            "--repeats",
            "1",
            "--warmup",
            "1",
            "--eval-python",
            eval_python(),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    curve = json.loads((destination / "tradeoff.json").read_text())
    assert [row["K"] for row in curve] == [1, 2, 5, 10, 15]
    for row in curve:
        assert row["HOTA"] == pytest.approx(100)
        assert row["FPS"] > 0
        assert row["inference_reduction_percent"] == pytest.approx(100 * (1 - np.ceil(16 / row["K"]) / 16))
    assert json.loads((destination / "manifest.json").read_text())["status"] == "complete"
