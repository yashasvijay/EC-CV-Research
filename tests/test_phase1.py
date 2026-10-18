import json
import time
import cv2
import numpy as np
from src.tracker import LKTracker
from src.pipeline import Pipeline
from src.benchmark import Benchmark


def scene():
    image = np.zeros((120, 160, 3), np.uint8)
    rng = np.random.default_rng(7)
    image[30:90, 40:100] = rng.integers(0, 256, (60, 60, 3), dtype=np.uint8)
    return image


def test_motion_and_identity():
    image = scene()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tracker = LKTracker()
    tracker.update(gray, np.array([[40, 30, 100, 90, 0.9, 0]], np.float32))
    moved = cv2.warpAffine(gray, np.float32([[1, 0, 3], [0, 1, 2]]), (160, 120))
    tracker.predict(moved)
    np.testing.assert_allclose(tracker.tracks[0].box, [43, 32, 103, 92], atol=0.5)
    assert tracker.tracks[0].confidence > 0.6
    tracker.update(moved, np.array([[43, 32, 103, 92, 0.9, 0]], np.float32))
    assert tracker.tracks[0].id == 1


def test_empty_texture_forces_detection():
    calls = []

    def detector(frame):
        calls.append(1)
        return np.array([[20, 20, 70, 70, 0.9, 0]], np.float32)

    pipeline = Pipeline(detector)
    for _ in range(3):
        pipeline.step(np.zeros((100, 100, 3), np.uint8))
    assert len(calls) == 3


def test_interval_and_baseline():
    def detector(frame):
        return np.array([[40, 30, 100, 90, 0.9, 0]], np.float32)

    for mode, expected in [("baseline", [1, 2, 3, 4, 5]), ("adaptive", [1, 4])]:
        pipeline = Pipeline(detector, mode, max_interval=3)
        rows = [pipeline.step(scene())[1] for _ in range(5)]
        assert [r["frame"] for r in rows if r["detected"]] == expected


def test_failed_track_is_not_exported():
    tracker = LKTracker()
    gray = cv2.cvtColor(scene(), cv2.COLOR_BGR2GRAY)
    tracker.update(gray, np.array([[40, 30, 100, 90, 0.9, 0]], np.float32))
    tracker.update(gray, np.empty((0, 6), np.float32))
    assert tracker.visible() == []


def test_benchmark(tmp_path):
    benchmark = Benchmark(tmp_path / "run", {})
    time.sleep(0.01)
    benchmark.record({"pipeline_ms": 2.0, "frame_work_ms": 3.0, "trigger": "baseline"})
    summary = benchmark.finish()
    assert summary["pipeline_fps"] == 500
    assert summary["MOTA"] is None and summary["HOTA"] is None
    assert json.loads((tmp_path / "run/frames.jsonl").read_text())["rss_mib"] > 0


def test_real_onnx_runtime_decode(tmp_path):
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    from src.detector import YOLODetector

    # Two overlapping same-class boxes; NMS must leave one.
    output = np.zeros((1, 84, 2), np.float32)
    output[0, :4, :] = np.array([[320, 321], [320, 320], [320, 320], [160, 160]])
    output[0, 4, :] = [0.9, 0.8]
    node = helper.make_node("Constant", [], ["output"], value=numpy_helper.from_array(output))
    graph = helper.make_graph(
        [node],
        "test",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 84, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    path = tmp_path / "fixture.onnx"
    onnx.save(model, path)
    detections = YOLODetector(path)(np.zeros((100, 200, 3), np.uint8))
    assert detections.shape == (1, 6)
    np.testing.assert_allclose(detections[0, :4], [50, 25, 150, 75], atol=0.01)
    # Exercise the full CLI with actual ORT execution and image decoding.
    import subprocess
    import sys

    source = tmp_path / "img1"
    source.mkdir()
    for i in range(1, 6):
        cv2.imwrite(str(source / f"{i:06d}.png"), scene())
    for mode in ("baseline", "adaptive"):
        destination = tmp_path / mode
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.run_phase1",
                "--model",
                str(path),
                "--source",
                str(source),
                "--mode",
                mode,
                "--output",
                str(destination),
                "--warmup",
                "1",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads((destination / "summary.json").read_text())
        assert report["status"] == "complete"
        assert report["frames"] == 5
        assert report["detection_fraction"] == (1.0 if mode == "baseline" else 0.2)
        lines = (destination / "tracks.txt").read_text().splitlines()
        assert len(lines) == 5 and all(line.split(",")[1] == "1" for line in lines)
