"""Isolated native correctness checks. No edits to active grid-search sources."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import cv2
import pytest

from src.native_gating import load_native
from src.native_pipeline import NativeGatingPipeline
from src.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def native():
    if not list((ROOT / "csrc/build/python").glob("edge_gating*.so")):
        pytest.skip("Build the optional native extension first")
    return load_native(ROOT / "csrc/build/python")


class StaticTracker:
    def __init__(self, confidence=0.9, displacement=0.03):
        self.track = SimpleNamespace(confidence=confidence, displacement=displacement)

    def predict(self, gray):
        pass

    def update(self, gray, detections):
        pass

    def visible(self):
        return [self.track]


@pytest.mark.parametrize(
    "confidence,displacement,mode,expected",
    [
        (0.9, 0.03, "adaptive", None),
        (0.5, 0.03, "adaptive", "confidence"),
        (0.9, 0.2, "adaptive", "displacement"),
        (0.9, 0.03, "baseline", "baseline"),
        (0.9, 0.03, "fixed", None),
    ],
)
def test_optin_pipeline_reason_parity(native, confidence, displacement, mode, expected):
    def detector(frame):
        return np.empty((0, 6), np.float32)

    python = Pipeline(detector, mode=mode)
    cpp = NativeGatingPipeline(detector, mode=mode, use_native_gating=True)
    python.frame_id = cpp.frame_id = 1
    python.last_detection = cpp.last_detection = 1
    python.tracker, cpp.tracker = (
        StaticTracker(confidence, displacement),
        StaticTracker(confidence, displacement),
    )
    frame = np.zeros((64, 64, 3), np.uint8)
    left, right = python.step(frame)[1], cpp.step(frame)[1]
    for key in ("trigger", "detected", "confidence_before_detection", "displacement_fraction"):
        assert left[key] == right[key]
    assert right["trigger"] == expected
    assert right["gating_ms"] >= 0 and right["gating_backend"] == "native"


def test_python_default_delegates_without_loading_native(monkeypatch):
    import src.native_pipeline as module

    monkeypatch.setattr(module, "make_engine", lambda *a, **k: pytest.fail("Default path loaded native"))
    pipeline = NativeGatingPipeline(lambda frame: np.empty((0, 6), np.float32))
    pipeline.frame_id, pipeline.last_detection = 1, 1
    pipeline.tracker = StaticTracker()
    row = pipeline.step(np.zeros((64, 64, 3), np.uint8))[1]
    assert "gating_backend" not in row and "gating_ms" not in row


def test_prepared_pipeline_patch_in_temp_copy(native, tmp_path):
    shutil.copytree(ROOT / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(ROOT / "scripts/run_phase1.py", tmp_path / "scripts/run_phase1.py")
    patch = ROOT / "csrc/patches/native-gating-integration.patch"
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "apply", str(patch)], cwd=tmp_path, check=True, capture_output=True)
    metadata = json.loads((ROOT / "csrc/patches/native-gating-integration.json").read_text())
    for path, entry in metadata["targets"].items():
        assert hashlib.sha256((tmp_path / path).read_bytes()).hexdigest() == entry["after_sha256"]
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == entry["before_sha256"]
    spec = importlib.util.spec_from_file_location("src._patched_pipeline", tmp_path / "src/pipeline.py")
    patched = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patched)
    pipeline = patched.Pipeline(lambda frame: np.empty((0, 6), np.float32), use_native_gating=True)
    assert pipeline.step(np.zeros((64, 64, 3), np.uint8))[1]["gating_backend"] == "native"
    python = patched.Pipeline(lambda frame: np.empty((0, 6), np.float32))
    assert "gating_backend" not in python.step(np.zeros((64, 64, 3), np.uint8))[1]


def test_guard_and_benchmark_output(native, tmp_path, monkeypatch):
    from scripts import benchmark_native as module

    isolated = tmp_path / "project"
    (isolated / "results/grid").mkdir(parents=True)
    (isolated / "results/grid/manifest.json").write_text(
        json.dumps({"status": "running", "config": {"max_intervals": [10]}})
    )
    monkeypatch.setattr(module, "ROOT", isolated)
    destination = tmp_path / "benchmark"
    args = [
        "--iterations",
        "3",
        "--warmup",
        "1",
        "--output",
        str(destination),
        "--native-module-dir",
        str(ROOT / "csrc/build/python"),
    ]
    with pytest.raises(SystemExit):
        module.main(args)
    assert not destination.exists()
    module.main(args + ["--allow-concurrent"])
    summary = json.loads((destination / "summary.json").read_text())
    assert summary["status"] == "complete" and len(summary["runs"]) == 5
    assert summary["HOTA"] is None and summary["MOTA"] is None
    assert all(row["calls"] == 3 and row["latency_us"]["mean"] > 0 for row in summary["runs"])
    assert len((destination / "calls.jsonl").read_text().splitlines()) == 15


def test_isolated_native_cli_and_metadata(native, tmp_path):
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    source = tmp_path / "images"
    source.mkdir()
    image = np.random.default_rng(42).integers(0, 256, (64, 64, 3), np.uint8)
    for frame in range(1, 5):
        cv2.imwrite(str(source / f"{frame:06d}.png"), image)
    output = np.zeros((1, 84, 1), np.float32)
    output[0, :5, 0] = [32, 32, 40, 40, 0.9]
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output"], value=numpy_helper.from_array(output))],
        "fixture",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 84, 1])],
    )
    model = tmp_path / "fixture.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9), model)
    summaries = []
    for enabled in (False, True):
        destination = tmp_path / str(enabled)
        command = [
            sys.executable,
            "-m",
            "scripts.run_native_phase1",
            "--model",
            str(model),
            "--source",
            str(source),
            "--mode",
            "adaptive",
            "--output",
            str(destination),
            "--warmup",
            "0",
        ]
        if enabled:
            command.append("--use-native-gating")
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        summaries.append(json.loads((destination / "summary.json").read_text()))
    assert all(summary["status"] == "complete" and summary["frames"] == 4 for summary in summaries)
    assert summaries[0]["triggers"] == summaries[1]["triggers"]
    assert "gating_backend" not in summaries[0]["metadata"]
    assert summaries[1]["metadata"]["gating_backend"] == "native"
    assert len(summaries[1]["metadata"]["native_module_sha256"]) == 64
