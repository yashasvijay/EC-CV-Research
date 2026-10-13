"""Pareto logic, accuracy constraints and actual CLI/resume integration."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from scripts.grid_search import (
    DEFAULT_CONFIDENCES, DEFAULT_DISPLACEMENTS, DEFAULT_INTERVALS,
    aggregate_rows, pareto_frontier,
)


def test_pareto_dominance_and_equal_points():
    rows = [
        {"id": "a", "reduction": 30, "latency": 10, "HOTA": 80},
        {"id": "b", "reduction": 20, "latency": 12, "HOTA": 90},
        {"id": "c", "reduction": 50, "latency": 15, "HOTA": 70},
        {"id": "tie", "reduction": 30, "latency": 10, "HOTA": 80},
        {"id": "missing", "reduction": 10, "latency": None, "HOTA": None},
    ]
    objectives = {"reduction": "max", "latency": "min"}
    assert {r["id"] for r in pareto_frontier(rows, objectives)} == {"a", "c", "tie"}
    assert {r["id"] for r in pareto_frontier(rows, objectives | {"HOTA": "max"})} == {"a", "b", "c", "tie"}


def test_requested_grid_size():
    assert len(DEFAULT_INTERVALS) * len(DEFAULT_CONFIDENCES) * len(DEFAULT_DISPLACEMENTS) == 48


def test_run_level_statistics_and_accuracy_constraint():
    rows = []
    for interval, hota in [(10, 80), (20, 60)]:
        for repeat, latency in [(1, 10), (2, 12)]:
            row = dict(
                max_interval=interval, min_confidence=.5, max_displacement=.15,
                job_id=f"{interval}-{repeat}", frames=100, detector_calls=40,
                inference_reduction_percent=60, throughput_fps=80, pipeline_fps=90,
                pipeline_latency_mean_ms=latency, pipeline_latency_p50_ms=8,
                pipeline_latency_p95_ms=20, pipeline_latency_p99_ms=30,
                process_cpu_percent_mean=90, sampled_peak_rss_mib=100,
                HOTA=hota, MOTA=70, IDF1=75, IDSW=3,
            )
            rows.append(row)
    aggregate = aggregate_rows(rows, baseline_hota=82, max_hota_drop=3)
    assert aggregate[0]["pipeline_latency_mean_ms"] == 11
    assert aggregate[0]["pipeline_latency_mean_ms_sd"] == pytest.approx(np.sqrt(2))
    assert aggregate[0]["within_hota_limit"] is True
    assert aggregate[1]["within_hota_limit"] is False
    assert aggregate[0]["pareto_within_hota_limit"] is True
    assert aggregate[1]["pareto_within_hota_limit"] is False
    rows[0]["HOTA"] = None
    assert aggregate_rows(rows[:1])[0]["HOTA"] is None
    assert aggregate_rows(rows[:1])[0]["pipeline_latency_mean_ms_sd"] is None


@pytest.fixture
def fixture_source(tmp_path):
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    path = tmp_path / "MOT17-02-FRCNN"
    (path / "img1").mkdir(parents=True)
    (path / "gt").mkdir()
    (path / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT17-02-FRCNN\nimDir=img1\nframeRate=30\n"
        "seqLength=8\nimWidth=64\nimHeight=64\nimExt=.png\n"
    )
    rng = np.random.default_rng(17)
    image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    for i in range(1, 9):
        cv2.imwrite(str(path / "img1" / f"{i:06d}.png"), image)
    (path / "gt/gt.txt").write_text("".join(f"{i},1,13,13,40,40,1,1,1\n" for i in range(1, 9)))
    output = np.zeros((1, 84, 1), np.float32)
    output[0, :5, 0] = [32, 32, 40, 40, .9]
    node = helper.make_node("Constant", [], ["output"], value=numpy_helper.from_array(output))
    graph = helper.make_graph([node], "fixture", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 84, 1])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    model_path = tmp_path / "fixture.onnx"
    onnx.save(model, model_path)
    return path, model_path


def test_evaluated_grid_exports_and_resume(fixture_source, tmp_path):
    source, model = fixture_source
    evaluator = str(Path(os.environ.get("EVAL_PYTHON", ".venv-eval/bin/python")).absolute())
    destination = tmp_path / "grid"
    baseline = tmp_path / "baseline"
    subprocess.run([sys.executable, "-m", "scripts.run_phase1", "--source", str(source),
                    "--model", str(model), "--mode", "baseline", "--output", str(baseline),
                    "--warmup", "1"], check=True, capture_output=True, text=True)
    command = [sys.executable, "-m", "scripts.grid_search", "--sequence", str(source / "img1"),
               "--model", str(model), "--output", str(destination), "--max-intervals", "10",
               "--min-confidences", "0.5", "0.6", "--max-displacements", "0.15", "--warmup", "1",
               "--eval-python", evaluator, "--baseline-run", str(baseline), "--max-hota-drop", "1"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    summary = json.loads((destination / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["completed_runs"] == 2
    assert len(summary["configurations"]) == 2
    for row in summary["configurations"]:
        assert row["HOTA"] == pytest.approx(100)
        assert row["inference_reduction_percent"] == pytest.approx(87.5)
        assert row["pipeline_latency_mean_ms"] > 0
        assert row["HOTA_sd"] is None
        assert row["hota_drop_pp"] == pytest.approx(0)
        assert row["within_hota_limit"] is True
    assert summary["baseline"]["HOTA"] == pytest.approx(100)
    assert summary["pareto_frontiers"]["within_hota_limit"]
    with (destination / "summary.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 2
    assert summary["pareto_frontiers"]["quality"]
    original = {p: p.stat().st_mtime_ns for p in (destination / "runs").glob("*/summary.json")}
    subprocess.run(command + ["--resume"], check=True, capture_output=True, text=True)
    assert {p: p.stat().st_mtime_ns for p in original} == original
    incompatible = subprocess.run(command + ["--resume", "--min-confidences", "0.5", "0.7"], capture_output=True, text=True)
    assert incompatible.returncode != 0 and "differs" in incompatible.stderr


def test_interrupted_search_retries_without_overwriting(fixture_source, tmp_path):
    source, model = fixture_source
    destination = tmp_path / "grid-off"
    command = [sys.executable, "-m", "scripts.grid_search", "--sequence", str(source),
               "--model", str(model), "--output", str(destination), "--max-intervals", "10",
               "--min-confidences", "0.5", "--max-displacements", "0.15", "--warmup", "0",
               "--evaluation", "off", "--max-frames", "4"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    summary = json.loads((destination / "summary.json").read_text())
    assert summary["runs"][0]["HOTA"] is None
    assert summary["pareto_frontiers"]["quality"] == []
    assert summary["runs"][0]["frames"] == 4
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(status="interrupted", runs=[])
    manifest_path.write_text(json.dumps(manifest))
    old_run = destination / "runs/job-001-attempt-01/summary.json"
    original = old_run.read_bytes()
    subprocess.run(command + ["--resume"], check=True, capture_output=True, text=True)
    assert old_run.read_bytes() == original
    assert (destination / "runs/job-001-attempt-02/summary.json").is_file()


def test_video_input_without_ground_truth(fixture_source, tmp_path):
    source, model = fixture_source
    video = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 64))
    assert writer.isOpened()
    try:
        for _ in range(5):
            writer.write(cv2.imread(str(source / "img1/000001.png")))
    finally:
        writer.release()
    output = tmp_path / "video-grid"
    subprocess.run([sys.executable, "-m", "scripts.grid_search", "--sequence", str(video),
                    "--model", str(model), "--output", str(output), "--max-intervals", "10",
                    "--min-confidences", "0.5", "--max-displacements", "0.15", "--warmup", "0"],
                   check=True, capture_output=True, text=True)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["runs"][0]["frames"] == 5
    assert summary["config"]["evaluation"] is False
    assert summary["pareto_frontiers"]["quality"] == []
