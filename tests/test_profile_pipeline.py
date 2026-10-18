import json
import pstats

import cv2
import numpy as np
import pytest

from scripts.profile_pipeline import StageTimer, instrument, main, running_searches
from src.detector import YOLODetector
from src.pipeline import Pipeline
from src import tracker as tracker_module


def test_nested_timers_reconcile_and_unwind():
    ticks = iter([0.0, 1.0, 4.0, 10.0])
    timer = StageTimer(clock=lambda: next(ticks))
    with timer.span("parent"):
        with timer.span("child"):
            pass
    assert timer.inclusive == {"parent": 10000.0, "child": 3000.0}
    assert timer.exclusive == {"parent": 7000.0, "child": 3000.0}
    timer.reset()
    timer.clock = iter([20.0, 21.0]).__next__
    with pytest.raises(ValueError), timer.span("failure"):
        raise ValueError("fixture")
    assert not timer.stack and timer.calls["failure"] == 1


@pytest.fixture
def tiny_sequence(tmp_path):
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    source = tmp_path / "MOT17-02-FRCNN"
    (source / "img1").mkdir(parents=True)
    (source / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT17-02-FRCNN\nimDir=img1\nframeRate=30\n"
        "seqLength=3\nimWidth=64\nimHeight=64\nimExt=.png\n"
    )
    image = np.random.default_rng(4).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    for i in range(1, 4):
        cv2.imwrite(str(source / "img1" / f"{i:06d}.png"), image)
    output = np.zeros((1, 84, 1), np.float32)
    output[0, :5, 0] = [32, 32, 40, 40, 0.9]
    node = helper.make_node("Constant", [], ["output"], value=numpy_helper.from_array(output))
    graph = helper.make_graph(
        [node],
        "fixture",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 84, 1])],
    )
    model = tmp_path / "fixture.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9), model)
    return source, model, image


def test_real_profile_dump_and_baseline_parity(tiny_sequence, tmp_path):
    source, model, image = tiny_sequence
    output = tmp_path / "profile"
    command = [
        "--source",
        str(source / "img1"),
        "--model",
        str(model),
        "--frames",
        "3",
        "--warmup",
        "1",
        "--output",
        str(output),
        "--search-root",
        str(tmp_path / "searches"),
    ]
    originals = (cv2.calcOpticalFlowPyrLK, cv2.goodFeaturesToTrack, tracker_module.box_iou)
    main(command)
    assert originals == (cv2.calcOpticalFlowPyrLK, cv2.goodFeaturesToTrack, tracker_module.box_iou)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["frames"] == 3 and summary["status"] == "complete"
    assert summary["HOTA"] is None and summary["detection_fraction"] == 1
    stages = summary["stage_breakdown"]
    assert sum(s["pipeline_percent"] for s in stages.values()) == pytest.approx(100)
    assert stages["yolo_inference"]["calls"] == 3
    assert stages["feature_selection"]["calls"] == 3
    assert stages["lk_flow"]["calls"] == 4
    stats = pstats.Stats(str(output / "baseline_profile.prof"))
    assert any("goodFeaturesToTrack" in key[2] for key in stats.stats)
    assert any("calcOpticalFlowPyrLK" in key[2] for key in stats.stats)
    rows = [json.loads(line) for line in (output / "frames.jsonl").read_text().splitlines()]
    for row in rows:
        assert sum(
            row["exclusive_stage_ms"][s] for s in stages if s in row["exclusive_stage_ms"]
        ) == pytest.approx(row["pipeline_ms"])
        assert row["process_cpu_percent"] is None or row["process_cpu_percent"] >= 0
        assert row["rss_mib"] > 0
    pipeline = Pipeline(YOLODetector(model), mode="baseline")
    expected = []
    for i in range(1, 4):
        tracks, _ = pipeline.step(image)
        for track in tracks:
            x1, y1, x2, y2 = track.box
            expected.append([i, track.id, x1 + 1, y1 + 1, x2 - x1, y2 - y1, track.score])
    actual = [
        [float(v) for v in line.split(",")[:7]] for line in (output / "tracks.txt").read_text().splitlines()
    ]
    np.testing.assert_allclose(actual, expected, atol=0.001)
    with pytest.raises(SystemExit):
        main(command)


def test_guard_runs_before_inference(tmp_path):
    manifest = tmp_path / "search/manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"status": "running", "config": {"max_intervals": [10]}}))
    assert running_searches(tmp_path) == [str(manifest.resolve())]
    output = tmp_path / "profile"
    with pytest.raises(SystemExit):
        main(["--search-root", str(tmp_path), "--model", "/missing/model.onnx", "--output", str(output)])
    assert not output.exists()


def test_patches_restore_on_exception(tiny_sequence):
    _, model, _ = tiny_sequence
    detector = YOLODetector(model)
    pipeline = Pipeline(detector, mode="baseline")
    original_session = detector.session
    original_feature = cv2.goodFeaturesToTrack
    with pytest.raises(RuntimeError):
        with instrument(pipeline, detector, StageTimer()):
            raise RuntimeError("fixture")
    assert detector.session is original_session and pipeline.detector is detector
    assert cv2.goodFeaturesToTrack is original_feature
