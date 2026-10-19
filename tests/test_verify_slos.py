import json
from pathlib import Path

import numpy as np
import pytest

from scripts import verify_slos as verifier


@pytest.fixture
def contract():
    return verifier.load_contract(Path("docs/DESIGN.md"))


def export_data(count=600, latency=5.0):
    rows = []
    for i in range(count):
        capture = 1_000_000_000 + int(i / 30 * 1e9)
        rows.append(
            {
                "frame": i + 1,
                "capture_ns": capture,
                "published_ns": capture + int(latency * 1e6),
                "capture_to_output_ms": latency,
                "tracks": 1,
                "track_detection_age_ms": 100.0,
                "accepted_detection_update": i < 3,
            }
        )
    summary = {
        "status": "complete",
        "ledger_complete": True,
        "runtime_state": "stopped",
        "worker_error": None,
        "main_error": None,
        "frames": count,
        "source_frames": count,
        "counts": {"useful_fusion_runs": min(3, count)},
        "worker_counts": {"completed": 30},
        "wall_seconds": count / 30,
        "config": {"frame_rate": 30.0, "deadline_ms": 33.0},
    }
    return summary, rows


def test_contract_and_successful_measurement(contract):
    summary, rows = export_data()
    result = verifier.assess(summary, rows, contract)
    assert result["status"] == "PASS"
    assert result["checks"]["throughput_fps"]["measured"] == 30
    assert result["checks"]["p99_e2e_ms"]["measured"] == 5
    assert result["diagnostics"]["frames_with_tracks"] == 600
    assert "accuracy" in result["qualification"]


def test_strict_latency_equality_fails(contract):
    summary, rows = export_data(latency=33.0)
    result = verifier.assess(summary, rows, contract)
    assert result["checks"]["p95_e2e_ms"]["status"] == "FAIL"
    assert result["checks"]["p99_e2e_ms"]["status"] == "FAIL"
    assert result["checks"]["deadline_miss_fraction"]["measured"] == 1


def test_exact_one_percent_misses_fails(contract):
    summary, rows = export_data()
    for row in rows[-6:]:
        row["published_ns"] = row["capture_ns"] + 34_000_000
        row["capture_to_output_ms"] = 34.0
    result = verifier.assess(summary, rows, contract)
    assert result["checks"]["deadline_miss_fraction"]["measured"] == 0.01
    assert result["checks"]["deadline_miss_fraction"]["status"] == "FAIL"


@pytest.mark.parametrize("change", ["truncated", "gap", "short", "clock", "config", "partial", "nan"])
def test_invalid_or_insufficient_telemetry_cannot_pass(contract, change):
    summary, rows = export_data()
    if change == "truncated":
        summary["ledger_complete"] = False
    if change == "gap":
        rows.pop(10)
        summary["frames"] -= 1
    if change == "short":
        summary, rows = export_data(50)
    if change == "clock":
        rows[2]["published_ns"] = rows[2]["capture_ns"] - 1
    if change == "config":
        summary["config"]["frame_rate"] = 60
    if change == "partial":
        summary["status"] = "partial"
    if change == "nan":
        rows[2]["capture_to_output_ms"] = float("nan")
    assert verifier.assess(summary, rows, contract)["status"] == "FAIL"


def test_slow_processing_fails_throughput(contract):
    summary, rows = export_data()
    summary["wall_seconds"] = 40.0
    assert verifier.assess(summary, rows, contract)["checks"]["throughput_fps"]["status"] == "FAIL"


def test_p99_uses_capture_latency_not_work_timer(contract):
    summary, rows = export_data(latency=50.0)
    for row in rows:
        row["frame_work_ms"] = 1.0
    assert verifier.assess(summary, rows, contract)["checks"]["p99_e2e_ms"]["measured"] == 50


def test_report_only_cli_exports_and_preserves_input(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    summary, rows = export_data()
    (run / "summary.json").write_text(json.dumps(summary))
    (run / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    before = (run / "summary.json").read_bytes()
    out = tmp_path / "report"
    assert verifier.main(["--run-dir", str(run), "--output", str(out)]) == 0
    assert json.loads((out / "slo_report.json").read_text())["status"] == "PASS"
    assert (out / "slo_report.md").is_file()
    assert (run / "summary.json").read_bytes() == before
    assert verifier.main(["--run-dir", str(run), "--output", str(out)]) == 2


def test_bad_contract_is_rejected(tmp_path):
    path = tmp_path / "design.md"
    path.write_text("no contract")
    with pytest.raises(ValueError):
        verifier.load_contract(path)


def test_real_runtime_mock_detector_path(tmp_path, monkeypatch):
    import cv2

    seq = tmp_path / "MOT17-02-FRCNN"
    (seq / "img1").mkdir(parents=True)
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT17-02-FRCNN\nimDir=img1\nframeRate=30\nseqLength=8\nimWidth=64\nimHeight=64\nimExt=.png\n"
    )
    image = np.random.default_rng(7).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    for f in range(1, 9):
        cv2.imwrite(str(seq / "img1" / f"{f:06}.png"), image)
    model = tmp_path / "mock.onnx"
    model.write_bytes(b"mock detector; no inference claim")
    monkeypatch.setattr(verifier, "idle_check", lambda: {"passed": True, "synthetic": True})
    monkeypatch.setattr(
        verifier,
        "YOLODetector",
        lambda *args, **kwargs: lambda frame: np.array([[10, 10, 50, 50, 0.9, 0]], dtype=np.float32),
    )
    output = tmp_path / "measured"
    code = verifier.main(
        ["--sequence", str(seq), "--model", str(model), "--output", str(output), "--warmup", "0"]
    )
    assert code == 1  # Eight frames cannot qualify the 600-frame contract.
    assert (output / "runtime/tracks.txt").is_file()
    report = json.loads((output / "slo_report.json").read_text())
    assert report["status"] == "FAIL"
    assert "insufficient measured frames" in report["checks"]["measurement_integrity"]["issues"]
    runtime = json.loads((output / "runtime/summary.json").read_text())
    assert runtime["status"] == "complete" and runtime["frames"] == 8


def test_busy_preflight_defers_without_inference(tmp_path, monkeypatch):
    # Covered independently from the real-runtime test: no model execution may begin.
    monkeypatch.setattr(verifier.time, "sleep", lambda _: None)
    monkeypatch.setattr(verifier.psutil, "cpu_percent", lambda interval: 20.0)
    assert verifier.idle_check()["passed"] is False


def test_empty_output_and_bootstrap_only_cannot_pass(contract):
    summary, rows = export_data()
    summary["counts"] = {"bootstrap_successes": 1, "useful_fusion_runs": 0}
    for row in rows:
        row["accepted_detection_update"] = False
    result = verifier.assess(summary, rows, contract)
    assert result["checks"]["track_frame_fraction"]["status"] == "PASS"
    assert result["checks"]["useful_fusion_ratio"]["status"] == "FAIL"
    for row in rows:
        row["tracks"] = 0
    result = verifier.assess(summary, rows, contract)
    assert result["status"] == "FAIL"
    assert result["checks"]["track_frame_fraction"]["status"] == "FAIL"


def test_fusion_counter_cannot_exceed_accepted_frame_ledger(contract):
    summary, rows = export_data()
    summary["counts"]["useful_fusion_runs"] = 4
    assert verifier.assess(summary, rows, contract)["status"] == "FAIL"


def test_invalid_fusion_ratio_contract_is_rejected(tmp_path):
    original = Path("docs/DESIGN.md").read_text()
    path = tmp_path / "design.md"
    path.write_text(
        original.replace('"minimum_useful_fusion_ratio": 0.1', '"minimum_useful_fusion_ratio": 1.1')
    )
    with pytest.raises(ValueError, match="Fusion ratio"):
        verifier.load_contract(path)


@pytest.mark.parametrize("coverage,passed", [(0.95, True), (0.9483333333333334, False), (1 / 600, False)])
def test_track_frame_fraction_requires_ninety_five_percent(contract, coverage, passed):
    summary, rows = export_data()
    populated = round(coverage * 600)
    for row in rows[populated:]:
        row["tracks"] = 0
    result = verifier.assess(summary, rows, contract)
    assert result["checks"]["track_frame_fraction"]["measured"] == pytest.approx(populated / 600)
    assert (result["checks"]["track_frame_fraction"]["status"] == "PASS") is passed


def test_legacy_run_counter_has_same_useful_fusion_semantics(contract):
    summary, rows = export_data()
    summary["counts"]["fused_with_tracks"] = summary["counts"].pop("useful_fusion_runs")
    report = verifier.assess(summary, rows, contract)
    assert report["checks"]["useful_fusion_ratio"]["measured"] == 0.1
    assert report["status"] == "PASS"


@pytest.mark.parametrize("age", [None, 300.0, float("nan")])
def test_missing_or_stale_track_freshness_fails(contract, age):
    summary, rows = export_data()
    for row in rows:
        row["track_detection_age_ms"] = age
    result = verifier.assess(summary, rows, contract)
    assert result["checks"]["p95_detection_age_ms"]["status"] == "FAIL"
