import json
from pathlib import Path
import numpy as np
from scripts.sweep_detector_latency import SyntheticScene, LatencyDetector, run_case, main


def test_delay_wrapper_uses_remaining_minimum_wall_time(monkeypatch):
    from scripts import sweep_detector_latency as sweep

    times = iter([10.0, 10.005])
    sleeps = []
    monkeypatch.setattr(sweep.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(sweep.time, "sleep", sleeps.append)
    scene = SyntheticScene(128, 96, 2, 42)
    frame = next(scene.frames(1))
    result = LatencyDetector(scene, 40)(frame)
    np.testing.assert_array_equal(result, scene.boxes(0))
    assert abs(sleeps[0] - 0.035) < 1e-6


def test_short_injected_case_exports_complete_raw_artifacts(tmp_path):
    result = run_case(
        tmp_path / "case",
        40,
        frames=4,
        width=128,
        height=96,
        tracks=2,
        seed=42,
        k=2,
        threads=1,
        design=Path("docs/DESIGN.md"),
    )
    assert result["frames"] == 4
    assert result["fallback_attempts"] == 1
    assert result["slo_status"] == "FAIL"  # Insufficient frames cannot qualify as a measured SLO pass.
    raw = json.loads((tmp_path / "case/runtime/summary.json").read_text())
    assert raw["status"] == "complete"
    assert raw["metadata"]["bootstrap_injection"] is True
    assert (tmp_path / "case/slo_report.md").is_file()
    assert len((tmp_path / "case/runtime/frames.jsonl").read_text().splitlines()) == 4


def test_busy_sweep_defers_without_child_work(tmp_path, monkeypatch):
    from scripts import sweep_detector_latency as sweep

    monkeypatch.setattr(sweep, "idle_check", lambda: {"passed": False})
    monkeypatch.setattr(
        sweep.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("child started"))
    )
    output = tmp_path / "sweep"
    assert (
        main(["--output", str(output), "--frames", "4", "--width", "128", "--height", "96", "--tracks", "2"])
        == 2
    )
    assert json.loads((output / "summary.json").read_text())["status"] == "deferred"
    assert not list(output.glob("*/runtime"))


def test_complete_sweep_collects_four_fresh_process_cases(tmp_path, monkeypatch):
    from scripts import sweep_detector_latency as sweep

    monkeypatch.setattr(sweep, "idle_check", lambda: {"passed": True})
    commands = []

    def child(command, **kwargs):
        commands.append(command)
        destination = Path(command[command.index("--output") + 1])
        destination.mkdir()
        delay = float(command[command.index("--case-latency-ms") + 1])
        (destination / "case.json").write_text(
            json.dumps({"injected_detector_min_ms": delay, "slo_status": "FAIL"})
        )

    monkeypatch.setattr(sweep.subprocess, "run", child)
    output = tmp_path / "sweep"
    assert (
        main(["--output", str(output), "--frames", "4", "--width", "128", "--height", "96", "--tracks", "2"])
        == 0
    )
    result = json.loads((output / "summary.json").read_text())
    assert result["status"] == "complete" and len(result["runs"]) == 4
    assert {r["injected_detector_min_ms"] for r in result["runs"]} == {40, 80, 160, 320}
    assert len(commands) == 4 and (output / "summary.csv").is_file()
