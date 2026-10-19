"""Run paced async tracking or verify an existing export against docs/DESIGN.md."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import itertools
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import cv2
import numpy as np
import psutil

from scripts.evaluate_tracking import atomic_json
from scripts.run_async_evaluation import policy_for
from src.async_pipeline import AsyncTrackerRuntime, RuntimeConfig
from src.datasets.mot17 import MOT17Sequence
from src.detector import YOLODetector

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    """Read the versioned executable contract embedded in the specification."""
    match = re.search(
        r"<!-- SLO_CONTRACT_START -->\s*```json\s*(.*?)\s*```\s*<!-- SLO_CONTRACT_END -->",
        path.read_text(),
        re.S,
    )
    if match is None:
        raise ValueError("Missing machine-readable SLO contract")
    contract = json.loads(match.group(1))
    expected = {
        "version",
        "camera_fps",
        "throughput_min_fps",
        "p95_e2e_max_ms",
        "p99_e2e_max_ms",
        "p95_detection_age_max_ms",
        "deadline_ms",
        "deadline_miss_max_fraction",
        "minimum_frames",
        "minimum_track_frame_fraction",
        "minimum_useful_fusion_ratio",
    }
    if not isinstance(contract, dict) or set(contract) != expected:
        raise ValueError("Unexpected SLO contract fields")
    for value in contract.values():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Contract values must be finite and positive")
    if contract["version"] != 4 or type(contract["minimum_frames"]) is not int:
        raise ValueError("Unsupported contract version or frame count")
    if contract["minimum_track_frame_fraction"] > 1:
        raise ValueError("Track-frame threshold exceeds one")
    if contract["minimum_useful_fusion_ratio"] > 1:
        raise ValueError("Fusion ratio threshold exceeds one")
    if contract["deadline_miss_max_fraction"] >= 1 or contract["throughput_min_fps"] > contract["camera_fps"]:
        raise ValueError("Inconsistent SLO thresholds")
    return contract


def assess(summary: dict[str, Any], rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    """Require >=95% live-frame coverage and useful background fusion-run ratio.

    One useful run accepts at least one correction, regardless of box count.
    Bootstrap, empty results and rejected results do not enter its numerator.
    """
    checks: dict[str, Any] = {}
    try:
        n = len(rows)
        if n == 0:
            raise ValueError("Empty frame ledger")
        if any(type(r[key]) is not int or r[key] < 0 for r in rows for key in ("capture_ns", "published_ns")):
            raise ValueError("Timestamps must be nonnegative integer nanoseconds")
        frames = [r["frame"] for r in rows]
        captures = np.array([r["capture_ns"] for r in rows], dtype=np.float64)
        published = np.array([r["published_ns"] for r in rows], dtype=np.float64)
        latencies = (published - captures) / 1e6
        if not np.isfinite(latencies).all() or np.any(latencies < 0):
            raise ValueError("Invalid capture/publication timestamps")
        if any(type(f) is not int or f < 1 for f in frames):
            raise ValueError("Invalid frame identifiers")
        if np.any(np.diff(captures) <= 0) or np.any(np.diff(published) <= 0):
            raise ValueError("Nonmonotonic timestamps")
        expected = summary["source_frames"]
        if type(expected) is not int or expected < n or expected < frames[-1]:
            raise ValueError("Invalid expected source-frame count")
        missing = expected - n
        recorded = np.array([r["capture_to_output_ms"] for r in rows], dtype=float)
        config = summary["config"]
        elapsed = float(summary["wall_seconds"])
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Invalid measured duration")
        issues = []
        conditions = {
            "runtime/export incomplete": summary.get("status") == "complete",
            "truncated ledger": summary.get("ledger_complete") is True,
            "worker did not stop successfully": summary.get("runtime_state") == "stopped"
            and not summary.get("worker_error")
            and not summary.get("main_error"),
            "frame count mismatch": summary.get("frames") == n,
            "source frame ledger has gaps": frames == list(range(1, expected + 1)),
            "insufficient measured frames": n >= contract["minimum_frames"],
            "wrong source cadence": math.isclose(config["frame_rate"], contract["camera_fps"]),
            "wrong runtime deadline": math.isclose(config["deadline_ms"], contract["deadline_ms"]),
            "capture latency disagrees with timestamps": np.allclose(recorded, latencies, rtol=0, atol=1e-6),
            "capture slots are not camera-paced": np.allclose(
                np.diff(captures), np.diff(np.array(frames)) * 1e9 / contract["camera_fps"], rtol=0, atol=2
            ),
        }
        issues.extend(name for name, valid in conditions.items() if not valid)
        track_frames = sum(r.get("tracks", 0) > 0 for r in rows)
        if any(type(r.get("tracks")) is not int or r["tracks"] < 0 for r in rows):
            raise ValueError("Invalid track counts")
        if not isinstance(summary.get("worker_counts"), dict) or not isinstance(summary.get("counts"), dict):
            raise ValueError("Missing detector/fusion counters")
        completed = summary.get("worker_counts", {}).get("completed", 0)
        counters = summary["counts"]
        # Contract-v2 saved artifacts used this legacy name for the same run-level count.
        fused = counters.get("useful_fusion_runs", counters.get("fused_with_tracks", 0))
        if (
            "useful_fusion_runs" in counters
            and "fused_with_tracks" in counters
            and counters["useful_fusion_runs"] != counters["fused_with_tracks"]
        ):
            raise ValueError("Conflicting useful fusion run counts")
        useful_rows = sum(r.get("accepted_detection_update") is True for r in rows)
        if (
            type(completed) is not int
            or type(fused) is not int
            or completed < 0
            or fused < 0
            or fused > completed
            or fused != useful_rows
        ):
            raise ValueError("Missing/inconsistent useful fusion counters")
        checks["track_frame_fraction"] = {
            "status": "PASS"
            if track_frames / expected >= contract["minimum_track_frame_fraction"]
            else "FAIL",
            "measured": track_frames / expected,
            "operator": ">=",
            "threshold": contract["minimum_track_frame_fraction"],
        }
        checks["useful_fusion_ratio"] = {
            "status": "PASS"
            if completed > 0 and fused / completed >= contract["minimum_useful_fusion_ratio"]
            else "FAIL",
            "measured": fused / completed if completed else 0.0,
            "operator": ">=",
            "threshold": contract["minimum_useful_fusion_ratio"],
        }
        ages = [r.get("track_detection_age_ms") for r in rows if r.get("tracks", 0) > 0]
        valid_ages = bool(ages) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
            for v in ages
        )
        age_p95 = float(np.percentile(ages, 95)) if valid_ages else None
        checks["p95_detection_age_ms"] = {
            "status": "PASS"
            if age_p95 is not None and age_p95 < contract["p95_detection_age_max_ms"]
            else "FAIL",
            "measured": age_p95,
            "operator": "<",
            "threshold": contract["p95_detection_age_max_ms"],
            "definition": "maximum active-track anchor age plus capture-to-output latency; missing data fails",
        }
        checks["measurement_integrity"] = {
            "status": "FAIL" if issues else "PASS",
            "issues": issues,
            "frames": n,
        }
        throughput = n / max(elapsed, (published[-1] - captures[0]) / 1e9, expected / contract["camera_fps"])
        values = {
            "throughput_fps": (throughput, ">=", contract["throughput_min_fps"]),
            "p95_e2e_ms": (float(np.percentile(latencies, 95)), "<", contract["p95_e2e_max_ms"]),
            "p99_e2e_ms": (float(np.percentile(latencies, 99)), "<", contract["p99_e2e_max_ms"]),
            "deadline_miss_fraction": (
                (missing + int(np.count_nonzero(latencies >= contract["deadline_ms"]))) / expected,
                "<",
                contract["deadline_miss_max_fraction"],
            ),
        }
        for name, (measured, operator, threshold) in values.items():
            passed = measured >= threshold if operator == ">=" else measured < threshold
            checks[name] = {
                "status": "PASS" if passed else "FAIL",
                "measured": measured,
                "operator": operator,
                "threshold": threshold,
            }
        diagnostics = {
            "time_to_first_track_ms": next(
                ((r["published_ns"] - rows[0]["capture_ns"]) / 1e6 for r in rows if r.get("tracks", 0) > 0),
                None,
            ),
            "max_request_queue_depth": max(r.get("request_queue_depth", 0) for r in rows),
            "max_result_queue_depth": max(r.get("result_queue_depth", 0) for r in rows),
            "missing_frames": missing,
            "frames_with_tracks": sum(r.get("tracks", 0) > 0 for r in rows),
            "useful_fusion_runs": fused,
            "completed_background_runs": completed,
            "useful_fusion_ratio_definition": "background runs with >=1 accepted correction / completed background detector runs; excludes bootstrap",
            "first_track_frame": next((r["frame"] for r in rows if r.get("tracks", 0) > 0), None),
            "process_cpu_percent_mean": summary.get("process_cpu_percent_mean"),
            "sampled_peak_rss_mib": summary.get("sampled_peak_rss_mib"),
            "worker_counts": summary.get("worker_counts"),
            "detector_freshness_ms": summary.get("detector_freshness_ms"),
            "track_detection_age_ms": summary.get("track_detection_age_ms"),
            "correction_acceptance_rate": summary.get("correction_acceptance_rate"),
            "packet_accounting": summary.get("packet_accounting"),
            "identification_metrics": summary.get("identification_metrics"),
        }
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        checks = {
            name: {"status": "FAIL", "reason": "Invalid telemetry: " + str(exc)}
            for name in [
                "measurement_integrity",
                "throughput_fps",
                "p95_e2e_ms",
                "p99_e2e_ms",
                "p95_detection_age_ms",
                "deadline_miss_fraction",
                "track_frame_fraction",
                "useful_fusion_ratio",
            ]
        }
        diagnostics = {}
    return {
        "status": "PASS" if all(c["status"] == "PASS" for c in checks.values()) else "FAIL",
        "checks": checks,
        "diagnostics": diagnostics,
        "qualification": "Run-level timing and useful-fusion checks; accuracy and physical-camera qualification remain separate",
    }


def idle_check() -> dict[str, Any]:
    """Defer real hardware timing when sustained host CPU exceeds the idle limits."""
    time.sleep(20)
    samples = [psutil.cpu_percent(interval=2) for _ in range(6)]
    return {
        "cpu_percent_samples": samples,
        "mean_cpu_percent": sum(samples) / 6,
        "passed": sum(samples) / 6 <= 5 and max(samples) <= 10,
        "thermal_conditions": "uncontrolled",
    }


def write_report(output: Path, report: dict[str, Any]) -> None:
    atomic_json(output / "slo_report.json", report)
    lines = [
        "# Async runtime SLO verification",
        "",
        "Status: **" + report["status"] + "**",
        "",
        "| Metric | Status | Measured | Rule |",
        "|---|---|---:|---|",
    ]
    for name, check in report.get("checks", {}).items():
        lines.append(
            f"| {name} | {check['status']} | {check.get('measured', '—')} | {check.get('operator', '')} {check.get('threshold', check.get('issues', check.get('reason', '')))} |"
        )
    lines += [
        "",
        report.get("qualification", report.get("reason", "")),
        "",
        "Latency uses nominal capture-to-publication timestamps, including backlog. No accuracy, thermal, power or hard real-time guarantee is inferred.",
        "",
    ]
    (output / "slo_report.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design", type=Path, default=ROOT / "docs/DESIGN.md")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, help="Verify saved async telemetry without inference")
    p.add_argument("--sequence", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--policy", choices=["fixed_k", "kalman", "heuristic", "oracle"], default="fixed_k")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--frames", type=int)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args(argv)
    reserved = False
    try:
        contract = load_contract(args.design)
        if args.output.exists():
            raise ValueError("Output exists; use a new directory")
        if args.threads < 1 or args.warmup < 0 or (args.frames is not None and args.frames < 1):
            raise ValueError("Invalid frame/thread/warmup settings")
        if args.run_dir is None and (args.sequence is None or args.model is None):
            raise ValueError("Provide --sequence and --model, or --run-dir")
        seq = MOT17Sequence(args.sequence) if args.run_dir is None else None
        if seq is not None and not math.isclose(seq.info.frame_rate, contract["camera_fps"]):
            raise ValueError("Sequence cadence does not match the 30 FPS SLO contract")
        count = min(len(seq), args.frames or len(seq)) if seq is not None else None
        if args.frames is not None and seq is not None and args.frames > len(seq):
            raise ValueError("Requested frame count exceeds sequence")
        # Resolve and validate policy before reserving the output directory.
        policy = policy_for(args.policy, count, args.k, args.threshold) if count is not None else None
        args.output.mkdir(parents=True, exist_ok=False)
        reserved = True
        provenance = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "contract": contract,
            "design_sha256": digest(args.design),
            "design_path": str(args.design.resolve()),
            "verifier_sha256": digest(Path(__file__)),
            "mode": "saved_export" if seq is None else "paced_replay",
        }
        if seq is not None:
            if args.model is None:
                raise ValueError("Missing model")
            provenance["model_sha256"] = digest(args.model)
            preflight = idle_check()
            atomic_json(args.output / "preflight.json", preflight)
            if not preflight["passed"]:
                write_report(
                    args.output,
                    {
                        **provenance,
                        "status": "DEFERRED",
                        "reason": "Host CPU failed idle preflight; inference not started",
                    },
                )
                return 2
            cv2.setNumThreads(args.threads)
            warm_image = seq.read_frame(1).image

            def factory() -> YOLODetector:
                detector = YOLODetector(args.model, threads=args.threads)
                for _ in range(args.warmup):
                    detector(warm_image)
                return detector

            config = RuntimeConfig(
                frame_rate=contract["camera_fps"],
                deadline_ms=contract["deadline_ms"],
                telemetry_frames=max(4096, count * 20),
                offline=args.policy == "oracle",
            )
            metadata = {
                "sequence": seq.metadata,
                "classes": [0],
                "model_sha256": provenance["model_sha256"],
                "code_sha256": {
                    str(f.relative_to(ROOT)): digest(f) for f in ROOT.joinpath("src").rglob("*.py")
                },
                "packages": {
                    name: version(name)
                    for name in ["numpy", "opencv-python-headless", "onnxruntime", "psutil"]
                },
                "threads": args.threads,
                "warmup": args.warmup,
                "ingestion": "nominal 30 FPS paced replay",
            }
            runtime = AsyncTrackerRuntime(factory, policy, config, metadata=metadata)
            try:
                runtime.start(timeout_s=60)
                for _ in runtime.run(itertools.islice(seq.images(), count), paced=True):
                    pass
            finally:
                stopped = runtime.close(timeout_s=30)
                if not stopped.stopped:
                    raise RuntimeError("Worker shutdown incomplete")
                runtime.export(args.output / "runtime")
            run = args.output / "runtime"
        else:
            run = args.run_dir
        summary = json.loads((run / "summary.json").read_text())
        rows = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
        report = assess(summary, rows, contract)
        report.update(provenance)
        report["input_hashes"] = {name: digest(run / name) for name in ["summary.json", "frames.jsonl"]}
        report["runtime_directory"] = str(run.resolve())
        write_report(args.output, report)
        for name, check in report["checks"].items():
            print(
                f"{name}: {check['status']} {check.get('measured', check.get('issues', check.get('reason', '')))}"
            )
        return 0 if report["status"] == "PASS" else 1
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        if reserved and not (args.output / "slo_report.json").exists():
            write_report(
                args.output,
                {
                    "status": "ERROR",
                    "reason": str(exc),
                    "checks": {
                        name: {"status": "FAIL", "reason": "Measurement unavailable: " + str(exc)}
                        for name in [
                            "measurement_integrity",
                            "throughput_fps",
                            "p95_e2e_ms",
                            "p99_e2e_ms",
                            "p95_detection_age_ms",
                            "deadline_miss_fraction",
                            "track_frame_fraction",
                            "useful_fusion_ratio",
                        ]
                    },
                },
            )
        print("SLO verification error: " + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
