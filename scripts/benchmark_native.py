"""Standalone Python/native gating microbenchmark (10,000 calls per backend).

No video inference or backend activation. Timing is refused during an active
grid search unless --allow-concurrent is explicitly supplied for diagnostics.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
from time import perf_counter, perf_counter_ns
from types import SimpleNamespace

import numpy as np
import psutil

from src.native_gating import load_native, native_decision, python_decision, python_motion

ROOT = Path(__file__).resolve().parents[1]


def active_searches(root):
    active = []
    for path in Path(root).glob("**/manifest.json"):
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if manifest.get("status") == "running" and "max_intervals" in manifest.get("config", {}):
            active.append(str(path))
    return active


def native_motion(engine, arguments, release_gil):
    result = engine.assess_motion(*arguments, release_gil=release_gil)
    # Include output conversion needed by a future tracker bridge, not just C++.
    return (
        result.confidence,
        result.displacement,
        np.asarray(result.delta, dtype=np.float32),
        np.asarray(result.box, dtype=np.float32),
        result.valid_indices,
        result.fb_error,
    )


def build_cases(engine, track_count, point_count, release_gil, seed):
    rng = np.random.default_rng(seed)
    decision_cases = []
    for index in range(5):
        tracks = [
            SimpleNamespace(confidence=float(rng.uniform(0.7, 0.99)), displacement=float(rng.uniform(0, 0.1)))
            for _ in range(track_count)
        ]
        frame, last = 2, 1
        if index == 1:
            tracks[0].confidence = 0.5
        if index == 2:
            tracks[0].displacement = 0.2
        if index == 3:
            frame, last = 11, 1
        if index == 4:
            tracks, frame, last = [], 1, 0
        confidence = np.array([t.confidence for t in tracks], np.float64)
        displacement = np.array([t.displacement for t in tracks], np.float64)
        decision_cases.append((tracks, frame, last, confidence, displacement))

    def python_schedule(case):
        tracks, frame, last, _, _ = case
        return python_decision(tracks, frame, last)

    def cpp_schedule(case):
        tracks, frame, last, _, _ = case
        return native_decision(engine, tracks, frame, last, release_gil=release_gil)

    def cpp_prepacked(case):
        _, frame, last, confidence, displacement = case
        result = engine.decide(frame, last, confidence, displacement, release_gil=release_gil)
        return result.reason, result.min_confidence, result.max_displacement

    for case in decision_cases:
        assert python_schedule(case) == cpp_schedule(case) == cpp_prepacked(case)
    motion_cases = []
    for index in range(3):
        previous = rng.uniform(10, 100, (point_count, 1, 2)).astype(np.float32)
        forward = (
            previous + rng.normal(0, 0.2, previous.shape).astype(np.float32) + np.array([1, 2], np.float32)
        )
        backward = previous + rng.normal(0, 0.15, previous.shape).astype(np.float32)
        fs, bs = np.ones((point_count, 1), np.uint8), np.ones((point_count, 1), np.uint8)
        if index == 1:
            fs[: max(1, point_count // 10)] = 0
        if index == 2:
            backward[: max(1, point_count // 10), 0, 0] += 4
        box = np.array([0, 0, 120, 120], np.float32)
        motion_cases.append((previous, forward, backward, fs, bs, box, 0.9, 200, 200))

    def python_check(case):
        return python_motion(*case)

    def cpp_check(case):
        return native_motion(engine, case, release_gil)

    for case in motion_cases:
        left, right = python_check(case), cpp_check(case)
        for index in (0, 1, 2, 3):
            np.testing.assert_allclose(left[index], right[index], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(left[4], right[4])
        if left[5] is None:
            assert right[5] is None
        else:
            np.testing.assert_allclose(left[5], right[5], rtol=1e-6, atol=1e-6)
    return {
        "decisions": (
            decision_cases,
            {
                "python": python_schedule,
                "native_with_packing": cpp_schedule,
                "native_prepacked_diagnostic": cpp_prepacked,
            },
        ),
        "motion": (motion_cases, {"python": python_check, "native": cpp_check}),
    }


def measure(function, cases, iterations, stream, suite, backend, repeat):
    process = psutil.Process()
    process.cpu_percent(None)
    cpu_started = sum(process.cpu_times()[:2])
    start = sample_time = perf_counter()
    latest_cpu = None
    latencies, rss_values = [], []
    for index in range(iterations):
        case = cases[index % len(cases)]
        before = perf_counter_ns()
        result = function(case)
        latency_us = (perf_counter_ns() - before) / 1000
        latencies.append(latency_us)
        now = perf_counter()
        refreshed = now - sample_time >= 0.1
        if refreshed:
            latest_cpu = process.cpu_percent(None)
            sample_time = now
        rss = process.memory_info().rss / 1024**2
        rss_values.append(rss)
        stream.write(
            json.dumps(
                {
                    "suite": suite,
                    "backend": backend,
                    "repeat": repeat,
                    "call": index + 1,
                    "latency_us": latency_us,
                    "process_cpu_percent": latest_cpu,
                    "cpu_sample_updated": refreshed,
                    "rss_mib": rss,
                }
            )
            + "\n"
        )
        del result  # destruction outside timing, consistently for both backends
    wall = perf_counter() - start
    cpu = sum(process.cpu_times()[:2]) - cpu_started
    mean = float(np.mean(latencies))
    return {
        "suite": suite,
        "backend": backend,
        "repeat": repeat,
        "calls": iterations,
        "latency_us": {"mean": mean, **{f"p{q}": float(np.percentile(latencies, q)) for q in (50, 95, 99)}},
        "calls_per_second": 1e6 / mean,
        "instrumented_calls_per_second": iterations / wall,
        "process_cpu_seconds": cpu,
        "process_cpu_percent_mean": 100 * cpu / wall,
        "sampled_peak_rss_mib": max(rss_values),
        "wall_seconds": wall,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--tracks", type=int, default=20)
    parser.add_argument("--points", type=int, default=60)
    parser.add_argument("--scope", choices=("all", "decisions", "motion"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hold-gil", action="store_true")
    parser.add_argument("--native-module-dir", default=str(ROOT / "csrc/build/python"))
    parser.add_argument("--output", default="results/benchmark_native")
    parser.add_argument(
        "--allow-concurrent", action="store_true", help="diagnostic timing only; allow an active grid search"
    )
    args = parser.parse_args(argv)
    if min(args.iterations, args.repeats, args.tracks) < 1 or args.warmup < 0 or args.points < 6:
        parser.error(
            "Positive iterations/repeats/tracks, nonnegative warmup and at least six points required"
        )
    active = active_searches(ROOT / "results")
    if active and not args.allow_concurrent:
        parser.error(
            "An active grid search is recorded. Run this benchmark after it finishes; "
            "--allow-concurrent is only for diagnostic checks."
        )
    native = load_native(args.native_module_dir)
    engine = native.GatingEngine()
    suites = build_cases(engine, args.tracks, args.points, not args.hold_gil, args.seed)
    if args.scope != "all":
        suites = {args.scope: suites[args.scope]}
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "status": "running",
        "config": vars(args),
        "active_grid_searches": active,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "psutil")},
        "native_module_version": native.__version__,
        "native_module_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [],
        "MOTA": None,
        "HOTA": None,
        "power_watts": None,
        "scope": "Synthetic scheduling and LK reductions; excludes optical-flow solver and neural inference.",
    }
    rng = random.Random(args.seed)
    try:
        with (output / "calls.jsonl").open("w") as stream:
            for repeat in range(1, args.repeats + 1):
                order = [(suite, backend) for suite, (_, backends) in suites.items() for backend in backends]
                rng.shuffle(order)
                for suite, backend in order:
                    cases, backends = suites[suite]
                    function = backends[backend]
                    for index in range(args.warmup):
                        function(cases[index % len(cases)])
                    print(f"{suite}/{backend}: {args.iterations} calls, repeat {repeat}", flush=True)
                    summary["runs"].append(
                        measure(function, cases, args.iterations, stream, suite, backend, repeat)
                    )
        aggregates = []
        for suite, (_, backends) in suites.items():
            for backend in backends:
                runs = [r for r in summary["runs"] if r["suite"] == suite and r["backend"] == backend]
                means = [r["latency_us"]["mean"] for r in runs]
                aggregates.append(
                    {
                        "suite": suite,
                        "backend": backend,
                        "calls": sum(r["calls"] for r in runs),
                        "latency_us": {
                            key: float(np.mean([r["latency_us"][key] for r in runs]))
                            for key in ("mean", "p50", "p95", "p99")
                        },
                        "mean_latency_sd_us": float(np.std(means, ddof=1)) if len(means) > 1 else None,
                    }
                )
        summary["aggregates"] = aggregates
        summary["speedup_python_over_native"] = {}
        for suite in suites:
            measurements = {r["backend"]: r["latency_us"]["mean"] for r in aggregates if r["suite"] == suite}
            comparison = "native_with_packing" if suite == "decisions" else "native"
            summary["speedup_python_over_native"][suite] = measurements["python"] / measurements[comparison]
        summary["status"] = "complete"
    except BaseException as exc:
        summary.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc)
        )
        raise
    finally:
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"latencies": summary["aggregates"], "speedup": summary["speedup_python_over_native"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
