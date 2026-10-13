"""Offline synthetic native-call instrumentation. Run only after active experiments finish.

This measures the wrapper and reductions, not OpenCV LK, NN inference, or MOT
accuracy. It does not imply an end-to-end speedup. Exports per-call JSONL and a
JSON latency/FPS/CPU/RSS summary. No Python pipeline or package modifications.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter, perf_counter_ns

import numpy as np
import psutil
import edge_gating


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--points", type=int, default=60)
    parser.add_argument("--output", default="csrc/benchmark-results")
    args = parser.parse_args()
    if args.iterations < 1 or args.points < 6:
        parser.error("Use positive iterations and at least six points")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    previous = rng.uniform(10,100,(args.points,1,2)).astype(np.float32)
    forward = previous + np.array([1,2],np.float32)
    backward = previous.copy()
    status = np.ones((args.points,1),np.uint8)
    box = np.array([0,0,120,120],np.float32)
    engine = edge_gating.GatingEngine()
    arguments = (previous,forward,backward,status,status,box,.9,200,200)
    for _ in range(100): engine.assess_motion(*arguments)
    process = psutil.Process()
    cpu_start = sum(process.cpu_times()[:2])
    process.cpu_percent(None)  # prime; never report initial meaningless sample
    sample_time = start = perf_counter()
    latest_cpu = None
    latencies, memory = [], []
    with (output / "calls.jsonl").open("w") as stream:
        for index in range(args.iterations):
            before = perf_counter_ns()
            result = engine.assess_motion(*arguments)
            duration_us = (perf_counter_ns() - before) / 1000
            latencies.append(duration_us)
            now = perf_counter()
            sampled = now - sample_time >= .1
            if sampled:
                latest_cpu = process.cpu_percent(None)
                sample_time = now
            rss = process.memory_info().rss / 1024**2
            memory.append(rss)
            stream.write(json.dumps({"call":index+1,"latency_us":duration_us,
                                     "process_cpu_percent":latest_cpu,"cpu_sample_updated":sampled,
                                     "rss_mib":rss,"valid_points":result.valid_points}) + "\n")
    elapsed = perf_counter() - start
    cpu_seconds = sum(process.cpu_times()[:2]) - cpu_start
    summary = {"iterations":args.iterations,"points":args.points,
               "call_latency_us":{"mean":float(np.mean(latencies)),
                                  **{f"p{q}":float(np.percentile(latencies,q)) for q in (50,95,99)}},
               "native_calls_per_second":1e6 / float(np.mean(latencies)),
               "instrumented_calls_per_second":args.iterations / elapsed,
               "process_cpu_percent_mean":100 * cpu_seconds / elapsed,
               "process_cpu_seconds":cpu_seconds,"sampled_peak_rss_mib":max(memory),
               "MOTA":None,"HOTA":None,"power_watts":None,
               "scope":"synthetic single-track binding/reduction calls, not video FPS or tracking accuracy"}
    (output / "summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False) + "\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
