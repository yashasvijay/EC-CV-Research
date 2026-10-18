"""Streaming per-frame JSONL and aggregate JSON instrumentation."""

from __future__ import annotations
from typing import Any

import json
import platform
from time import perf_counter
from pathlib import Path
from collections import Counter
import numpy as np
import psutil


class Benchmark:
    """Export frame-level latency and process-resource measurements."""

    def __init__(self, output: str | Path, metadata: dict[str, Any]) -> None:
        """Initialize validated configuration and owned runtime state."""
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.stream = (self.output / "frames.jsonl").open("w")
        self.metadata = metadata
        self.process = psutil.Process()
        self.latencies: list[float] = []
        self.e2e_latencies: list[float] = []
        self.rss: list[float] = []
        self.triggers: Counter[str] = Counter()
        self.cpu_started = sum(self.process.cpu_times()[:2])
        self.started = self.sample_time = perf_counter()
        self.process.cpu_percent(None)  # prime, never report this initial value
        self.last_cpu: float | None = None

    def record(self, row: dict[str, Any]) -> None:
        """Append frame timing and sampled process resources to the ledger."""
        now = perf_counter()
        sampled = now - self.sample_time >= 0.1
        if sampled:
            self.last_cpu = self.process.cpu_percent(None)
            self.sample_time = now
        rss = self.process.memory_info().rss / (1024**2)
        row.update(process_cpu_percent=self.last_cpu, cpu_sample_updated=sampled, rss_mib=rss)
        self.latencies.append(row["pipeline_ms"])
        self.e2e_latencies.append(row["frame_work_ms"])
        self.rss.append(rss)
        self.triggers[row["trigger"] or "skipped"] += 1
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")

    def finish(self) -> dict[str, Any]:
        """Close the ledger and export aggregate timing and resource metrics."""
        self.stream.close()
        elapsed = perf_counter() - self.started
        cpu_seconds = sum(self.process.cpu_times()[:2]) - self.cpu_started
        count = len(self.latencies)

        def stats(values: list[float]) -> dict[str, float] | None:
            """Summarize nonempty latency samples as mean and quantiles."""
            return (
                {
                    "mean": float(np.mean(values)),
                    "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95)),
                    "p99": float(np.percentile(values, 99)),
                }
                if values
                else None
            )

        summary = {
            "status": self.metadata.get("status", "complete"),
            "metadata": self.metadata,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpus": psutil.cpu_count(),
            "frames": count,
            "wall_seconds": elapsed,
            "throughput_fps": count / elapsed,
            "pipeline_fps": 1000 * count / sum(self.latencies) if count else None,
            "pipeline_latency_ms": stats(self.latencies),
            "frame_work_latency_ms": stats(self.e2e_latencies),
            "process_cpu_seconds": cpu_seconds,
            "process_cpu_percent_mean": 100 * cpu_seconds / elapsed,
            "sampled_peak_rss_mib": max(self.rss, default=None),
            "detection_fraction": (count - self.triggers["skipped"]) / count if count else None,
            "triggers": dict(self.triggers),
            "MOTA": None,
            "HOTA": None,
            "IDF1": None,
            "IDSW": None,
            "power_watts": None,
            "accuracy_status": "not_evaluated; use MOT tracks with ground truth in TrackEval",
        }
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        return summary
