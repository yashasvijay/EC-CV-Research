"""Show run-level measurements; do not average reciprocal frame latencies."""

import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+")
a = p.parse_args()
groups = {}
for name in a.runs:
    row = json.loads((Path(name) / "summary.json").read_text())
    if row.get("status") != "complete":
        raise ValueError(f"Incomplete run: {name}")
    groups.setdefault(row["metadata"]["mode"], []).append(row)
for mode, rows in groups.items():
    print(mode, "runs=", len(rows))
    for metric in (
        "throughput_fps",
        "process_cpu_percent_mean",
        "sampled_peak_rss_mib",
        "detection_fraction",
    ):
        values = [r[metric] for r in rows]
        print(
            f"  {metric}: mean={np.mean(values):.3f}, SD={np.std(values, ddof=1) if len(values) > 1 else 0:.3f}"
        )
