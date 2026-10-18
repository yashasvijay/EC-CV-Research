"""Plot full-sequence policy accuracy against paced throughput and capture latency."""

import json
from pathlib import Path
import math


def plot_policies(summary_path, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads(Path(summary_path).read_text())
    if summary["status"] != "complete" or summary["completed_runs"] != summary["planned_runs"]:
        raise ValueError("Only complete evaluated sweeps may be plotted")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = summary["runs"]
    outputs = []
    colors = {"fixed_k": "#A34828", "kalman": "#D9A05B", "heuristic": "#8C402F", "oracle": "#2C2A29"}
    for sequence in sorted({r["sequence"] for r in rows}):
        selected = [r for r in rows if r["sequence"] == sequence]
        for metric, label, maximize in [
            ("throughput_fps", "Observed throughput (FPS)", True),
            ("capture_p95_ms", "Capture-to-output p95 (ms)", False),
        ]:
            for r in selected:
                if not all(math.isfinite(float(r[k])) for k in [metric, "HOTA"]):
                    raise ValueError("Nonfinite plot metric")
            frontier = [
                r
                for r in selected
                if not any(
                    (
                        o["HOTA"] >= r["HOTA"]
                        and (o[metric] >= r[metric] if maximize else o[metric] <= r[metric])
                        and (o["HOTA"] > r["HOTA"] or o[metric] != r[metric])
                    )
                    for o in selected
                )
            ]
            fig, ax = plt.subplots(figsize=(8, 5), layout="constrained", facecolor="#F9F7F5")
            ax.set_facecolor("#F9F7F5")
            for r in selected:
                name = r["policy"] + (" (every-frame reference)" if r["policy"] == "oracle" else "")
                ax.scatter(r[metric], r["HOTA"], color=colors[r["policy"]], label=name, s=65)
            frontier = sorted(frontier, key=lambda r: r[metric])
            ax.plot([r[metric] for r in frontier], [r["HOTA"] for r in frontier], color="#8C402F", alpha=0.6)
            ax.set(xlabel=label, ylabel="HOTA (%)", title=sequence + " — causal async policy comparison")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
            fig.text(
                0.5,
                0.005,
                "Single-run measurements; oracle requests every frame and has no optimality claim.",
                ha="center",
                fontsize=8,
            )
            out = root / f"{sequence}-{metric}.png"
            fig.savefig(out, dpi=240)
            plt.close(fig)
            outputs.append(str(out))
    (root / "policy_plots.json").write_text(
        json.dumps(
            {"status": "complete", "source": str(Path(summary_path).resolve()), "plots": outputs}, indent=2
        )
        + "\n"
    )
    return outputs
