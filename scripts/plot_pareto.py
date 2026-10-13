"""Plot evaluated HOTA versus inference reduction after a grid search completes."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile


def measured(value, label):
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Missing or non-numeric {label}; run TrackEval before plotting") from exc
    if not math.isfinite(result) or not 0 <= result <= 100:
        raise ValueError(f"{label} must be a finite percentage in [0,100]")
    return result


def frontier_2d(points):
    """Maximize both coordinates; collapse tied coordinates only for the line."""
    coordinates = {(point["inference_reduction_percent"], point["HOTA"]) for point in points}
    return sorted((x,y) for x,y in coordinates if not any(
        other_x >= x and other_y >= y and (other_x > x or other_y > y)
        for other_x,other_y in coordinates
    ))


def load_results(input_dir, baseline_summary=None, allow_partial=False):
    root = Path(input_dir)
    paths = {"csv":root/"summary.csv", "frontier":root/"pareto_frontier.json",
             "summary":root/"summary.json"}
    contents = {name:path.read_bytes() for name,path in paths.items()}
    summary = json.loads(contents["summary"])
    completed, planned = summary.get("completed_runs"), summary.get("planned_runs")
    complete = (summary.get("status") == "complete" and isinstance(planned,int) and planned > 0
                and completed == planned)
    if not complete and not allow_partial:
        raise ValueError(f"Search is not complete ({completed}/{planned}, status={summary.get('status')}). "
                         "Wait for completion or explicitly use --allow-partial for a labeled preview.")
    reader = csv.DictReader(io.StringIO(contents["csv"].decode("utf-8-sig")))
    required = {"configuration_id","inference_reduction_percent","HOTA"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"summary.csv requires columns: {sorted(required)}")
    points, seen = [], set()
    for row in reader:
        identity = row["configuration_id"]
        if not identity or identity in seen:
            raise ValueError("CSV configuration IDs must be nonempty and unique")
        seen.add(identity)
        points.append({"configuration_id":identity,
                       "inference_reduction_percent":measured(row["inference_reduction_percent"],"inference reduction"),
                       "HOTA":measured(row["HOTA"],"HOTA")})
    if not points:
        raise ValueError("No evaluated configurations to plot")
    exported = json.loads(contents["frontier"])
    if not isinstance(exported,dict) or not isinstance(exported.get("quality"),list):
        raise ValueError("pareto_frontier.json requires the quality frontier list")
    indexed = {point["configuration_id"]:point for point in points}
    recorded = {row["configuration_id"]:row for row in summary.get("configurations",[])}
    if set(recorded) != set(indexed):
        raise ValueError("CSV and summary.json configurations disagree; retry after exports settle")
    for identity,point in indexed.items():
        for metric in ("HOTA","inference_reduction_percent"):
            if not math.isclose(point[metric],measured(recorded[identity].get(metric),metric),rel_tol=1e-9,abs_tol=1e-9):
                raise ValueError("CSV and JSON values disagree; retry after exports settle")
    quality_ids = set()
    for row in exported["quality"]:
        identity = row.get("configuration_id")
        if identity not in indexed:
            raise ValueError("Exported frontier references a missing CSV configuration")
        for metric in ("HOTA","inference_reduction_percent"):
            if not math.isclose(indexed[identity][metric],measured(row.get(metric),metric),rel_tol=1e-9,abs_tol=1e-9):
                raise ValueError("CSV and frontier values disagree; retry after exports settle")
        quality_ids.add(identity)
    if not quality_ids:
        raise ValueError("No evaluated quality-frontier points in pareto_frontier.json")
    baseline = summary.get("baseline")
    if baseline_summary is not None:
        path = Path(baseline_summary)
        if path.is_dir(): path = path/"summary.json"
        paths["baseline"] = path
        contents["baseline"] = path.read_bytes()
        baseline = json.loads(contents["baseline"])
        if baseline.get("status") != "complete" or baseline.get("metadata",{}).get("mode") != "baseline":
            raise ValueError("Baseline override must be a complete full-inference benchmark")
        config = summary.get("config",{})
        metadata = baseline.get("metadata",{})
        source = Path(config.get("source",""))
        expected_sequence = source.parent if source.name == "img1" else source
        baseline_sequence = metadata.get("sequence") or {}
        if (baseline_sequence.get("name") != expected_sequence.name
                or metadata.get("model_sha256") != config.get("model_sha256")
                or baseline.get("frames") != summary["configurations"][0].get("frames")
                or baseline.get("detection_fraction") != 1.0):
            raise ValueError("Baseline override must match the search sequence, model and frame count")
    if not isinstance(baseline,dict):
        raise ValueError("No baseline in summary.json; provide --baseline-summary with an evaluated baseline")
    baseline_point = {"configuration_id":"baseline", "inference_reduction_percent":0.,
                      "HOTA":measured(baseline.get("HOTA"),"baseline HOTA")}
    return {"points":points,"quality_ids":quality_ids,"baseline":baseline_point,
            "line":frontier_2d(points+[baseline_point]),"complete":complete,
            "completed_runs":completed,"planned_runs":planned,
            "input_hashes":{name:{"path":str(paths[name].resolve()),
                                  "sha256":hashlib.sha256(data).hexdigest()} for name,data in contents.items()}}


def create_plot(data, output, dpi=300, title=None):
    # Use the installed evaluation/plotting runtime; no package changes to the
    # live benchmark environment. Agg supports headless execution/export.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    if output.suffix.lower() != ".png":
        raise ValueError("Output must be a PNG path")
    output.parent.mkdir(parents=True,exist_ok=True)
    points,baseline = data["points"],data["baseline"]
    candidates = [point for point in points if point["configuration_id"] in data["quality_ids"]]
    with plt.rc_context({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.titleweight":"bold","font.family":"DejaVu Sans"}):
        figure,axis = plt.subplots(figsize=(8,5.5),layout="constrained")
        axis.scatter([p["inference_reduction_percent"] for p in points],[p["HOTA"] for p in points],
                     s=42,color="#526D82",alpha=.55,edgecolors="white",linewidths=.6,
                     label="Evaluated configurations",zorder=2)
        axis.scatter([p["inference_reduction_percent"] for p in candidates],[p["HOTA"] for p in candidates],
                     s=90,facecolors="none",edgecolors="#007F73",linewidths=1.6,
                     label="Exported 3-objective Pareto candidates",zorder=3)
        line = data["line"]
        axis.plot([p[0] for p in line],[p[1] for p in line],color="#C24A27",linewidth=2,
                  marker="o",markersize=5,label="2D HOTA–reduction frontier",zorder=4)
        axis.scatter([0],[baseline["HOTA"]],marker="*",s=230,color="#202A44",edgecolors="white",
                     linewidths=.8,label=f"Baseline: HOTA {baseline['HOTA']:.2f}",zorder=5)
        axis.annotate("Full inference",(0,baseline["HOTA"]),xytext=(12,10),textcoords="offset points",
                      fontsize=9,color="#202A44")
        axis.set_xlabel("Inference reduction (%)")
        axis.set_ylabel("HOTA (%)")
        heading = title or "Tracking accuracy vs. inference reduction"
        if not data["complete"]:
            heading += f"\nPARTIAL PREVIEW — {data['completed_runs']}/{data['planned_runs']} runs"
        axis.set_title(heading,pad=14)
        axis.set_xlim(-3,min(103,max(5,max(p["inference_reduction_percent"] for p in points)+5)))
        hota_values = [p["HOTA"] for p in points]+[baseline["HOTA"]]
        padding = max(1.5,(max(hota_values)-min(hota_values))*.15)
        axis.set_ylim(max(0,min(hota_values)-padding),min(100,max(hota_values)+padding))
        axis.grid(alpha=.2,linewidth=.6)
        axis.set_axisbelow(True)
        axis.legend(loc="best",fontsize=8.5,framealpha=.95)
        figure.get_layout_engine().set(rect=(0,.06,1,.94))
        figure.text(.5,.025,"Line connects measured non-dominated points; intermediate settings were not evaluated.",
                    ha="center",fontsize=8,color="#59636E")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent,suffix=".png",delete=False) as stream:
                temporary = Path(stream.name)
            figure.savefig(temporary,dpi=dpi,format="png",facecolor="white")
            temporary.replace(output)
        finally:
            plt.close(figure)
            if temporary: temporary.unlink(missing_ok=True)
    provenance = {"timestamp_utc":datetime.now(timezone.utc).isoformat(),
                  "status":"complete" if data["complete"] else "partial_preview",
                  "completed_runs":data["completed_runs"],"planned_runs":data["planned_runs"],
                  "input_hashes":data["input_hashes"],"baseline":baseline,
                  "plotted_configurations":len(points),"two_dimensional_frontier":data["line"],
                  "objectives":{"HOTA":"maximize","inference_reduction_percent":"maximize"},
                  "exported_frontier":"quality (HOTA, reduction and latency)","dpi":dpi}
    output.with_suffix(".plot.json").write_text(json.dumps(provenance,indent=2,allow_nan=False)+"\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",default="results/grid_search")
    parser.add_argument("--output",default="reports/ablations/pareto_frontier.png")
    parser.add_argument("--baseline-summary",help="evaluated baseline summary.json or run directory, if absent from search")
    parser.add_argument("--allow-partial",action="store_true",help="explicitly render a labeled incomplete preview")
    parser.add_argument("--dpi",type=int,default=300)
    parser.add_argument("--title")
    args = parser.parse_args(argv)
    if args.dpi < 72: parser.error("Use DPI >= 72")
    try:
        data = load_results(args.input_dir,args.baseline_summary,args.allow_partial)
        output = create_plot(data,args.output,args.dpi,args.title)
    except (OSError,ValueError,ImportError) as exc:
        parser.error(str(exc))
    print(f"Saved {output} ({len(data['points'])} evaluated configurations)")


if __name__ == "__main__":
    main()
