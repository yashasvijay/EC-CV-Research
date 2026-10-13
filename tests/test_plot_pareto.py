"""Plot data validation and headless rendering, using the evaluation runtime."""
import csv
import json
import os
from pathlib import Path
import struct
import subprocess

import pytest
from scripts.plot_pareto import frontier_2d, load_results


@pytest.fixture
def exports(tmp_path):
    directory = tmp_path/"grid"
    directory.mkdir()
    points = [
        {"configuration_id":"a","inference_reduction_percent":20.,"HOTA":85.},
        {"configuration_id":"b","inference_reduction_percent":40.,"HOTA":80.},
        {"configuration_id":"c","inference_reduction_percent":35.,"HOTA":75.},
        {"configuration_id":"d","inference_reduction_percent":60.,"HOTA":70.},
        {"configuration_id":"tie","inference_reduction_percent":40.,"HOTA":80.},
    ]
    with (directory/"summary.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    (directory/"summary.json").write_text(json.dumps({"status":"complete","completed_runs":5,
        "planned_runs":5,"baseline":{"HOTA":90.},"configurations":points}))
    # A 3D candidate may be dominated in the 2D projection if its latency is lower.
    (directory/"pareto_frontier.json").write_text(json.dumps({"quality":points,"efficiency":[],"within_hota_limit":[]}))
    return directory


def test_2d_frontier_baseline_and_duplicate_coordinates(exports):
    data = load_results(exports)
    assert data["line"]==[(0.,90.),(20.,85.),(40.,80.),(60.,70.)]
    assert "c" in data["quality_ids"]
    assert (35.,75.) not in data["line"]
    assert frontier_2d([{"HOTA":80.,"inference_reduction_percent":50.},
                        {"HOTA":70.,"inference_reduction_percent":50.}])==[(50.,80.)]


def test_partial_search_requires_explicit_preview(exports):
    path = exports/"summary.json"
    summary = json.loads(path.read_text())
    summary.update(status="running",planned_runs=48)
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match="not complete"): load_results(exports)
    assert load_results(exports,allow_partial=True)["complete"] is False


def test_missing_baseline_and_inconsistent_frontier_rejected(exports):
    path = exports/"summary.json"
    summary = json.loads(path.read_text())
    baseline = summary.pop("baseline")
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match="No baseline"): load_results(exports)
    summary["baseline"] = baseline
    path.write_text(json.dumps(summary))
    frontier_path = exports/"pareto_frontier.json"
    frontier = json.loads(frontier_path.read_text())
    frontier["quality"][0]["HOTA"] = 10
    frontier_path.write_text(json.dumps(frontier))
    with pytest.raises(ValueError,match="values disagree"): load_results(exports)


def test_actual_headless_png_export(exports,tmp_path):
    python = Path(os.environ.get("EVAL_PYTHON",".venv-eval/bin/python")).absolute()
    assert python.is_file(), "Install the documented evaluation environment"
    image = tmp_path/"figures/pareto_frontier.png"
    environment = os.environ | {"MPLCONFIGDIR":str(tmp_path/"mpl-cache")}
    subprocess.run([str(python),"-m","scripts.plot_pareto","--input-dir",str(exports),
                    "--output",str(image)],env=environment,check=True,capture_output=True,text=True)
    content = image.read_bytes()
    assert content[:8]==b"\x89PNG\r\n\x1a\n"
    width,height = struct.unpack(">II",content[16:24])
    assert width>=2000 and height>=1500
    provenance = json.loads(image.with_suffix(".plot.json").read_text())
    assert provenance["status"]=="complete" and provenance["baseline"]["HOTA"]==90
    assert provenance["two_dimensional_frontier"]==[[0.,90.],[20.,85.],[40.,80.],[60.,70.]]
