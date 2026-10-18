# Reproduce the MOT17 experiment

> Current scope (2026-10-18): This guide reproduces the historical synchronous comparison. For the current paced async SLO command and development environment, use README and engineering.md. See [verified performance](verified-performance.md).

Run commands from the repository root. Python 3.13 was used for the recorded
experiment. Package versions and the model hash are preserved in
`docs/results-mot17-02.json`; resolved environments are under `reports/`.

## Environments

```bash
git clone https://github.com/yashasvijay/EC-CV-Research.git
cd EC-CV-Research
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest onnx
python3 -m venv .venv-eval
.venv-eval/bin/python -m pip install -r requirements-eval.txt
python -m pytest -q
```

TrackEval uses a separate environment so its OpenCV and plotting dependencies
do not alter the benchmark runtime. For tests, set `EVAL_PYTHON` if the evaluator
is installed somewhere other than `.venv-eval/bin/python`.

## Dataset

Obtain the official archive from https://motchallenge.net/data/MOT17/. Downloaded
frames, annotations, and archives are excluded from Git. Follow the dataset's
usage and citation terms; this repository does not redistribute the dataset.

```bash
mkdir -p data/raw/MOT17
curl -L --fail --retry 3 --continue-at - --output data/raw/MOT17.zip https://motchallenge.net/data/MOT17.zip
unzip data/raw/MOT17.zip 'train/MOT17-02-FRCNN/*' -d data/raw/MOT17
```

Required layout:

```text
data/raw/MOT17/train/MOT17-02-FRCNN/
├── img1/000001.jpg ... 000600.jpg
├── gt/gt.txt
└── seqinfo.ini
```

`MOT17-04-FRCNN` can be extracted with the same archive command and sequence name.
Choose one suffix per physical video to avoid triplicate image sequences.
The project uses its own detector, rather than MOT17 public detections.

## Detector export

Model weights and exports are not committed. Export YOLO11n in an isolated
runtime with raw detection output, FP32, batch one, and fixed 640x640 input:

```bash
python3 -m venv /tmp/ec-cv-export
/tmp/ec-cv-export/bin/python -m pip install ultralytics onnx
/tmp/ec-cv-export/bin/python -c 'from ultralytics import YOLO; YOLO("yolo11n.pt").export(format="onnx", imgsz=640, batch=1, dynamic=False, half=False, nms=False, simplify=False, opset=17, device="cpu")'
mkdir -p models
mv yolo11n.onnx models/yolo11n.onnx
```

The recorded model SHA-256 is in the result JSON. A different checkpoint/export
or dependency version may produce different detections and timing. Review the
model provider's license for your intended use and redistribution.

## Full-sequence runs and evaluation

Output directories must be new; previous measurements are never overwritten.
Use unique names for each repetition.

```bash
source .venv/bin/activate
SEQ=data/raw/MOT17/train/MOT17-02-FRCNN
python -m scripts.run_phase1 --model models/yolo11n.onnx --source "$SEQ/img1" --mode baseline --output results/baseline-real --threads 1
python -m scripts.run_phase1 --model models/yolo11n.onnx --source "$SEQ/img1" --mode adaptive --output results/adaptive-real --threads 1 --max-interval 10 --min-confidence 0.6 --max-displacement 0.15
python -m scripts.compare results/baseline-real results/adaptive-real
.venv-eval/bin/python -m scripts.evaluate_tracking --sequence "$SEQ" --run-dir results/baseline-real
.venv-eval/bin/python -m scripts.evaluate_tracking --sequence "$SEQ" --run-dir results/adaptive-real
```

Each run should show `status: "complete"` and `frames: 600` in `summary.json`.
MOTA/HOTA/IDF1 are null before offline evaluation. Interruptions produce partial
runs and must not be included in complete-sequence comparisons.

## Next experiments

Run at least five timing repetitions with balanced or randomized order, recording
hardware, temperature, power mode and background load. Keep all other settings
fixed. Tune gates only on development sequences and reserve held-out sequences
for the final evaluation. Accuracy metrics across a corpus require TrackEval's
sequence aggregation, not a simple average of sequence-level percentages.

```bash
python -m scripts.ablation_fixed_interval --sequence "$SEQ" --model models/yolo11n.onnx --output results/fixed-interval-week2 --repeats 5 --threads 1 --eval-python .venv-eval/bin/python
```

The fixed-K sweep holds the tracker configuration constant, runs each benchmark
in a fresh process and exports `tradeoff.csv`, `tradeoff.json` and per-run records.
Check `manifest.json` for full completion. A real dataset sweep has not yet been
claimed as completed by the initial single-sequence result.
