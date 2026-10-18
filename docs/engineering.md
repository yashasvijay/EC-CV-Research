# Development and reproducibility

> Current scope (2026-10-18): Current lint/test results are recorded in reports/publication-validation/validation.json. A full saved SLO run passes latency-only gates with zero track outputs; historical test counts and prior deferred-load checks remain archival. See [verified performance](verified-performance.md).

Use Python 3.13, matching the validated benchmark interpreter and CI. Create an isolated environment before installing:

```bash
python3.13 -m venv .venv-dev
source .venv-dev/bin/activate
pip install -e '.[dev,eval]'
make test
make lint
```

`pip install -e .` installs the runtime and exposes `src` and `scripts` for module execution. `make install` adds test, lint and typing tools. `make lint` runs Ruff and strict mypy across every source module. The optional native extension remains a separate CMake build; its integration tests explicitly skip if unavailable. CI runs the full Python suite, including synthetic ONNX and TrackEval checks; it does not download MOT17 or assert hardware performance.

Only the untyped third-party ONNX Runtime import is exempted from missing-library-stub reporting. Project modules remain in strict mode. Arrays use an explicit variable-dtype NumPy boundary type where OpenCV operations accept multiple dtypes. Heterogeneous JSON telemetry and the dynamically loaded native extension use explicit `Any` at serialization/extension boundaries; strict checking does not make these external values runtime-validated schemas. The LK adapter documents OpenCV's incorrect optional-output stub and preserves native calls and profiler instrumentation.

Dependency ranges in `pyproject.toml` define installation compatibility, not a frozen experimental environment. Preserve original benchmark environments; record `pip freeze`, source/model hashes, thread settings and host load for each new experiment. TrackEval brings the regular OpenCV wheel; do not install evaluation extras into the benchmark environment. GitHub Actions is configured locally and is only verified remotely after the workflow is pushed and runs.

## Policy evaluation

The default oracle is an **every-frame OraclePolicy baseline reference**. It requests every source frame through the same asynchronous queue and fusion delays. It is not a theoretical optimum, a ground-truth oracle, or a proven accuracy upper bound. No hindsight schedule is inferred.

```bash
.venv/bin/python -m scripts.ablation_policies \
  --sequences data/raw/MOT17/train/MOT17-02-FRCNN data/raw/MOT17/train/MOT17-04-FRCNN \
  --model models/yolo11n.onnx --output results/policy_ablation_roi \
  --eval-python .venv-eval/bin/python
```

The harness runs policies in seeded randomized order in fresh processes, one repetition by default. It exports full causal ledgers, deadline-valid predictions, TrackEval HOTA/MOTA/IDF1/IDSW, capture-to-output and work quantiles, queue/freshness telemetry, CPU/RAM, and model/source hashes. A single repetition does not establish repeatability. Paced replay models camera cadence without dropping input frames; backlogs remain in capture-to-output latency. It is not a live camera driver.

Plot the completed policy sweep with:

```bash
.venv-eval/bin/python -m scripts.plot_pareto \
  --policy-summary results/policy_ablation_roi/summary.json \
  --output reports/ablations/policies/pareto.png
```

This saves separate HOTA-versus-FPS and HOTA-versus-capture-p95 figures for each sequence, with nondominated points connected. The original grid plot remains historical, from the earlier synchronous tracker; it is not evidence for the optimized async runtime.

## SLO verifier

`python -m scripts.verify_slos --sequence data/raw/MOT17/train/MOT17-02-FRCNN --model models/yolo11n.onnx --policy fixed_k --k 3 --output results/slos/mot17-02-fixed3-repeat` performs idle preflight and writes a new run. `--run-dir results/slos/mot17-02-fixed3/runtime --output NEW_DIRECTORY` checks the saved export without inference. The saved run has zero tracks; its timing PASS is not a release qualification. Use a new output path for every invocation.

Publication check (2026-10-18): `make test` passed **137 tests**; `make lint` passed Ruff and strict mypy across **26 source modules**. See [validation record](../reports/publication-validation/validation.json). These software checks do not qualify tracking accuracy.
