# Native sparse-motion gating

This module reduces the sparse point arrays returned by OpenCV's forward and
backward Lucas–Kanade calls, then makes inference scheduling decisions. It
moves point filtering, median translation, drift confidence, normalized
motion and threshold evaluation into C++. OpenCV already executes LK itself
in native code; this extension does not replace the optical-flow solver.

The native module is opt-in. The current Python pipeline and grid-search files
are unchanged, and an active experiment continues using its original backend.

## Files

```text
csrc/
├── CMakeLists.txt
├── bindings.cpp
├── include/gating.hpp
├── src/gating.cpp
├── requirements-dev.txt
├── examples/pipeline_bridge.py
├── tests/test_gating.cpp
├── tests/test_bindings.py
└── tests/benchmark_gating.py
```

`gating_core` is a standalone C++17 static library. `edge_gating` is the Python
extension. No OpenCV, Eigen or additional native library is required by the
core. The selected Python must have a development-module target compatible
with the architecture and ABI that will import the extension.

## macOS ARM64 build

Xcode Command Line Tools must be available. Use an isolated build environment
so installing CMake/pybind11 cannot alter an environment used by a live search.
Run these commands from the repository root:

```bash
.venv/bin/python -m venv csrc/.venv-build
csrc/.venv-build/bin/python -m pip install -r csrc/requirements-dev.txt
csrc/.venv-build/bin/cmake -S csrc -B csrc/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DPython_EXECUTABLE="$PWD/csrc/.venv-build/bin/python"
csrc/.venv-build/bin/cmake --build csrc/build --parallel 1
csrc/.venv-build/bin/ctest --test-dir csrc/build --output-on-failure
PYTHONPATH="$PWD/csrc/build/python" csrc/.venv-build/bin/python -m pytest -q csrc/tests/test_bindings.py
```

The output is `csrc/build/python/edge_gating.<Python ABI>.so`. Apple Silicon
builds default to ARM64 unless another architecture is explicitly supplied.
An ARM64 Python is required for an ARM64 extension; do not mix a Rosetta x86_64
Python with this build. CMake discovers pybind11 through the selected Python;
a separately installed package can be supplied with `-Dpybind11_DIR=...`.

To build only the core and C++ tests, set `-DGATING_BUILD_PYTHON=OFF`. Release
builds enable normal compiler optimization, but avoid fast-math and fused
floating-point contraction in the core so finite-value and threshold checks
remain meaningful. Build output and the isolated environment are ignored by Git.

## Single-track motion API

```python
import numpy as np
from edge_gating import Config, GatingEngine

engine = GatingEngine(Config(
    min_points=6, fb_threshold=1.5, spread_scale=3.0,
    max_interval=10, min_confidence=0.6, max_displacement=0.15,
))

result = engine.assess_motion(
    previous, forward, backward, forward_status, backward_status,
    box, previous_confidence=0.9, width=1920, height=1080,
)

retained_points = forward[result.valid_indices]
confidence = result.confidence
normalized_displacement = result.displacement
translated_box = np.asarray(result.box, dtype=np.float32)
```

`previous`, `forward` and `backward` are native-endian float32 arrays of shape
`(N,2)` or OpenCV's `(N,1,2)`. Status arrays are uint8 `(N,)` or `(N,1)`, with
exactly 1 indicating success. `box` is float32 xyxy of shape `(4,)`.
All buffers must be aligned and C-contiguous. Float64 points, boolean statuses,
strided arrays and Python lists are rejected rather than silently copied.
Read-only arrays are accepted; no input array is mutated or retained after return.
Missing OpenCV results (`None`) must be handled by the caller as tracking failure
before invoking the reduction API.

Returned fields include:

| Field | Meaning |
|---|---|
| `valid_indices`, `valid_points` | Forward/backward-consistent in-frame survivors |
| `delta` | Median x/y translation in pixels |
| `survival_ratio` | Surviving points / input points |
| `fb_error` | Median forward/backward distance in pixels, or None if insufficient points |
| `spread` | Median residual distance from median motion |
| `quality` | Survival ratio multiplied by error/coherence penalties |
| `confidence` | Previous confidence multiplied by quality |
| `displacement` | Median-motion norm / max(1, original box diagonal) |
| `box` | Translated box clipped to the frame |
| `failure` | none, insufficient_points, nonfinite_forward, or invalid_box |

The filter accepts backward error equal to `fb_threshold`, requires the minimum
point count, and uses the current tracker formula:

```text
quality = survival_ratio * exp(-median_fb_error / fb_threshold)
                         * exp(-median_spread / spread_scale)
confidence = previous_confidence * quality
```

Insufficient points and nonfinite forward outputs force confidence to zero.
A nonfinite forward point rejects the whole track, matching the existing
tracker's early failure. Individual failed-status, out-of-frame or inconsistent
backward points are filtered. Degenerate/invalid boxes also fail conservatively.
Point arithmetic/medians use float32; confidence penalties use double precision.
Parity is checked with tolerances, not claimed bit-identical across compilers.

## Scheduling API

```python
confidences = np.array([track.confidence for track in tracks], dtype=np.float64)
displacements = np.array([track.displacement for track in tracks], dtype=np.float64)
decision = engine.decide(
    frame_id, last_detection, confidences, displacements, mode="adaptive",
)
should_detect = decision.should_detect
trigger = decision.reason
```

Metric buffers are aligned, C-contiguous float64 vectors of equal length.
Frame IDs are one-based, with `last_detection=0` before the first detector call.
The engine is stateless: the caller updates `last_detection` after inference.

Adaptive precedence is **no tracks → maximum interval → invalid motion → low
confidence → displacement**. The normal valid-input order and comparisons
match `Pipeline.step`: confidence `< threshold`, displacement `> threshold`,
and elapsed interval `>= max_interval`. NaN/Inf or out-of-range track metrics
produce `invalid_motion` and force detection, an explicit defensive behavior.
`mode="baseline"` always detects; `mode="fixed"` detects on `1,1+K,...`.
A skipped frame has `decision.reason is None`.

`examples/pipeline_bridge.py` contains explicit bridge functions using the
current tracker/pipeline object fields. These functions must be called from a
future integration; nothing automatically switches the backend. In particular,
handle absent LK outputs and leave tracker association/reseeding in Python.
Enable native execution only between experiments, then verify full-sequence
latency, detector schedules, tracks and MOTA/HOTA against the Python backend.

## Batch API and ownership

`assess_batch` accepts the same point/status arrays, concatenated for M tracks,
plus int64 `offsets` of shape `(M+1,)`, float32 `boxes` of shape `(M,4)`, and
float64 `previous_confidences` of shape `(M,)`. Offsets start at 0, are monotonic,
and end at the total point count. Results are returned in track order;
`valid_indices` are relative to each track's slice.

Scratch capacity is reused within a batch, without persistent shared state.
Batching can reduce Python crossings, but constructing concatenated arrays also
costs time. Prefer existing shared/persistent buffers when available, or pass
OpenCV's direct per-track outputs to `assess_motion`; measure the whole integration.

Inputs are borrowed without an implicit dtype/layout conversion. Returned small
buffers/objects own their storage; `valid_indices` is an owned NumPy copy.
Thus this is zero-copy input access, not an entirely allocation-free call.

Array references keep input owners alive through the call. By default the GIL
is released only around pure C++ computation and reacquired before constructing
Python results. **Do not mutate, resize, or free shared underlying storage from
another thread while a native call is running.** A read-only NumPy view alone
does not make its writable base immutable. Use exclusive ownership or external
synchronization; `release_gil=False` is available for debugging or simpler
Python-level serialization. Even with the GIL held, NumPy/native writers in
other threads still require synchronization. Immutable engine configuration
and call-local scratch permit concurrent independent-buffer calls.

## Offline instrumentation

Run timing measurements after concurrent grid searches/builds finish:

```bash
PYTHONPATH="$PWD/csrc/build/python" csrc/.venv-build/bin/python \
  csrc/tests/benchmark_gating.py --points 60 --iterations 10000 \
  --output csrc/benchmark-results-single-track
```

This exports `calls.jsonl` with per-call latency, sampled CPU/RSS, and a JSON
summary with mean/p50/p95/p99 call latency, calls per second, CPU time, mean
CPU usage and sampled peak RSS. GIL-release and Python-boundary overhead are
included in call latency; resource sampling/output are excluded from it and
included in the separate instrumented throughput. Detector loading, image
flow, NN inference, association and video decoding are absent. MOTA/HOTA and
watts remain null. This microbenchmark cannot establish a pipeline speedup,
tracking accuracy, or energy savings; use the existing full pipeline logger
and TrackEval after controlled integration.

References: [pybind11 NumPy buffers](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html),
[GIL handling](https://pybind11.readthedocs.io/en/stable/advanced/misc.html),
[CMake integration](https://pybind11.readthedocs.io/en/stable/compiling.html).
