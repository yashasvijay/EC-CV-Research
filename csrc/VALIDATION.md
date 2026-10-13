# Native gating validation

Date: 2026-10-12

Build: AppleClang 15.0.0, C++17 Release, macOS ARM64, Python 3.13.3,
pybind11 3.1.0, CMake 4.4.4. Dependencies were installed into
`csrc/.venv-build`; the existing pipeline environment was not modified.
The local resolved dependency list is in ignored `csrc/build/dependencies.txt`.

- CMake configuration and single-job compilation succeeded.
- `file csrc/build/python/edge_gating.cpython-313-darwin.so` reports a Mach-O
  64-bit ARM64 bundle.
- CTest: one standalone C++ correctness test passed (0.18 seconds).
- Native Python tests: 15 passed (final check: 0.32 seconds).
- The extension imported successfully in the existing `.venv` interpreter.
- Native-source attribution checks and whitespace checks passed.

Tests cover known translation and normalized motion, even-sized medians,
forward/backward consistency filtering, point-loss failure, nonfinite outputs,
box clipping, randomized NumPy formula parity, threshold equality and trigger
precedence, frame/config validation, dtype/shape/stride/alignment rejection,
read-only buffers, independent result ownership, ragged batches, empty batches,
and concurrent calls with immutable inputs. Root test discovery skips optional
binding tests when the extension is not on the Python path.

All changes for this task are under `csrc/`. Python pipeline, tracker, grid-search
source, live run records and runtime dependencies were not edited, and no grid
search process was restarted. Small local correctness builds/tests were run;
no native timing benchmark or complete video experiment was launched while the
search was active. Backend activation remains a separate integration step.

No speedup, accuracy gain, energy reduction or fully allocation-free execution
is established by these correctness checks. The module borrows contiguous input
arrays without copies; result objects and scratch storage allocate memory.
Use the included instrumented microbenchmark after concurrent jobs finish, then
perform an end-to-end native/Python pipeline comparison with TrackEval before
making a performance claim.
