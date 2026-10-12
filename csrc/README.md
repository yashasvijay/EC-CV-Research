# Phase 3 extension boundary
Reserve `include/`, `src/`, `bindings.cpp`, and `CMakeLists.txt` for a pybind11 extension.
No native implementation is enabled yet. Profile before moving LK/association code.
The future API must validate dtype, shape, strides, writability and array lifetime;
release the GIL only while buffers remain owned. Test numerical parity with Python.
