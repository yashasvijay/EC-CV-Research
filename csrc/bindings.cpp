#include "gating.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

namespace py = pybind11;
using namespace edge_gating;

namespace {

template <typename T>
void buffer_check(const py::array& array, const char* name) {
    // No array_t forcecast: wrong dtype or layout is rejected, never copied.
    if (!array.dtype().equal(py::dtype::of<T>()))
        throw py::type_error(std::string(name) + " has the wrong native-endian dtype");
    if (!(array.flags() & py::array::c_style))
        throw py::value_error(std::string(name) + " must be C-contiguous");
    if (array.size() && reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T))
        throw py::value_error(std::string(name) + " must be naturally aligned");
    // Buffers are read-only inputs: writable and read-only arrays are accepted.
    // Python argument references retain their owners through the native call.
}

std::size_t point_count(const py::array& array, const char* name) {
    buffer_check<float>(array, name);
    if (!((array.ndim() == 2 && array.shape(1) == 2) ||
          (array.ndim() == 3 && array.shape(1) == 1 && array.shape(2) == 2)))
        throw py::value_error(std::string(name) + " must have shape (N,2) or (N,1,2)");
    return static_cast<std::size_t>(array.shape(0));
}

void status_check(const py::array& array, std::size_t count, const char* name) {
    buffer_check<std::uint8_t>(array, name);
    if (!((array.ndim() == 1) || (array.ndim() == 2 && array.shape(1) == 1)) ||
        static_cast<std::size_t>(array.shape(0)) != count)
        throw py::value_error(std::string(name) + " must have shape (N,) or (N,1)");
}

std::size_t vector_count(const py::array& array, const char* name) {
    buffer_check<double>(array, name);
    if (array.ndim() != 1) throw py::value_error(std::string(name) + " must be one-dimensional");
    return static_cast<std::size_t>(array.shape(0));
}

void point_buffers(const py::array& previous, const py::array& forward, const py::array& backward,
                   const py::array& fs, const py::array& bs) {
    const auto count = point_count(previous, "previous");
    if (point_count(forward, "forward") != count || point_count(backward, "backward") != count)
        throw py::value_error("All LK point buffers must have equal counts");
    status_check(fs, count, "forward_status");
    status_check(bs, count, "backward_status");
}

template <typename Function>
auto compute(bool release_gil, Function&& function) {
    if (release_gil) {
        py::gil_scoped_release released;
        return function();
    }
    return function();
}

Mode parse_mode(const std::string& mode) {
    if (mode == "adaptive") return Mode::adaptive;
    if (mode == "baseline") return Mode::baseline;
    if (mode == "fixed") return Mode::fixed;
    throw py::value_error("mode must be adaptive, baseline, or fixed");
}

}  // namespace

PYBIND11_MODULE(edge_gating, module) {
    module.doc() = "Validated sparse-LK reductions and inference gating (float32 inputs, no implicit copies).";
    module.attr("__version__") = "0.1.0";

    py::class_<Config>(module, "Config")
        .def(py::init([](std::size_t min_points, double fb_threshold, double spread_scale,
                         std::uint64_t max_interval, double min_confidence, double max_displacement) {
            Config config{min_points, fb_threshold, spread_scale, max_interval, min_confidence, max_displacement};
            config.validate();
            return config;
        }), py::arg("min_points") = 6, py::arg("fb_threshold") = 1.5, py::arg("spread_scale") = 3.0,
            py::arg("max_interval") = 10, py::arg("min_confidence") = 0.6, py::arg("max_displacement") = 0.15)
        .def_readonly("min_points", &Config::min_points)
        .def_readonly("fb_threshold", &Config::fb_threshold)
        .def_readonly("spread_scale", &Config::spread_scale)
        .def_readonly("max_interval", &Config::max_interval)
        .def_readonly("min_confidence", &Config::min_confidence)
        .def_readonly("max_displacement", &Config::max_displacement);

    py::class_<MotionResult>(module, "MotionResult")
        .def_readonly("input_points", &MotionResult::input_points)
        .def_readonly("valid_points", &MotionResult::valid_points)
        .def_property_readonly("valid_indices", [](const MotionResult& result) {
            py::array_t<std::int64_t> output(static_cast<py::ssize_t>(result.valid_indices.size()));
            std::copy(result.valid_indices.begin(), result.valid_indices.end(), output.mutable_data());
            return output;  // owned copy; never a dangling view into a result
        })
        .def_readonly("delta", &MotionResult::delta)
        .def_readonly("box", &MotionResult::box)
        .def_readonly("survival_ratio", &MotionResult::survival_ratio)
        .def_readonly("fb_error", &MotionResult::fb_error)
        .def_readonly("spread", &MotionResult::spread)
        .def_readonly("quality", &MotionResult::quality)
        .def_readonly("confidence", &MotionResult::confidence)
        .def_readonly("displacement", &MotionResult::displacement)
        .def_property_readonly("failure", [](const MotionResult& result) { return failure_name(result.failure); });

    py::class_<Decision>(module, "Decision")
        .def_readonly("should_detect", &Decision::should_detect)
        .def_property_readonly("reason", [](const Decision& result) -> py::object {
            if (result.reason == GateReason::none) return py::none();
            return py::str(reason_name(result.reason));
        })
        .def_readonly("min_confidence", &Decision::min_confidence)
        .def_readonly("max_displacement", &Decision::max_displacement)
        .def_readonly("track_count", &Decision::track_count);

    py::class_<Engine>(module, "GatingEngine")
        .def(py::init<Config>(), py::arg("config") = Config{})
        .def_property_readonly("config", &Engine::config, py::return_value_policy::copy)
        .def("assess_motion", [](const Engine& engine, py::array previous, py::array forward,
                py::array backward, py::array fs, py::array bs, py::array box,
                double prior, int width, int height, bool release_gil) {
            point_buffers(previous, forward, backward, fs, bs);
            buffer_check<float>(box, "box");
            if (box.ndim() != 1 || box.shape(0) != 4) throw py::value_error("box must have shape (4,)");
            const auto count = static_cast<std::size_t>(previous.shape(0));
            const auto* p = static_cast<const float*>(previous.data());
            const auto* f = static_cast<const float*>(forward.data());
            const auto* b = static_cast<const float*>(backward.data());
            const auto* fs_ptr = static_cast<const std::uint8_t*>(fs.data());
            const auto* bs_ptr = static_cast<const std::uint8_t*>(bs.data());
            const auto* bb = static_cast<const float*>(box.data());
            const std::array<float, 4> box_value{bb[0], bb[1], bb[2], bb[3]};
            return compute(release_gil, [&] { return engine.assess_motion(p, f, b, fs_ptr, bs_ptr,
                                                                        count, box_value, prior, width, height); });
        }, py::arg("previous").noconvert(), py::arg("forward").noconvert(),
           py::arg("backward").noconvert(), py::arg("forward_status").noconvert(),
           py::arg("backward_status").noconvert(), py::arg("box").noconvert(),
           py::arg("previous_confidence"), py::arg("width"), py::arg("height"), py::arg("release_gil") = true)
        .def("assess_batch", [](const Engine& engine, py::array previous, py::array forward,
                py::array backward, py::array fs, py::array bs, py::array offsets,
                py::array boxes, py::array priors, int width, int height, bool release_gil) {
            point_buffers(previous, forward, backward, fs, bs);
            buffer_check<std::int64_t>(offsets, "offsets");
            buffer_check<float>(boxes, "boxes");
            if (boxes.ndim() != 2 || boxes.shape(1) != 4) throw py::value_error("boxes must have shape (M,4)");
            const auto tracks = static_cast<std::size_t>(boxes.shape(0));
            if (offsets.ndim() != 1 || static_cast<std::size_t>(offsets.shape(0)) != tracks + 1 ||
                vector_count(priors, "previous_confidences") != tracks)
                throw py::value_error("offsets must have M+1 entries; previous_confidences must have M entries");
            const auto count = static_cast<std::size_t>(previous.shape(0));
            const auto* p = static_cast<const float*>(previous.data());
            const auto* f = static_cast<const float*>(forward.data());
            const auto* b = static_cast<const float*>(backward.data());
            const auto* fs_ptr = static_cast<const std::uint8_t*>(fs.data());
            const auto* bs_ptr = static_cast<const std::uint8_t*>(bs.data());
            const auto* starts = static_cast<const std::int64_t*>(offsets.data());
            const auto* bb = static_cast<const float*>(boxes.data());
            const auto* confidence = static_cast<const double*>(priors.data());
            return compute(release_gil, [&] { return engine.assess_batch(p, f, b, fs_ptr, bs_ptr,
                                                   count, starts, bb, confidence, tracks, width, height); });
        }, py::arg("previous").noconvert(), py::arg("forward").noconvert(),
           py::arg("backward").noconvert(), py::arg("forward_status").noconvert(),
           py::arg("backward_status").noconvert(), py::arg("offsets").noconvert(),
           py::arg("boxes").noconvert(), py::arg("previous_confidences").noconvert(),
           py::arg("width"), py::arg("height"), py::arg("release_gil") = true)
        .def("decide", [](const Engine& engine, std::uint64_t frame_id, std::uint64_t last_detection,
                py::array confidences, py::array displacements, const std::string& mode, bool release_gil) {
            const auto tracks = vector_count(confidences, "confidences");
            if (vector_count(displacements, "displacements") != tracks)
                throw py::value_error("Confidence and displacement counts differ");
            const auto parsed_mode = parse_mode(mode);
            const auto* confidence = static_cast<const double*>(confidences.data());
            const auto* displacement = static_cast<const double*>(displacements.data());
            return compute(release_gil, [&] { return engine.decide(frame_id, last_detection,
                                                               confidence, displacement, tracks, parsed_mode); });
        }, py::arg("frame_id"), py::arg("last_detection"), py::arg("confidences").noconvert(),
           py::arg("displacements").noconvert(), py::arg("mode") = "adaptive", py::arg("release_gil") = true);
}
