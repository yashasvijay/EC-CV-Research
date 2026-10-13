#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace edge_gating {

struct Config {
    std::size_t min_points = 6;
    double fb_threshold = 1.5;
    double spread_scale = 3.0;
    std::uint64_t max_interval = 10;
    double min_confidence = 0.6;
    double max_displacement = 0.15;
    void validate() const;
};

enum class MotionFailure { none, insufficient_points, nonfinite_forward, invalid_box };
enum class GateReason { none, baseline, fixed_interval, no_tracks, max_interval,
                        invalid_motion, confidence, displacement };
enum class Mode { adaptive, baseline, fixed };

struct MotionResult {
    std::size_t input_points = 0;
    std::size_t valid_points = 0;
    std::vector<std::int64_t> valid_indices;  // relative to this track's point slice
    std::array<float, 2> delta{0.0F, 0.0F};
    std::array<float, 4> box{};               // clipped translated xyxy
    double survival_ratio = 0.0;
    std::optional<double> fb_error;
    std::optional<double> spread;
    double quality = 0.0;
    double confidence = 0.0;                 // previous_confidence * quality
    double displacement = 0.0;               // median shift / original box diagonal
    MotionFailure failure = MotionFailure::none;
};

struct Decision {
    bool should_detect = false;
    GateReason reason = GateReason::none;
    double min_confidence = 0.0;
    double max_displacement = 0.0;
    std::size_t track_count = 0;
};

// Pointer arguments are borrowed synchronously. Coordinates are interleaved
// float32 x/y; statuses are uint8, with exactly 1 meaning LK success.
// Config is copied and immutable inside Engine. No Python/OpenCV dependency,
// shared mutable workspace, thread creation, retained buffers or hidden state.
class Engine {
public:
    explicit Engine(Config config = {});
    const Config& config() const noexcept { return config_; }

    MotionResult assess_motion(
        const float* previous, const float* forward, const float* backward,
        const std::uint8_t* forward_status, const std::uint8_t* backward_status,
        std::size_t count, const std::array<float, 4>& box,
        double previous_confidence, int width, int height) const;

    // Ragged tracks share concatenated point buffers; offsets has tracks+1
    // entries, starts at 0 and ends at point_count. Scratch capacity is reused
    // within the batch. Result storage is owned, never a view of input memory.
    std::vector<MotionResult> assess_batch(
        const float* previous, const float* forward, const float* backward,
        const std::uint8_t* forward_status, const std::uint8_t* backward_status,
        std::size_t point_count, const std::int64_t* offsets,
        const float* boxes, const double* previous_confidences,
        std::size_t tracks, int width, int height) const;

    Decision decide(std::uint64_t frame_id, std::uint64_t last_detection,
                    const double* confidences, const double* displacements,
                    std::size_t tracks, Mode mode = Mode::adaptive) const;

private:
    const Config config_;
};

const char* failure_name(MotionFailure failure) noexcept;
const char* reason_name(GateReason reason) noexcept;

}  // namespace edge_gating
