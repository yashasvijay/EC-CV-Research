#include "gating.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace edge_gating {
namespace {

struct Workspace {
    std::vector<float> shifts_x, shifts_y, errors, distances;
    void reset(std::size_t count) {
        shifts_x.clear(); shifts_y.clear(); errors.clear(); distances.clear();
        shifts_x.reserve(count); shifts_y.reserve(count);
        errors.reserve(count); distances.reserve(count);
    }
};

// Float32 reduction mirrors NumPy median on the tracker's float32 LK outputs.
// nth_element avoids sorting every point; even-sized medians average the two
// middle values. The workspace may be permuted because it is caller-local.
float median(std::vector<float>& values) {
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    const float upper = *middle;
    return values.size() % 2 ? upper :
        (*std::max_element(values.begin(), middle) + upper) * 0.5F;
}

float norm(float x, float y) { return std::sqrt(x * x + y * y); }

bool valid_box(const std::array<float, 4>& box) {
    return std::all_of(box.begin(), box.end(), [](float value) { return std::isfinite(value); }) &&
           box[2] > box[0] && box[3] > box[1];
}

void validate_frame(int width, int height) {
    if (width <= 0 || height <= 0) throw std::invalid_argument("Frame dimensions must be positive");
}

void validate_prior(double confidence) {
    if (!std::isfinite(confidence) || confidence < 0.0 || confidence > 1.0)
        throw std::invalid_argument("Previous confidence must be finite in [0,1]");
}

MotionResult assess(const Config& config, const float* previous, const float* forward,
                    const float* backward, const std::uint8_t* fs, const std::uint8_t* bs,
                    std::size_t count, const std::array<float, 4>& box,
                    double prior, int width, int height, Workspace& scratch) {
    validate_prior(prior);
    MotionResult result;
    result.input_points = count;
    result.box = box;
    if (!valid_box(box)) { result.failure = MotionFailure::invalid_box; return result; }
    if (count < config.min_points) {
        result.failure = MotionFailure::insufficient_points;
        return result;
    }
    // Current LKTracker fails the whole track if any forward point is nonfinite.
    for (std::size_t i = 0; i < count * 2; ++i) {
        if (!std::isfinite(forward[i])) {
            result.failure = MotionFailure::nonfinite_forward;
            return result;
        }
    }
    scratch.reset(count);
    result.valid_indices.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const float x = forward[2 * i], y = forward[2 * i + 1];
        if (fs[i] != 1 || bs[i] != 1 || x < 0.0F || x >= width || y < 0.0F || y >= height)
            continue;
        const float error = norm(backward[2 * i] - previous[2 * i],
                                 backward[2 * i + 1] - previous[2 * i + 1]);
        const float dx = x - previous[2 * i], dy = y - previous[2 * i + 1];
        if (!std::isfinite(error) || error > config.fb_threshold ||
            !std::isfinite(dx) || !std::isfinite(dy)) continue;
        result.valid_indices.push_back(static_cast<std::int64_t>(i));
        scratch.shifts_x.push_back(dx); scratch.shifts_y.push_back(dy);
        scratch.errors.push_back(error);
    }
    result.valid_points = result.valid_indices.size();
    result.survival_ratio = static_cast<double>(result.valid_points) / static_cast<double>(count);
    if (result.valid_points < config.min_points) {
        result.failure = MotionFailure::insufficient_points;
        return result;
    }
    result.delta = {median(scratch.shifts_x), median(scratch.shifts_y)};
    for (const auto index : result.valid_indices) {
        const auto i = static_cast<std::size_t>(index);
        scratch.distances.push_back(norm(forward[2 * i] - previous[2 * i] - result.delta[0],
                                         forward[2 * i + 1] - previous[2 * i + 1] - result.delta[1]));
    }
    result.fb_error = median(scratch.errors);
    result.spread = median(scratch.distances);
    result.quality = result.survival_ratio * std::exp(-*result.fb_error / config.fb_threshold) *
                     std::exp(-*result.spread / config.spread_scale);
    result.confidence = prior * result.quality;
    const double diagonal = std::max(1.0, static_cast<double>(norm(box[2] - box[0], box[3] - box[1])));
    result.displacement = static_cast<double>(norm(result.delta[0], result.delta[1])) / diagonal;
    for (std::size_t i = 0; i < 4; ++i) {
        const float limit = static_cast<float>(i % 2 ? height : width);
        result.box[i] = std::clamp(box[i] + result.delta[i % 2], 0.0F, limit);
    }
    if (!valid_box(result.box)) {
        result.confidence = 0.0;
        result.failure = MotionFailure::invalid_box;
    }
    return result;
}

void check_point_count(std::size_t count) {
    if (count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) / 2)
        throw std::invalid_argument("Point count is too large");
}

}  // namespace

void Config::validate() const {
    if (min_points == 0 || max_interval == 0 || !std::isfinite(fb_threshold) || fb_threshold <= 0.0 ||
        !std::isfinite(spread_scale) || spread_scale <= 0.0 ||
        !std::isfinite(min_confidence) || min_confidence < 0.0 || min_confidence > 1.0 ||
        !std::isfinite(max_displacement) || max_displacement <= 0.0)
        throw std::invalid_argument("Invalid gating configuration");
}

Engine::Engine(Config config) : config_(std::move(config)) { config_.validate(); }

MotionResult Engine::assess_motion(const float* previous, const float* forward, const float* backward,
    const std::uint8_t* fs, const std::uint8_t* bs, std::size_t count,
    const std::array<float, 4>& box, double prior, int width, int height) const {
    validate_frame(width, height); check_point_count(count);
    if (count && (!previous || !forward || !backward || !fs || !bs))
        throw std::invalid_argument("Nonempty point buffers cannot be null");
    Workspace scratch;
    return assess(config_, previous, forward, backward, fs, bs, count, box, prior, width, height, scratch);
}

std::vector<MotionResult> Engine::assess_batch(const float* previous, const float* forward,
    const float* backward, const std::uint8_t* fs, const std::uint8_t* bs,
    std::size_t point_count, const std::int64_t* offsets, const float* boxes,
    const double* priors, std::size_t tracks, int width, int height) const {
    validate_frame(width, height); check_point_count(point_count);
    if (!offsets || offsets[0] != 0 || offsets[tracks] != static_cast<std::int64_t>(point_count))
        throw std::invalid_argument("Offsets must start at 0 and end at point_count");
    if ((point_count && (!previous || !forward || !backward || !fs || !bs)) ||
        (tracks && (!boxes || !priors))) throw std::invalid_argument("Missing batch buffers");
    for (std::size_t i = 0; i < tracks; ++i) {
        if (offsets[i] < 0 || offsets[i] > offsets[i + 1] ||
            offsets[i + 1] > static_cast<std::int64_t>(point_count))
            throw std::invalid_argument("Offsets must be monotonic and in bounds");
    }
    Workspace scratch;
    std::vector<MotionResult> results;
    results.reserve(tracks);
    for (std::size_t i = 0; i < tracks; ++i) {
        const auto begin = static_cast<std::size_t>(offsets[i]);
        const auto count = static_cast<std::size_t>(offsets[i + 1] - offsets[i]);
        const std::array<float, 4> box{boxes[4*i], boxes[4*i+1], boxes[4*i+2], boxes[4*i+3]};
        // Avoid pointer arithmetic on nullptr for an entirely empty batch.
        results.push_back(assess(config_, point_count ? previous + 2*begin : nullptr,
            point_count ? forward + 2*begin : nullptr, point_count ? backward + 2*begin : nullptr,
            point_count ? fs + begin : nullptr, point_count ? bs + begin : nullptr,
            count, box, priors[i], width, height, scratch));
    }
    return results;
}

Decision Engine::decide(std::uint64_t frame_id, std::uint64_t last_detection,
    const double* confidences, const double* displacements, std::size_t tracks, Mode mode) const {
    if (frame_id == 0 || last_detection > frame_id)
        throw std::invalid_argument("Use one-based frame IDs and last_detection <= frame_id");
    if (tracks && (!confidences || !displacements)) throw std::invalid_argument("Missing track metrics");
    Decision result;
    result.track_count = tracks;
    result.min_confidence = tracks ? 1.0 : 0.0;
    bool invalid = false;
    for (std::size_t i = 0; i < tracks; ++i) {
        invalid = invalid || !std::isfinite(confidences[i]) || confidences[i] < 0.0 || confidences[i] > 1.0 ||
                  !std::isfinite(displacements[i]) || displacements[i] < 0.0;
        result.min_confidence = std::min(result.min_confidence, confidences[i]);
        result.max_displacement = std::max(result.max_displacement, displacements[i]);
    }
    if (invalid) { result.min_confidence = 0.0; result.max_displacement = 0.0; }
    if (mode == Mode::baseline) result.reason = GateReason::baseline;
    else if (mode == Mode::fixed) {
        if ((frame_id - 1) % config_.max_interval == 0) result.reason = GateReason::fixed_interval;
    } else if (!tracks) result.reason = GateReason::no_tracks;
    else if (frame_id - last_detection >= config_.max_interval) result.reason = GateReason::max_interval;
    else if (invalid) result.reason = GateReason::invalid_motion;
    else if (result.min_confidence < config_.min_confidence) result.reason = GateReason::confidence;
    else if (result.max_displacement > config_.max_displacement) result.reason = GateReason::displacement;
    result.should_detect = result.reason != GateReason::none;
    return result;
}

const char* failure_name(MotionFailure failure) noexcept {
    switch (failure) {
        case MotionFailure::none: return "none";
        case MotionFailure::insufficient_points: return "insufficient_points";
        case MotionFailure::nonfinite_forward: return "nonfinite_forward";
        case MotionFailure::invalid_box: return "invalid_box";
    }
    return "invalid_box";
}

const char* reason_name(GateReason reason) noexcept {
    switch (reason) {
        case GateReason::none: return "none";
        case GateReason::baseline: return "baseline";
        case GateReason::fixed_interval: return "fixed_interval";
        case GateReason::no_tracks: return "no_tracks";
        case GateReason::max_interval: return "max_interval";
        case GateReason::invalid_motion: return "invalid_motion";
        case GateReason::confidence: return "confidence";
        case GateReason::displacement: return "displacement";
    }
    return "invalid_motion";
}

}  // namespace edge_gating
