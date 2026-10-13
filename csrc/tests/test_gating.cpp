#include "gating.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace edge_gating;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        Engine engine;
        const std::array<float, 12> previous{10,10, 20,10, 30,10, 10,20, 20,20, 30,20};
        auto forward = previous;
        for (std::size_t i = 0; i < 6; ++i) { forward[2*i] += 3; forward[2*i+1] += 4; }
        const std::array<std::uint8_t, 6> status{1,1,1,1,1,1};
        const std::array<float, 4> box{0,0,30,40};
        const auto motion = engine.assess_motion(previous.data(), forward.data(), previous.data(),
            status.data(), status.data(), 6, box, 0.8, 100, 100);
        require(motion.valid_points == 6, "Expected six surviving points");
        require(motion.delta[0] == 3 && motion.delta[1] == 4, "Median translation mismatch");
        require(std::abs(motion.displacement - 0.1) < 1e-12, "Normalization mismatch");
        require(motion.confidence == 0.8, "Perfect motion must retain prior confidence");
        require(motion.box == std::array<float,4>{3,4,33,44}, "Translated box mismatch");
        const double confidence = 0.6, displacement = 0.15;
        require(!engine.decide(2, 1, &confidence, &displacement, 1).should_detect,
                "Equality at thresholds must skip");
        require(engine.decide(11, 1, &confidence, &displacement, 1).reason == GateReason::max_interval,
                "Interval boundary mismatch");
        require(engine.decide(1, 0, nullptr, nullptr, 0).reason == GateReason::no_tracks,
                "No-track refresh mismatch");
        const auto empty = engine.assess_motion(nullptr, nullptr, nullptr, nullptr, nullptr,
                                                0, box, 1.0, 100, 100);
        require(empty.failure == MotionFailure::insufficient_points, "Empty motion must fail closed");
        forward[0] = std::numeric_limits<float>::quiet_NaN();
        require(engine.assess_motion(previous.data(), forward.data(), previous.data(), status.data(),
                status.data(), 6, box, 1.0, 100, 100).failure == MotionFailure::nonfinite_forward,
                "Nonfinite motion must fail closed");
        bool rejected = false;
        try { Config invalid; invalid.fb_threshold = 0; Engine bad(invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Invalid threshold must be rejected");
        std::cout << "Native gating correctness checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
