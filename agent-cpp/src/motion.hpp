// Port of server/shared/motion.py (itself a port of capture.mjs's
// poolLuma/motionLevel). Arithmetic order matches the Python: pool, diff
// against the previous frame, subtract the lower-middle median, threshold,
// grid-score.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>

namespace bunnycam {

inline constexpr int kFrameSide = 224;
inline constexpr std::size_t kFrameBytes = kFrameSide * kFrameSide * 3;  // RGB24
inline constexpr int kPooledSide = 56;

struct Motion {
    double level = 0.0;  // percent of pooled pixels changed, 0 on a lighting change
    int hot_cells = 0;
    bool lighting = false;
};

// One instance per continuous frame stream. Holds the previous pooled frame.
class MotionDetector {
public:
    // frame: kFrameBytes of 224x224 RGB24, row-major.
    Motion motion_level(std::span<const std::uint8_t> frame);

private:
    std::optional<std::array<float, kPooledSide * kPooledSide>> prev_;
};

}  // namespace bunnycam
