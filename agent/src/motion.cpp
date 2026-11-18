#include "motion.hpp"

#include <algorithm>
#include <cmath>

namespace bunnycam {

Motion MotionDetector::motion_level(std::span<const std::uint8_t> frame) {
    constexpr int pool = kFrameSide / kPooledSide;
    constexpr int grid = 8;
    constexpr int cell_side = kPooledSide / grid;
    constexpr float pixel_diff = 15.0f;          // on pooled luma
    constexpr double cell_hot = 0.25;            // fraction of a cell changed to call it hot
    constexpr int lighting_cells = 64 * 7 / 10;  // int(GRID * GRID * 0.7): scene-wide change
    std::array<float, kPooledSide * kPooledSide> pooled{};
    for (int y = 0; y < kPooledSide; ++y) {
        for (int x = 0; x < kPooledSide; ++x) {
            float sum = 0.0f;
            for (int dy = 0; dy < pool; ++dy) {
                for (int dx = 0; dx < pool; ++dx) {
                    const auto i = ((y * pool + dy) * kFrameSide + x * pool + dx) * 3;
                    sum += (static_cast<float>(frame[i]) + 2.0f * static_cast<float>(frame[i + 1]) +
                            static_cast<float>(frame[i + 2])) /
                           4.0f;
                }
            }
            pooled[y * kPooledSide + x] = sum / (pool * pool);
        }
    }

    if (!prev_) {
        prev_ = pooled;
        return {};
    }

    std::array<float, kPooledSide * kPooledSide> diffs;
    for (std::size_t i = 0; i < pooled.size(); ++i) {
        diffs[i] = pooled[i] - (*prev_)[i];
    }
    prev_ = pooled;

    auto sorted = diffs;
    const auto middle = sorted.begin() + (sorted.size() >> 1);
    std::nth_element(sorted.begin(), middle, sorted.end());
    const float median = *middle;

    int changed = 0;
    std::array<int, grid * grid> cell_changed{};
    for (int y = 0; y < kPooledSide; ++y) {
        for (int x = 0; x < kPooledSide; ++x) {
            if (std::abs(diffs[y * kPooledSide + x] - median) > pixel_diff) {
                ++changed;
                ++cell_changed[(y / cell_side) * grid + x / cell_side];
            }
        }
    }

    const int hot_cells = std::count_if(cell_changed.begin(), cell_changed.end(), [](int count) {
        return count >= cell_side * cell_side * cell_hot;
    });
    const bool lighting = hot_cells >= lighting_cells;
    const double level = lighting ? 0.0 : (static_cast<double>(changed) / pooled.size()) * 100;
    return {level, hot_cells, lighting};
}

}  // namespace bunnycam
