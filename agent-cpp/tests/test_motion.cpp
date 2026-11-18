#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "motion.hpp"
#include "support.hpp"

using namespace bunnycam;

TEST_CASE("motion matches Python fixtures across the continuous stream", "[motion]") {
    const auto frames = test::read_npy_u8(test::kMotionFrames);
    const auto expected = test::read_json(test::kFixtures / "motion.json");
    REQUIRE(frames.shape == std::vector<std::size_t>{67, kFrameSide, kFrameSide, 3});
    REQUIRE(expected.size() == frames.shape[0]);

    MotionDetector detector;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CAPTURE(i);
        const auto result = detector.motion_level(
            std::span<const std::uint8_t>(frames.data).subspan(i * kFrameBytes, kFrameBytes));
        CHECK(std::abs(result.level - expected[i]["level"].get<double>()) <= 1e-9);
        CHECK(result.hot_cells == expected[i]["hotCells"].get<int>());
        CHECK(result.lighting == expected[i]["lighting"].get<bool>());
    }
}

TEST_CASE("motion matches JS ground truth across the continuous stream", "[motion]") {
    const auto frames = test::read_npy_u8(test::kMotionFrames);
    const auto expected = test::read_json(test::kRepoDir / "server" / "tests" / "fixtures" /
                                          "motion_fixture_ground_truth.json");
    REQUIRE(frames.shape == std::vector<std::size_t>{67, kFrameSide, kFrameSide, 3});
    REQUIRE(expected.size() == frames.shape[0]);

    MotionDetector detector;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CAPTURE(i);
        const auto result = detector.motion_level(
            std::span<const std::uint8_t>(frames.data).subspan(i * kFrameBytes, kFrameBytes));
        CHECK(std::abs(result.level - expected[i]["level"].get<double>()) <= 0.1);
        CHECK(result.hot_cells == expected[i]["hotCells"].get<int>());
        CHECK(result.lighting == expected[i]["lighting"].get<bool>());
    }
}

TEST_CASE("a fresh motion detector returns zeros on its first frame", "[motion]") {
    const auto frames = test::read_npy_u8(test::kMotionFrames);
    MotionDetector detector;
    const auto result =
        detector.motion_level(std::span<const std::uint8_t>(frames.data).first(kFrameBytes));
    CHECK(result.level == 0.0);
    CHECK(result.hot_cells == 0);
    CHECK_FALSE(result.lighting);
}
