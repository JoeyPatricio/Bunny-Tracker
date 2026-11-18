#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdlib>

#include "inference.hpp"
#include "support.hpp"

using namespace bunnycam;
using Catch::Matchers::WithinAbs;
using nlohmann::json;

// ORT 1.29 leaks 14 bytes inside Ort::Env's constructor even when the Env is
// destroyed. A two-line program that only constructs an Env reproduces it.
extern "C" const char* __lsan_default_suppressions() { return "leak:libonnxruntime.so"; }

namespace {

constexpr std::size_t kFrameBytes = 224 * 224 * 3;
constexpr double kTolerance = 1e-5;

const json& fixture() {
    static const json data = test::read_json(test::kFixtures / "inference.json");
    return data;
}

const test::NpyU8& frames() {
    static const test::NpyU8 data = test::read_npy_u8(test::kMotionFrames);
    return data;
}

std::span<const std::uint8_t> frame(std::size_t i) {
    return std::span(frames().data).subspan(i * kFrameBytes, kFrameBytes);
}

OnnxPredictor make_predictor(int window) {
    return OnnxPredictor(test::kModelsDir / "backbone_int8.onnx", test::kModelsDir / "head.onnx",
                         window);
}

void require_close(const std::vector<float>& actual, const json& expected) {
    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        CHECK_THAT(actual[i], WithinAbs(expected[i].get<double>(), kTolerance));
    }
}

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        std::string pattern = (std::filesystem::temp_directory_path() / "bunny-deployment-XXXXXX");
        if (!mkdtemp(pattern.data())) throw std::runtime_error("mkdtemp failed");
        path = pattern;
    }
    ~TempDir() { std::filesystem::remove_all(path); }
};

void write_promoted(const std::filesystem::path& dir, const json& record) {
    std::ofstream(dir / "promoted.json") << record.dump();
}

json good_record() {
    return {{"manifest",
             {{"labels", kLabels}, {"window", 12}, {"view", "full"}, {"cv_arm", "gru@albu_v1"}}},
            {"gate_result", {{"cv", {{"gru_macro_f1", 0.61}}}}}};
}

std::string promoted_error(const json& record) {
    TempDir dir;
    write_promoted(dir.path, record);
    try {
        read_promoted(dir.path);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return "";
}

}  // namespace

TEST_CASE("normalize matches the Python samples bit for bit") {
    for (std::size_t i = 0; i < fixture()["frames"].size(); ++i) {
        const auto pixels = normalize(frame(i));
        REQUIRE(pixels.size() == 3 * 224 * 224);
        for (const auto& [index, value] : fixture()["frames"][i]["normalize"].items()) {
            CHECK(pixels[std::stoul(index)] == static_cast<float>(value.get<double>()));
        }
    }
}

TEST_CASE("embeddings match the Python backbone") {
    auto predictor = make_predictor(8);
    for (std::size_t i = 0; i < fixture()["frames"].size(); ++i) {
        require_close(predictor.embed(normalize(frame(i))), fixture()["frames"][i]["embedding"]);
    }
}

// Pushes every fixture frame through a fresh predictor and checks warmup,
// eviction and probabilities against fixture key `probs_key`.
void check_push_frames(OnnxPredictor& predictor, std::size_t window, const char* probs_key) {
    for (std::size_t i = 0; i < fixture()["frames"].size(); ++i) {
        const auto probs = predictor.push_frame(frame(i));
        const auto& expected = fixture()["frames"][i][probs_key];
        if (i + 1 < window) {
            CHECK_FALSE(probs.has_value());
            CHECK(expected.is_null());
            CHECK(predictor.window_fill() == i + 1);
        } else {
            REQUIRE(probs.has_value());
            require_close(*probs, expected);
            CHECK(predictor.window_fill() == window);
        }
    }
}

TEST_CASE("push_frame waits for a full window, then returns the Python probabilities") {
    const int window = fixture()["window"].get<int>();
    REQUIRE(window == 8);
    auto predictor = make_predictor(window);
    check_push_frames(predictor, static_cast<std::size_t>(window), "probs");
    predictor.clear();
    CHECK(predictor.window_fill() == 0);
    CHECK_FALSE(predictor.push_frame(frame(0)).has_value());
}

TEST_CASE("push_frame matches the Python probabilities at a three-frame window") {
    const int window = fixture()["altWindow"].get<int>();
    REQUIRE(window == 3);
    auto predictor = make_predictor(window);
    check_push_frames(predictor, static_cast<std::size_t>(window), "probsAltWindow");
}

TEST_CASE("classify matches the Python head logits") {
    auto predictor = make_predictor(8);
    const auto& records = fixture()["frames"];
    for (std::size_t end = 8; end <= records.size(); ++end) {
        std::vector<std::vector<float>> window;
        for (std::size_t i = end - 8; i < end; ++i) {
            window.push_back(records[i]["embedding"].get<std::vector<float>>());
        }
        require_close(predictor.classify(window), records[end - 1]["logits"]);
    }
}

TEST_CASE("num_classes reads the on-disk head's output width") {
    CHECK(make_predictor(8).num_classes() == 5);
    CHECK(fixture()["numClasses"] == 5);
    // The backbone's [1,1280] output stands in for a head of another width.
    const OnnxPredictor wide(test::kModelsDir / "backbone_int8.onnx",
                             test::kModelsDir / "backbone_int8.onnx", 8);
    CHECK(wide.num_classes() == 1280);
}

TEST_CASE("softmax sums to one and ignores a constant shift") {
    const std::vector<float> logits = {1.5f, -2.0f, 0.25f, 3.0f, -0.75f};
    const auto probs = softmax(logits);
    float sum = 0.0f;
    for (float p : probs) sum += p;
    CHECK_THAT(sum, WithinAbs(1.0, 1e-6));
    std::vector<float> shifted;
    for (float v : logits) shifted.push_back(v + 100.0f);
    const auto shifted_probs = softmax(shifted);
    for (std::size_t i = 0; i < probs.size(); ++i) {
        CHECK_THAT(shifted_probs[i], WithinAbs(probs[i], 1e-6));
    }
}

TEST_CASE("read_promoted accepts a full-view model with the current labels") {
    TempDir dir;
    write_promoted(dir.path, good_record());
    const auto promoted = read_promoted(dir.path);
    CHECK(promoted.labels == kLabels);
    CHECK(promoted.window == 12);

    auto record = good_record();
    record["manifest"].erase("view");
    write_promoted(dir.path, record);
    CHECK(read_promoted(dir.path).window == 12);
}

TEST_CASE("read_promoted refuses a missing promotion") {
    TempDir dir;
    try {
        read_promoted(dir.path);
        FAIL("expected an exception");
    } catch (const std::runtime_error& e) {
        CHECK(std::string(e.what()) == "no promoted model");
    }
}

TEST_CASE("read_promoted refuses stale vocabularies") {
    const std::string message =
        "Promoted model vocabulary differs from the current ethogram. Retrain before use.";
    auto record = good_record();
    record["manifest"]["labels"] = {"normal"};
    CHECK(promoted_error(record) == message);

    auto reordered = kLabels;
    std::swap(reordered[0], reordered[1]);
    record["manifest"]["labels"] = reordered;
    CHECK(promoted_error(record) == message);

    record["manifest"].erase("labels");
    CHECK(promoted_error(record) == message);
}

TEST_CASE("read_promoted refuses crop models") {
    auto record = good_record();
    record["manifest"]["view"] = "crop";
    CHECK(promoted_error(record) ==
          "The promoted crop model needs ROI preprocessing that the agent does not implement.");
}

TEST_CASE("read_promoted refuses windows that are not positive integers") {
    const std::string message = "Promoted model must record a positive integer window.";
    for (const json& window : {json(0), json(-3), json(true), json(12.0), json("12"), json()}) {
        auto record = good_record();
        record["manifest"]["window"] = window;
        INFO("window " << window.dump());
        CHECK(promoted_error(record) == message);
    }
    auto record = good_record();
    record["manifest"].erase("window");
    CHECK(promoted_error(record) == message);
}
