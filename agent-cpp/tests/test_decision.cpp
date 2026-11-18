#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <memory>

#include "decision.hpp"
#include "support.hpp"

namespace {
using namespace bunnycam;
using nlohmann::json;

class ScriptedPredictor : public FramePredictor {
public:
    explicit ScriptedPredictor(int window) : window_(window) {}
    std::optional<std::vector<float>> push_frame(std::span<const std::uint8_t>) override {
        fill_ = std::min(fill_ + 1, window_);
        return fill_ < window_ ? std::nullopt : next_probs;
    }
    std::size_t window_fill() const override { return fill_; }
    void clear() override { fill_ = 0; }
    std::optional<std::vector<float>> next_probs;

private:
    std::size_t window_;
    std::size_t fill_ = 0;
};

DecisionParams read_params(const json& j) {
    DecisionParams p;
    p.embed_window = j.value("embed_window", p.embed_window);
    p.interest_thresh = j.value("interest_thresh", p.interest_thresh);
    p.motion_floor = j.value("motion_floor", p.motion_floor);
    p.alert_streak = j.value("alert_streak", p.alert_streak);
    p.motion_capture = j.value("motion_capture", p.motion_capture);
    p.motion_streak = j.value("motion_streak", p.motion_streak);
    p.capture_cooldown_sec = j.value("capture_cooldown_sec", p.capture_cooldown_sec);
    p.clip_threshold = j.value("clip_threshold", p.clip_threshold);
    p.clip_streak = j.value("clip_streak", p.clip_streak);
    p.clip_cooldown_sec = j.value("clip_cooldown_sec", p.clip_cooldown_sec);
    p.clip_run_model = j.value("clip_run_model", p.clip_run_model);
    return p;
}

ClippingConfig read_config(const json& j) {
    ClippingConfig c;
    c.clipping = j.at("clipping");
    if (j.contains("threshold") && !j.at("threshold").is_null()) c.threshold = j.at("threshold");
    if (j.contains("cooldown") && !j.at("cooldown").is_null()) c.cooldown = j.at("cooldown");
    if (j.contains("streak") && !j.at("streak").is_null()) c.streak = j.at("streak");
    c.dry_run = j.value("dry_run", false);
    return c;
}

TEST_CASE("DecisionLoop replays every Python golden operation", "[decision][golden]") {
    std::ifstream input(bunnycam::test::kFixtures / "decisions.jsonl");
    REQUIRE(input.is_open());
    std::unique_ptr<ScriptedPredictor> predictor;
    std::unique_ptr<DecisionLoop> loop;
    std::string scenario;
    std::vector<std::uint8_t> frame(kFrameBytes, 0);
    std::size_t line_number = 0;
    for (std::string line; std::getline(input, line);) {
        ++line_number;
        const auto record = json::parse(line);
        const auto op = record.at("op").get<std::string>();
        if (op == "new") scenario = record.at("scenario");
        INFO("scenario: " << scenario << ", line: " << line_number);
        if (op == "new") {
            const auto params = read_params(record.at("params"));
            loop.reset();
            predictor = record.at("model").get<bool>()
                            ? std::make_unique<ScriptedPredictor>(params.embed_window)
                            : nullptr;
            loop = std::make_unique<DecisionLoop>(params, predictor.get());
        } else {
            REQUIRE(loop);
            if (op == "config") {
                REQUIRE(loop->apply_clipping_config(read_config(record.at("args"))) ==
                        record.at("result").get<bool>());
            } else if (op == "reset") {
                loop->reset();
            } else if (op == "frame") {
                if (predictor) {
                    predictor->next_probs =
                        record.at("probs").is_null()
                            ? std::nullopt
                            : std::optional(record.at("probs").get<std::vector<float>>());
                }
                const auto& m = record.at("motion");
                const Motion motion{m.at("level"), m.at("hotCells"), m.at("lighting")};
                const auto actual =
                    bunnycam::to_json(loop->decide(motion, frame, record.at("now")));
                REQUIRE(actual == record.at("expect"));
            } else {
                FAIL("unknown operation: " << op);
            }
        }
    }
    REQUIRE(input.eof());
    REQUIRE(line_number > 0);
}

TEST_CASE("interest_score excludes resting labels and preserves float32 arithmetic", "[decision]") {
    std::vector<float> probs(kLabels.size(), 0.0f);
    for (const auto& label : kRestingLabels) {
        probs.at(std::find(kLabels.begin(), kLabels.end(), label) - kLabels.begin()) = 0.9f;
    }
    REQUIRE(interest_score(probs) == 0.0);
    probs[0] = 0.1f;
    probs[1] = 0.3f;
    REQUIRE(interest_score(probs) == static_cast<double>(0.3f * 100.0f));
    std::fill(probs.begin(), probs.end(), -0.5f);
    REQUIRE(interest_score(probs) == 0.0);
    probs[0] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(interest_score(probs) == 0.0);
    REQUIRE(interest_score({}) == 0.0);
}

TEST_CASE("DecisionLoop rounds confidence half to even after widening", "[decision]") {
    DecisionParams p;
    p.embed_window = 1;
    ScriptedPredictor predictor(1);
    DecisionLoop loop(p, &predictor);
    // These binary fractions produce exact halfway percentages.
    for (const auto& [prob, expected] : std::array<std::pair<float, int>, 4>{
             {{0.125f, 12}, {0.375f, 38}, {0.625f, 62}, {0.875f, 88}}}) {
        predictor.next_probs = std::vector<float>(kLabels.size(), 0.0f);
        (*predictor.next_probs)[0] = prob;
        const auto d = loop.decide({}, {}, 0.0);
        REQUIRE(d.prediction.confidence == expected);
        REQUIRE(d.prediction.alert_confidence == expected);
    }
    predictor.next_probs = std::vector<float>(kLabels.size(), 0.0f);
    (*predictor.next_probs)[0] = 0.015f;
    REQUIRE(loop.decide({}, {}, 0.0).prediction.confidence == 1);
}

TEST_CASE("DecisionLoop rounds streak averages half to even and keeps first ties", "[decision]") {
    DecisionParams p;
    p.embed_window = 1;
    p.interest_thresh = 0;
    p.alert_streak = 2;
    p.capture_cooldown_sec = 100;
    ScriptedPredictor predictor(1);
    DecisionLoop loop(p, &predictor);
    const Motion motion{1.0, 0, false};
    const auto feed = [&](std::size_t index, float prob) {
        predictor.next_probs = std::vector<float>(kLabels.size(), 0.0f);
        (*predictor.next_probs)[index] = prob;
        return loop.decide(motion, {}, 0.0);
    };
    feed(1, 0.25f);
    auto d = feed(0, 0.125f);
    REQUIRE(d.alert.has_value());
    REQUIRE(d.alert->label == kLabels[1]);
    REQUIRE(d.alert->confidence == 12);
    loop.reset();
    feed(1, 0.75f);
    d = feed(0, 0.125f);
    REQUIRE(d.alert->confidence == 38);
    loop.reset();
    feed(1, 0.25f);
    d = feed(0, 0.25f);
    REQUIRE(d.alert->label == kLabels[1]);
    predictor.next_probs = std::vector<float>(kLabels.size(), 0.25f);
    d = loop.decide(motion, {}, 0.0);
    REQUIRE(d.prediction.label == kLabels[0]);
    REQUIRE(d.prediction.alert_label == kLabels[0]);
}
}  // namespace
