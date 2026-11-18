// The per-frame decision loop and interest_score, the alert gate that
// server/shared/labels.py copies for training/calibrate_gate.py. Pure logic:
// the caller passes `now`, so tests replay golden traces.
#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "inference.hpp"
#include "motion.hpp"

namespace bunnycam {

struct DecisionParams {
    int embed_window = 8;
    double interest_thresh = 60;
    double motion_floor = 1.0;
    int alert_streak = 3;
    double motion_capture = 2.5;
    int motion_streak = 3;
    double capture_cooldown_sec = 30;
    double clip_threshold = 2.5;
    int clip_streak = 1;
    double clip_cooldown_sec = 8;
    bool clip_run_model = false;
};

struct ClippingConfig {
    bool clipping = false;
    std::optional<double> threshold;
    std::optional<double> cooldown;
    std::optional<int> streak;
    bool dry_run = false;
};

struct Prediction {
    std::optional<std::string> label;
    int confidence = 0;
    std::size_t window_fill = 0;
    double interest = 0.0;
    std::optional<std::string> alert_label;
    int alert_confidence = 0;
};

struct Alert {
    std::string label;
    int confidence = 0;
};

struct Decision {
    Motion motion;
    Prediction prediction;
    bool warm = false;
    bool candidate = false;
    bool behavior_alert = false;
    std::optional<Alert> alert;
    bool motion_event = false;
    bool publish_prediction = false;
    double threshold = 0.0;
    std::string action = "none";  // none | alert | motion_capture | clip
};

// Same keys and shapes as the Python dict (camelCase, null for None).
nlohmann::json to_json(const Decision& decision);

double interest_score(std::span<const float> probs);

class DecisionLoop {
public:
    // predictor may be null: motion-only, as when no model is promoted.
    DecisionLoop(DecisionParams params, FramePredictor* predictor);

    // Returns true when the mode flipped (the caller restarts ffmpeg).
    bool apply_clipping_config(const ClippingConfig& config);

    // Agent._reset_decision_state: new motion detector, empty window, streaks,
    // last label and last capture cleared.
    void reset();

    Decision handle_frame(std::span<const std::uint8_t> frame, double now);

    // handle_frame after the motion step. Tests drive it with recorded motion.
    Decision decide(const Motion& motion, std::span<const std::uint8_t> frame, double now);

    bool clipping() const { return clipping_; }
    bool dry_run() const { return dry_run_; }
    const DecisionParams& params() const { return params_; }

private:
    Prediction classify(std::span<const std::uint8_t> frame);

    DecisionParams params_;
    FramePredictor* predictor_;
    MotionDetector motion_;
    bool clipping_ = false;
    bool dry_run_ = false;
    std::optional<std::string> last_label_;
    int streak_count_ = 0;
    std::vector<std::pair<std::string, double>> streak_scores_;  // insertion order matters
    int motion_streak_ = 0;
    double last_capture_at_ = 0.0;
};

}  // namespace bunnycam
