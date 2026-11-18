#include "decision.hpp"

#include <algorithm>
#include <cmath>

namespace bunnycam {
namespace {

bool is_resting(std::size_t index) {
    return std::any_of(kRestingLabels.begin(), kRestingLabels.end(), [&](const auto& label) {
        return static_cast<std::size_t>(std::find(kLabels.begin(), kLabels.end(), label) -
                                        kLabels.begin()) == index;
    });
}

int confidence(double value) { return static_cast<int>(std::nearbyint(value)); }

nlohmann::json nullable_label(const std::optional<std::string>& label) {
    return label ? nlohmann::json(*label) : nlohmann::json(nullptr);
}

}  // namespace

nlohmann::json to_json(const Decision& d) {
    const auto& p = d.prediction;
    return {{"motion",
             {{"level", d.motion.level},
              {"hotCells", d.motion.hot_cells},
              {"lighting", d.motion.lighting}}},
            {"prediction",
             {{"label", nullable_label(p.label)},
              {"confidence", p.confidence},
              {"windowFill", p.window_fill},
              {"interest", p.interest},
              {"alertLabel", nullable_label(p.alert_label)},
              {"alertConfidence", p.alert_confidence}}},
            {"warm", d.warm},
            {"candidate", d.candidate},
            {"behaviorAlert", d.behavior_alert},
            {"alert", d.alert ? nlohmann::json{{"label", d.alert->label},
                                               {"confidence", d.alert->confidence}}
                              : nlohmann::json(nullptr)},
            {"motionEvent", d.motion_event},
            {"publishPrediction", d.publish_prediction},
            {"threshold", d.threshold},
            {"action", d.action}};
}

double interest_score(std::span<const float> probs) {
    float result = 0.0f;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        if (!is_resting(i) && probs[i] > result) result = probs[i];
    }
    // NumPy keeps the selected float32 scalar in float32 for this product.
    return static_cast<double>(result * 100.0f);
}

DecisionLoop::DecisionLoop(DecisionParams params, FramePredictor* predictor)
    : params_(params), predictor_(predictor) {}

bool DecisionLoop::apply_clipping_config(const ClippingConfig& config) {
    const bool changed = clipping_ != config.clipping;
    clipping_ = config.clipping;
    if (config.threshold) params_.clip_threshold = *config.threshold;
    if (config.cooldown) params_.clip_cooldown_sec = *config.cooldown;
    if (config.streak) params_.clip_streak = std::max(1, *config.streak);
    dry_run_ = config.dry_run;
    if (changed) {
        motion_streak_ = 0;
        streak_count_ = 0;
        streak_scores_.clear();
        last_label_.reset();
        last_capture_at_ = 0.0;
        if (predictor_) predictor_->clear();
    }
    return changed;
}

void DecisionLoop::reset() {
    motion_ = MotionDetector{};
    if (predictor_) predictor_->clear();
    last_label_.reset();
    motion_streak_ = 0;
    streak_count_ = 0;
    streak_scores_.clear();
    last_capture_at_ = 0.0;
}

Prediction DecisionLoop::classify(std::span<const std::uint8_t> frame) {
    Prediction result;
    if (!predictor_) return result;
    const auto probs = predictor_->push_frame(frame);
    result.window_fill = predictor_->window_fill();
    if (!probs) return result;
    const auto best = std::max_element(probs->begin(), probs->end()) - probs->begin();
    std::size_t alert_index = 0;
    const auto alert_score = [&](std::size_t i) {
        return is_resting(i) ? -1.0 : static_cast<double>((*probs)[i]);
    };
    for (std::size_t i = 1; i < kLabels.size(); ++i) {
        if (alert_score(i) > alert_score(alert_index)) alert_index = i;
    }
    result.label = kLabels[best];
    // Widen before multiplying to preserve Python's float(float32) arithmetic.
    result.confidence = confidence(static_cast<double>((*probs)[best]) * 100);
    result.interest = interest_score(*probs);
    result.alert_label = kLabels[alert_index];
    result.alert_confidence = confidence(static_cast<double>((*probs)[alert_index]) * 100);
    return result;
}

Decision DecisionLoop::handle_frame(std::span<const std::uint8_t> frame, double now) {
    return decide(motion_.motion_level(frame), frame, now);
}

Decision DecisionLoop::decide(const Motion& motion, std::span<const std::uint8_t> frame,
                              double now) {
    Decision d;
    d.motion = motion;
    if (predictor_ && (!clipping_ || params_.clip_run_model)) d.prediction = classify(frame);
    const auto& pred = d.prediction;
    d.warm = static_cast<int>(pred.window_fill) >= params_.embed_window;
    d.candidate = !clipping_ && d.warm && pred.interest >= params_.interest_thresh &&
                  motion.level >= params_.motion_floor;
    if (d.candidate) {
        ++streak_count_;
        auto entry =
            std::find_if(streak_scores_.begin(), streak_scores_.end(),
                         [&](const auto& score) { return score.first == pred.alert_label; });
        if (entry == streak_scores_.end()) {
            streak_scores_.emplace_back(*pred.alert_label, pred.alert_confidence);
        } else {
            entry->second += pred.alert_confidence;
        }
    } else {
        streak_count_ = 0;
        streak_scores_.clear();
    }
    d.behavior_alert = d.candidate && streak_count_ >= params_.alert_streak;
    if (d.behavior_alert) {
        const auto best =
            std::max_element(streak_scores_.begin(), streak_scores_.end(),
                             [](const auto& a, const auto& b) { return a.second < b.second; });
        d.alert = Alert{best->first, best->first == pred.alert_label
                                         ? pred.alert_confidence
                                         : confidence(best->second / streak_count_)};
    }
    d.threshold = clipping_ ? params_.clip_threshold : params_.motion_capture;
    const int streak_needed = clipping_ ? params_.clip_streak : params_.motion_streak;
    const double cooldown = clipping_ ? params_.clip_cooldown_sec : params_.capture_cooldown_sec;
    motion_streak_ = motion.level >= d.threshold ? motion_streak_ + 1 : 0;
    d.motion_event = motion_streak_ >= streak_needed;
    const bool capture_ready = now - last_capture_at_ >= cooldown;
    d.publish_prediction = !clipping_ && d.warm && pred.label != last_label_;
    if (d.publish_prediction) last_label_ = pred.label;
    if ((d.behavior_alert || d.motion_event) && capture_ready) {
        last_capture_at_ = now;
        streak_count_ = 0;
        streak_scores_.clear();
        motion_streak_ = 0;
        d.action = clipping_ ? "clip" : (d.behavior_alert ? "alert" : "motion_capture");
    }
    return d;
}

}  // namespace bunnycam
