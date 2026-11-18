// Port of server/inference/{backbone,temporal_head,predictor,deployment}.py
// and shared/frames.normalize: ONNX Runtime sessions, the embedding ring
// buffer, softmax, and the promoted-model checks the agent runs at boot.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace bunnycam {

// Model classes in output-axis order. Mirrors server/shared/labels.py LABELS;
// tests/test_labels.cpp fails if the two drift.
inline const std::vector<std::string> kLabels = {
    "feeding", "grooming",      "locomotion",      "locomotion_rapid",
    "rearing", "resting_lying", "resting_sitting",
};
inline const std::vector<std::string> kRestingLabels = {"resting_lying", "resting_sitting"};

// uint8 HWC RGB 224x224 -> float32 CHW, ImageNet mean/std, same op order as
// shared/frames.normalize.
std::vector<float> normalize(std::span<const std::uint8_t> frame);

// What the decision loop needs from a predictor. A fake implements it in tests.
class FramePredictor {
public:
    virtual ~FramePredictor() = default;
    // Embed one raw frame and push it into the window. Returns class
    // probabilities once the window is full, else nullopt.
    virtual std::optional<std::vector<float>> push_frame(std::span<const std::uint8_t> frame) = 0;
    virtual std::size_t window_fill() const = 0;
    virtual void clear() = 0;
};

// The fields of models/promoted.json the agent uses.
struct PromotedModel {
    std::vector<std::string> labels;
    int window = 0;
};

// Reads <models_dir>/promoted.json. Throws std::runtime_error when it is
// missing, its labels differ from kLabels, its view is not "full", or its
// window is not a positive integer (deployment.py read_promoted).
PromotedModel read_promoted(const std::filesystem::path& models_dir);

// backbone_int8.onnx + head.onnx behind FramePredictor.
class OnnxPredictor : public FramePredictor {
public:
    OnnxPredictor(const std::filesystem::path& backbone, const std::filesystem::path& head,
                  int window);
    ~OnnxPredictor() override;

    std::optional<std::vector<float>> push_frame(std::span<const std::uint8_t> frame) override;
    std::size_t window_fill() const override;
    void clear() override;

    int num_classes() const;  // last dim of the head's output shape
    std::vector<float> embed(std::span<const float> pixel_values);  // [3,224,224] -> [1280]
    std::vector<float> classify(const std::vector<std::vector<float>>& window);  // -> logits

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::vector<float> softmax(std::span<const float> logits);

}  // namespace bunnycam
