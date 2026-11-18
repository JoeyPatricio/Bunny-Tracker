#include "inference.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace bunnycam {

namespace {

constexpr std::size_t kSide = 224;
constexpr std::size_t kPlane = kSide * kSide;
constexpr std::array<float, 3> kMean = {0.485f, 0.456f, 0.406f};
constexpr std::array<float, 3> kStd = {0.229f, 0.224f, 0.225f};

// ORT allows one Env per process. Sessions from every predictor share it.
Ort::Env& ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bunnycam");
    return env;
}

Ort::Session open_session(const std::filesystem::path& path) {
    // Loading by path lets ORT resolve head.onnx.data next to head.onnx.
    return Ort::Session(ort_env(), path.c_str(), Ort::SessionOptions{});
}

std::string input_name(const Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    return session.GetInputNameAllocated(0, allocator).get();
}

std::string output_name(const Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    return session.GetOutputNameAllocated(0, allocator).get();
}

}  // namespace

std::vector<float> normalize(std::span<const std::uint8_t> frame) {
    std::vector<float> out(3 * kPlane);
    for (std::size_t i = 0; i < kPlane; ++i) {
        for (std::size_t c = 0; c < 3; ++c) {
            const float x = static_cast<float>(frame[i * 3 + c]) / 255.0f;
            out[c * kPlane + i] = (x - kMean[c]) / kStd[c];
        }
    }
    return out;
}

PromotedModel read_promoted(const std::filesystem::path& models_dir) {
    const auto path = models_dir / "promoted.json";
    std::ifstream in(path);
    if (!in) throw std::runtime_error("no promoted model");
    std::ostringstream text;
    text << in.rdbuf();
    nlohmann::json record;
    try {
        record = nlohmann::json::parse(text.str());
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error(path.string() + " is not valid JSON: " + e.what());
    }
    const auto manifest = record.find("manifest");
    if (manifest == record.end() || !manifest->is_object()) {
        throw std::runtime_error(path.string() + " has no manifest");
    }
    const auto labels = manifest->find("labels");
    if (labels == manifest->end() || *labels != nlohmann::json(kLabels)) {
        throw std::runtime_error(
            "Promoted model vocabulary differs from the current ethogram. Retrain before use.");
    }
    const auto view = manifest->find("view");
    if (view != manifest->end() && *view != "full") {
        throw std::runtime_error(
            "The promoted crop model needs ROI preprocessing that the agent does not implement.");
    }
    // is_number_integer() is false for JSON booleans and floats, matching
    // Python's `type(window) is not int`.
    const auto window = manifest->find("window");
    if (window == manifest->end() || !window->is_number_integer() || *window <= 0 ||
        *window > std::numeric_limits<int>::max()) {
        throw std::runtime_error("Promoted model must record a positive integer window.");
    }
    return PromotedModel{kLabels, window->get<int>()};
}

struct OnnxPredictor::Impl {
    Ort::Session backbone;
    Ort::Session head;
    std::string backbone_input;
    std::string backbone_output;
    std::string head_input;
    std::string head_output;
    int num_classes;
    std::size_t window;
    std::deque<std::vector<float>> buffer;
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Impl(const std::filesystem::path& backbone_path, const std::filesystem::path& head_path,
         int window_size)
        : backbone(open_session(backbone_path)),
          head(open_session(head_path)),
          backbone_input(input_name(backbone)),
          backbone_output(output_name(backbone)),
          head_input(input_name(head)),
          head_output(output_name(head)),
          num_classes(static_cast<int>(
              head.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape().back())),
          window(static_cast<std::size_t>(window_size)) {}

    std::vector<float> run(Ort::Session& session, const std::string& in_name,
                           const std::string& out_name, std::span<const float> data,
                           const std::array<std::int64_t, 4>& dims, std::size_t rank) {
        // ORT reads inputs without writing them, so the const_cast is safe.
        auto input = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(data.data()),
                                                     data.size(), dims.data(), rank);
        const char* in_names[] = {in_name.c_str()};
        const char* out_names[] = {out_name.c_str()};
        auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
        const float* begin = outputs[0].GetTensorData<float>();
        const auto count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        return std::vector<float>(begin, begin + count);
    }
};

OnnxPredictor::OnnxPredictor(const std::filesystem::path& backbone,
                             const std::filesystem::path& head, int window)
    : impl_(std::make_unique<Impl>(backbone, head, window)) {}

OnnxPredictor::~OnnxPredictor() = default;

std::optional<std::vector<float>> OnnxPredictor::push_frame(std::span<const std::uint8_t> frame) {
    impl_->buffer.push_back(embed(normalize(frame)));
    if (impl_->buffer.size() > impl_->window) impl_->buffer.pop_front();
    if (impl_->buffer.size() < impl_->window) return std::nullopt;
    const std::vector<std::vector<float>> embeddings(impl_->buffer.begin(), impl_->buffer.end());
    return softmax(classify(embeddings));
}

std::size_t OnnxPredictor::window_fill() const { return impl_->buffer.size(); }

void OnnxPredictor::clear() { impl_->buffer.clear(); }

int OnnxPredictor::num_classes() const { return impl_->num_classes; }

std::vector<float> OnnxPredictor::embed(std::span<const float> pixel_values) {
    return impl_->run(impl_->backbone, impl_->backbone_input, impl_->backbone_output, pixel_values,
                      {1, 3, kSide, kSide}, 4);
}

std::vector<float> OnnxPredictor::classify(const std::vector<std::vector<float>>& window) {
    const std::size_t dim = window.front().size();
    std::vector<float> flat;
    flat.reserve(window.size() * dim);
    for (const auto& embedding : window)
        flat.insert(flat.end(), embedding.begin(), embedding.end());
    return impl_->run(
        impl_->head, impl_->head_input, impl_->head_output, flat,
        {1, static_cast<std::int64_t>(window.size()), static_cast<std::int64_t>(dim), 0}, 3);
}

std::vector<float> softmax(std::span<const float> logits) {
    const float max = *std::max_element(logits.begin(), logits.end());
    std::vector<float> out(logits.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        out[i] = std::exp(logits[i] - max);
        sum += out[i];
    }
    for (auto& v : out) v /= sum;
    return out;
}

}  // namespace bunnycam
