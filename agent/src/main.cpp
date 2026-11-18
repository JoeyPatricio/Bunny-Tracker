// Port of the Agent class and main() from server/agent/capture.py: boot,
// model loading, the monitor state machine, the frame handler, background
// loops, and shutdown on SIGTERM/SIGINT.
#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <mutex>
#include <regex>
#include <stop_token>
#include <string>
#include <thread>

#include "decision.hpp"
#include "ffmpeg.hpp"
#include "inference.hpp"
#include "log.hpp"
#include "server.hpp"

namespace bunnycam {
namespace {

namespace fs = std::filesystem;
using nlohmann::json;

// The binary lives at <repo>/agent/<build dir>/bunnycam-agent.
fs::path repo_root() { return fs::read_symlink("/proc/self/exe").parent_path() / ".." / ".."; }

std::string_view trim(std::string_view s) {
    const auto first = s.find_first_not_of(" \t");
    if (first == std::string_view::npos) return {};
    return s.substr(first, s.find_last_not_of(" \t") - first + 1);
}

// ${NAME} and ${NAME:-default}, which python-dotenv expands in unquoted and
// double-quoted values. Earlier lines are already in the environment, and a
// variable the process was started with wins over the file, as with
// load_dotenv(override=False).
std::string expand(const std::string& value) {
    static const std::regex var(R"(\$\{([^}:]+)(?::-([^}]*))?\})");
    std::string out;
    auto it = value.cbegin();
    for (std::smatch m; std::regex_search(it, value.cend(), m, var); it = m[0].second) {
        out.append(it, m[0].first);
        const char* set = std::getenv(m[1].str().c_str());
        out += set ? set : m[2].str();
    }
    out.append(it, value.cend());
    return out;
}

// python-dotenv's load_dotenv() for what data/.env uses: KEY=VALUE lines,
// optional `export `, # comments, single or double quotes, CRLF endings,
// variable expansion. Never overrides a variable already in the environment.
void load_dotenv(const fs::path& path) {
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::string_view rest = trim(line);
        if (rest.empty() || rest.front() == '#') continue;
        if (rest.starts_with("export ")) rest = trim(rest.substr(7));
        const auto eq = rest.find('=');
        if (eq == std::string_view::npos) continue;
        const std::string key(trim(rest.substr(0, eq)));
        std::string_view raw = trim(rest.substr(eq + 1));
        std::string value;
        if (!raw.empty() && (raw.front() == '"' || raw.front() == '\'')) {
            // Anything after the closing quote, such as a comment, is dropped.
            const auto close = raw.find(raw.front(), 1);
            value = raw.substr(1, close == std::string_view::npos ? close : close - 1);
            if (raw.front() == '"') value = expand(value);
        } else {
            if (const auto hash = raw.find(" #"); hash != std::string_view::npos) {
                raw = trim(raw.substr(0, hash));
            }
            value = expand(std::string(raw));
        }
        setenv(key.c_str(), value.c_str(), 0);
    }
}

std::string env(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    return value ? value : fallback;
}

double now_seconds() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}

// Sleeps for `seconds`, waking early on stop. Returns false once stopped.
bool sleep_for(const std::stop_token& stop, double seconds) {
    std::mutex mu;
    std::condition_variable_any cv;
    std::unique_lock lock(mu);
    cv.wait_for(lock, stop, std::chrono::duration<double>(seconds), [] { return false; });
    return !stop.stop_requested();
}

struct Config {
    fs::path root = repo_root();
    fs::path models_dir = root / "server" / "models";
    std::string camera_format;
    std::string camera_input;
    std::string camera_size;
    std::string camera_fps;
    std::string ffmpeg_path;
    double frame_interval;
    int segment_secs;
    double clip_frame_interval;
    int clip_segment_secs;
    std::optional<std::string> embed_window_override;
    DecisionParams params;
    fs::path seg_dir;
    std::string server_url;
    std::string token;

    static Config from_env() {
        Config c;
        load_dotenv(c.root / "data" / ".env");
        // dshow is gone with Windows support, so the defaults are the Pi's.
        c.camera_format = env("CAMERA_FORMAT", "v4l2");
        if (c.camera_format != "v4l2" && c.camera_format != "lavfi") {
            throw std::runtime_error("CAMERA_FORMAT=" + c.camera_format +
                                     " is not supported. Use v4l2 or lavfi.");
        }
        c.camera_input = env("CAMERA_INPUT", "/dev/video0");
        c.camera_size = env("CAMERA_SIZE", "1280x720");
        c.camera_fps = env("CAMERA_FPS", "30");
        c.ffmpeg_path = env("FFMPEG_PATH", "ffmpeg");
        c.frame_interval = std::stod(env("AGENT_FRAME_SECONDS", "1.5"));
        c.segment_secs = std::stoi(env("AGENT_SEGMENT_SECONDS", "12"));
        c.clip_frame_interval = std::stod(env("AGENT_CLIPPING_FRAME_SECONDS", "0.5"));
        c.clip_segment_secs = std::stoi(env("AGENT_CLIPPING_SEGMENT_SECONDS", "8"));
        if (const char* w = std::getenv("AGENT_EMBED_WINDOW")) c.embed_window_override = w;

        DecisionParams& p = c.params;
        p.embed_window = std::stoi(env("AGENT_EMBED_WINDOW", "8"));
        p.interest_thresh = std::stod(env("AGENT_INTEREST_THRESHOLD", "60"));
        p.motion_floor = std::stod(env("AGENT_MOTION_FLOOR", "1"));
        p.alert_streak = std::stoi(env("AGENT_ALERT_STREAK", "3"));
        p.motion_capture = std::stod(env("AGENT_MOTION_CAPTURE", "2.5"));
        p.motion_streak = std::stoi(env("AGENT_MOTION_STREAK", "3"));
        p.capture_cooldown_sec = std::stod(env("AGENT_CAPTURE_COOLDOWN_SEC", "30"));
        p.clip_streak = std::stoi(env("AGENT_CLIPPING_STREAK", "1"));
        const std::string run_model = env("AGENT_CLIPPING_RUN_MODEL", "0");
        p.clip_run_model =
            run_model != "" && run_model != "0" && run_model != "false" && run_model != "False";

        c.seg_dir = env("AGENT_SEGMENT_DIR", (c.root / "data" / "agent" / "segments").string());
        c.server_url = env("AGENT_SERVER_URL", "http://localhost:3001");
        c.token = env("AGENT_TOKEN", "");
        return c;
    }
};

class Agent {
public:
    explicit Agent(Config config)
        : config_(std::move(config)),
          client_(config_.server_url, config_.token),
          live_client_(config_.server_url, config_.token),
          clips_client_(config_.server_url, config_.token),
          clips_(ClipOptions{.seg_dir = config_.seg_dir,
                             .segment_max_age_sec = config_.segment_secs * 2.5,
                             .clip_segment_secs = static_cast<double>(config_.clip_segment_secs)},
                 clips_client_) {}

    void boot() {
        log("\U0001f407 BunnyCam Agent starting...");
        fs::create_directories(config_.seg_dir);
        client_.verify();
        log("✓ AGENT_TOKEN verified");
        load_models();

        json state;
        try {
            state = client_.get("/api/monitor").json();
        } catch (const std::exception&) {
            state = {{"enabled", true}};
        }
        paused_ = true;
        apply_monitor_state(state);
        if (paused_) log("⏸ Monitoring is OFF — camera idle until enabled from dashboard");
    }

    void poll_monitor_loop(std::stop_token stop) {
        while (sleep_for(stop, 5)) {
            json state;
            try {
                state = client_.get("/api/monitor").json();
            } catch (const std::exception&) {
                continue;  // server unreachable: keep current state
            }
            apply_monitor_state(state);
        }
    }

    void prune_segments_loop(std::stop_token stop) {
        while (sleep_for(stop, 60)) {
            try {
                clips_.prune(now_seconds());
            } catch (const std::exception&) {
            }
        }
    }

    void drain_pending_clips_loop(std::stop_token stop) {
        while (sleep_for(stop, 2)) {
            if (!clips_.has_pending()) continue;
            try {
                clips_.drain_once(now_seconds());
            } catch (const std::exception& err) {
                log(std::string("  clip drain error: ") + err.what());
            }
        }
    }

    // Stops ffmpeg now, before the loops are joined (an in-flight upload can
    // hold one for its full timeout), and keeps the monitor thread from
    // starting another capture meanwhile.
    void begin_shutdown() {
        {
            std::lock_guard lock(capture_mu_);
            shutting_down_ = true;
        }
        stop_capture();
    }

    void stop_capture() {
        std::unique_ptr<FfmpegCapture> capture;
        {
            std::lock_guard lock(capture_mu_);
            capture = std::move(capture_);
        }
        if (capture) capture->stop_and_wait();
    }

private:
    void load_models() {
        log("Loading ONNX models...");
        try {
            const PromotedModel promoted = read_promoted(config_.models_dir);
            if (config_.embed_window_override &&
                std::stoi(*config_.embed_window_override) != promoted.window) {
                throw std::runtime_error(std::format(
                    "AGENT_EMBED_WINDOW={} differs from the promoted window {}. Remove the "
                    "override or retrain with the intended window.",
                    *config_.embed_window_override, promoted.window));
            }
            config_.params.embed_window = promoted.window;
            auto predictor =
                std::make_unique<OnnxPredictor>(config_.models_dir / "backbone_int8.onnx",
                                                config_.models_dir / "head.onnx", promoted.window);
            // The head's output width and the label vocabulary have to agree.
            // After an ethogram change they won't until a retrain lands, and
            // index 3 would keep resolving to a name, just the wrong one.
            if (predictor->num_classes() != static_cast<int>(kLabels.size())) {
                throw std::runtime_error(std::format(
                    "head.onnx predicts {} classes but the ethogram declares {}. Retrain and "
                    "re-export before the classifier can run.",
                    predictor->num_classes(), kLabels.size()));
            }
            predictor_ = std::move(predictor);
        } catch (const std::exception& err) {
            predictor_.reset();
            log(std::format(
                "⚠ ONNX models unavailable ({}) — running motion-only, alerts disabled.",
                err.what()));
            log("  Clipping Mode still works; it never uses the model.");
        }
        loop_ = std::make_unique<DecisionLoop>(config_.params, predictor_.get());
        if (predictor_) {
            std::string classes;
            for (const auto& label : kLabels) classes += (classes.empty() ? "" : ", ") + label;
            log("✓ Models ready (backend: onnxruntime; classes: " + classes + ")");
        }
    }

    void start_capture() {
        stop_capture();
        {
            std::lock_guard lock(capture_mu_);
            if (shutting_down_) return;
        }
        // frame_interval and segment_secs are baked into ffmpeg's args, so
        // Clipping Mode's cadence only takes effect across a restart.
        const bool clipping = loop_->clipping();
        const int segment_secs = clipping ? config_.clip_segment_secs : config_.segment_secs;
        clips_.set_timing(segment_secs * 2.5, config_.clip_segment_secs);
        auto capture = std::make_unique<FfmpegCapture>(
            CaptureConfig{
                .ffmpeg_path = config_.ffmpeg_path,
                .camera_format = config_.camera_format,
                .camera_input = config_.camera_input,
                .camera_size = config_.camera_size,
                .camera_fps = config_.camera_fps,
                .frame_interval = clipping ? config_.clip_frame_interval : config_.frame_interval,
                .segment_secs = segment_secs,
                .seg_dir = config_.seg_dir},
            [this](std::vector<std::uint8_t> frame, FfmpegCapture::Clock::time_point at) {
                on_frame(frame, at);
            },
            [this](std::string jpeg) {
                try {
                    live_client_.post_live_frame(jpeg);
                } catch (const std::exception&) {
                }
            });
        capture->start();
        {
            std::lock_guard lock(capture_mu_);
            if (!shutting_down_) {
                capture_ = std::move(capture);
                return;
            }
        }
        capture->stop_and_wait();  // shutdown began while this one started
    }

    void apply_monitor_state(const json& state) {
        const bool enabled = state.value("enabled", true);
        const bool clipping = state.value("clipping", false);
        const bool mode_changed = clipping != loop_->clipping();
        const bool stopping = !enabled && !paused_;
        const bool resuming = enabled && paused_;
        // Stop the frame consumer before taking the lock it may hold.
        if (stopping || mode_changed) stop_capture();

        auto number = [&](const char* key) -> std::optional<double> {
            const auto it = state.find(key);
            if (it == state.end() || !it->is_number()) return std::nullopt;
            return it->get<double>();
        };
        {
            std::lock_guard lock(decision_mu_);
            loop_->apply_clipping_config({.clipping = clipping,
                                          .threshold = number("clippingMotionThreshold"),
                                          .cooldown = number("clippingCooldownSec"),
                                          .streak = std::nullopt,
                                          .dry_run = state.value("clippingDryRun", false)});
            if (stopping || mode_changed) loop_->reset();
            if (mode_changed || resuming) clips_.clear_queue();
            paused_ = !enabled;
        }

        if (stopping) log("Monitoring turned OFF from dashboard, camera released");
        if (resuming) log("Monitoring turned ON from dashboard, starting capture");
        if (mode_changed) {
            clips_.reset_budget_block();
            if (clipping) {
                log(std::format(
                    "Clipping Mode ON{}, motion >= {:.1f}, {}s clips, {:.0f}s cooldown. Alerts "
                    "and predictions are paused.",
                    loop_->dry_run() ? " (dry run, nothing will be saved)" : "",
                    loop_->params().clip_threshold, config_.clip_segment_secs,
                    loop_->params().clip_cooldown_sec));
            } else {
                const auto reason = state.value("clippingStoppedReason", json()).is_string()
                                        ? ": " + state["clippingStoppedReason"].get<std::string>()
                                        : std::string();
                log("Clipping Mode OFF" + reason);
            }
        }
        if (enabled && (mode_changed || resuming)) start_capture();
    }

    void on_frame(std::span<const std::uint8_t> frame,
                  FfmpegCapture::Clock::time_point demuxed_at) {
        std::lock_guard lock(decision_mu_);
        const Decision d = loop_->handle_frame(frame, now_seconds());
        const double ms =
            std::chrono::duration<double, std::milli>(FfmpegCapture::Clock::now() - demuxed_at)
                .count();
        const Motion& m = d.motion;
        const std::string head =
            std::format("frame  t={:.1f}ms  motion={:.1f}  cells={}{}", ms, m.level, m.hot_cells,
                        m.lighting ? "  (scene-wide change, scored 0)" : "");
        const bool dry_run = loop_->dry_run();

        if (loop_->clipping()) {
            const bool clip = d.action == "clip";
            log(std::format("{}  →  clipping (threshold {:.1f}){}", head, d.threshold,
                            clip ? (dry_run ? "  ✂ WOULD CLIP" : "  ✂ CLIP") : ""));
            // Dashboard tuning telemetry. A hiccup must never interrupt capture.
            try {
                client_.post_json("/api/motion", {{"level", m.level},
                                                  {"hotCells", m.hot_cells},
                                                  {"lighting", m.lighting},
                                                  {"threshold", d.threshold},
                                                  {"wouldCapture", clip},
                                                  {"captured", clip && !dry_run}});
            } catch (const std::exception&) {
            }
        } else {
            const Prediction& p = d.prediction;
            const std::string warmup = d.warm ? ""
                                              : std::format("  (warmup {}/{})", p.window_fill,
                                                            loop_->params().embed_window);
            const std::string tag = d.behavior_alert ? "  ⚠ ALERT " + d.alert->label
                                    : d.motion_event ? "  ● MOTION"
                                                     : "";
            log(std::format("{}  →  {} {}%  interest={:.0f}/{:.0f}{}{}", head,
                            p.label.value_or("None"), p.confidence, p.interest,
                            loop_->params().interest_thresh, warmup, tag));
        }

        if (d.publish_prediction) {
            try {
                client_.post_json("/api/predictions", {{"label", d.prediction.label.value_or("")},
                                                       {"confidence", d.prediction.confidence}});
            } catch (const std::exception&) {
            }
        }

        try {
            if (d.action == "alert") {
                clips_.upload_and_alert(*d.alert, now_seconds());
            } else if (d.action == "motion_capture") {
                clips_.upload_motion_clip(m.level, now_seconds());
            } else if (d.action == "clip" && !dry_run) {
                clips_.queue_clip(m.level);
            }
        } catch (const std::exception& err) {
            const char* what = d.action == "alert"            ? "  alert error: "
                               : d.action == "motion_capture" ? "  capture error: "
                                                              : "  clip error: ";
            log(std::string(what) + err.what());
        }
    }

    Config config_;
    ServerClient client_;
    // One client per caller that may be slow, since a client serializes its
    // requests: JPEGs and frame telemetry never queue behind a clip upload.
    ServerClient live_client_;
    ServerClient clips_client_;
    Clips clips_;
    std::unique_ptr<OnnxPredictor> predictor_;
    std::unique_ptr<DecisionLoop> loop_;
    std::mutex decision_mu_;  // capture.py's _decision_lock
    std::mutex capture_mu_;
    std::unique_ptr<FfmpegCapture> capture_;
    bool shutting_down_ = false;  // guarded by capture_mu_
    bool paused_ = false;         // monitor thread only
};

}  // namespace
}  // namespace bunnycam

int main() {
    using namespace bunnycam;
    // Block the shutdown signals before any thread exists, so every thread
    // inherits the mask and only sigwait below receives them.
    sigset_t signals;
    sigemptyset(&signals);
    sigaddset(&signals, SIGTERM);
    sigaddset(&signals, SIGINT);
    pthread_sigmask(SIG_BLOCK, &signals, nullptr);

    std::atomic<int> exit_code = 0;
    std::unique_ptr<Agent> agent;
    try {
        agent = std::make_unique<Agent>(Config::from_env());
        agent->boot();
    } catch (const std::exception& err) {
        log(std::string("✗ Agent failed to start: ") + err.what());
        if (agent) agent->stop_capture();
        return 1;
    }

    // A crash in a loop ends the process so systemd restarts it, as an
    // uncaught exception in capture.py's run() did.
    auto guarded = [&](auto body) {
        return [&, body](std::stop_token stop) {
            try {
                (agent.get()->*body)(stop);
            } catch (const std::exception& err) {
                log(std::string("✗ Agent loop failed: ") + err.what());
                exit_code = 1;
                kill(getpid(), SIGTERM);
            }
        };
    };
    std::jthread monitor(guarded(&Agent::poll_monitor_loop));
    std::jthread prune(guarded(&Agent::prune_segments_loop));
    std::jthread drain(guarded(&Agent::drain_pending_clips_loop));

    int sig = 0;
    sigwait(&signals, &sig);
    if (exit_code == 0) log("Shutdown signal received — stopping ffmpeg child before exit");
    agent->begin_shutdown();
    monitor.request_stop();
    prune.request_stop();
    drain.request_stop();
    monitor.join();
    prune.join();
    drain.join();
    return exit_code;
}
