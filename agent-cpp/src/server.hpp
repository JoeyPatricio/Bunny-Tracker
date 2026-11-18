// HTTP to the FastAPI server (client.py and the HTTP half of capture.py's
// Agent), plus the clip harvesting and alert uploads built on it.
#pragma once

#include <filesystem>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "decision.hpp"

namespace bunnycam {

struct HttpResponse {
    int status = 0;
    std::string body;
    nlohmann::json json() const;  // throws on a non-JSON body
};

// Sends X-Agent-Token on every request, 10s timeouts. Transport failures
// throw std::runtime_error, as httpx raises. Thread-safe: requests on one
// instance are serialized, so the live-view relay gets its own instance and
// never waits behind a clip upload.
class ServerClient {
public:
    ServerClient(std::string base_url, std::string token);  // throws if token is empty
    ~ServerClient();

    void verify();  // GET /api/auth/check; throws unless {"authed": true}
    HttpResponse get(const std::string& path);
    HttpResponse post_json(const std::string& path, const nlohmann::json& body);
    // POST /api/stream/frame. Drops frames over 2MB without sending.
    void post_live_frame(std::string_view jpeg);
    // POST /api/recordings, multipart: video=("clip.webm", bytes, video/webm)
    // and a "source" form field when given.
    HttpResponse post_recording(std::string_view webm, const std::optional<std::string>& source);
    const std::string& base_url() const { return base_url_; }

private:
    struct Impl;
    std::string base_url_;
    std::unique_ptr<Impl> impl_;
    std::mutex mu_;
};

struct SegmentInfo {
    std::string name;
    double mtime = 0.0;  // seconds since epoch
};

// *.webm in dir, oldest first. Missing dir or a file vanishing mid-listing
// is not an error.
std::vector<SegmentInfo> segments_by_mtime(const std::filesystem::path& dir);

struct ClipOptions {
    std::filesystem::path seg_dir;
    double segment_max_age_sec = 30;  // segment_secs * 2.5
    double clip_segment_secs = 8;
    std::optional<std::string> webhook_url;
    std::string webhook_secret;
};

// Agent's segment queue and uploads (_upload_segment, _upload_and_alert,
// _notify_webhook, _upload_motion_clip, _queue_clip,
// _drain_pending_clips_once, _prune_segments_loop). Thread-safe: the frame
// thread queues and alerts while the drain thread uploads. `now` is wall
// clock seconds, injected for tests.
class Clips {
public:
    Clips(ClipOptions options, ServerClient& client);

    void set_timing(double segment_max_age_sec, double clip_segment_secs);
    void upload_and_alert(const Alert& alert, double now);
    void upload_motion_clip(double motion_level, double now);
    void queue_clip(double motion_level);
    void drain_once(double now);
    void prune(double now, double cutoff_age_sec = 600);
    void clear_queue();         // mode change or resume: pending + clipped sets
    void reset_budget_block();  // mode change
    bool has_pending() const;

private:
    std::optional<std::string> upload_segment(std::optional<std::string> name,
                                              const std::optional<std::string>& source, double now);

    struct Pending {
        std::string name;
        double mtime;
        double motion;
    };

    ClipOptions options_;
    ServerClient& client_;
    mutable std::mutex mu_;
    std::vector<Pending> pending_;
    std::vector<std::pair<std::string, double>> clipped_;  // (name, mtime rounded to 3dp)
    bool budget_blocked_ = false;
};

}  // namespace bunnycam
