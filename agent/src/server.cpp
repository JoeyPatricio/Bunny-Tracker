#include "server.hpp"

#include <httplib.h>
#include <sys/stat.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "log.hpp"

namespace bunnycam {

namespace {

bool truthy(const nlohmann::json& v) {
    if (v.is_null()) return false;
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number()) return v != 0;
    if (v.is_string()) return !v.get_ref<const std::string&>().empty();
    return !v.empty();
}

std::string display(const nlohmann::json& v) {
    return v.is_null() ? "None" : v.is_string() ? v.get<std::string>() : v.dump();
}

std::string motion_text(double v) {
    std::ostringstream s;
    s << std::fixed << std::setprecision(1) << v;
    return s.str();
}

void timeouts(httplib::Client& c) {
    c.set_connection_timeout(10);
    c.set_read_timeout(10);
    c.set_write_timeout(10);
}

HttpResponse response(httplib::Result r) {
    if (!r) throw std::runtime_error(httplib::to_string(r.error()));
    return {r->status, r->body};
}

double rounded(double t) { return std::nearbyint(t * 1000) / 1000; }
}  // namespace

nlohmann::json HttpResponse::json() const { return nlohmann::json::parse(body); }
struct ServerClient::Impl {
    httplib::Client client;
    explicit Impl(const std::string& url) : client(url) {}
};
ServerClient::ServerClient(std::string url, std::string token) {
    if (token.empty())
        throw std::runtime_error(
            "AGENT_TOKEN not set in data/.env — the agent cannot authenticate");
    impl_ = std::make_unique<Impl>(url);
    impl_->client.set_default_headers({{"X-Agent-Token", token}});
    timeouts(impl_->client);
}

ServerClient::~ServerClient() = default;
void ServerClient::verify() {
    auto r = get("/api/auth/check");
    if (r.status != 200 || !truthy(r.json().value("authed", nlohmann::json())))
        throw std::runtime_error("AGENT_TOKEN rejected by server — check data/.env on both sides");
}

HttpResponse ServerClient::get(const std::string& p) {
    std::lock_guard lock(mu_);
    return response(impl_->client.Get(p));
}

HttpResponse ServerClient::post_json(const std::string& p, const nlohmann::json& b) {
    std::lock_guard lock(mu_);
    return response(impl_->client.Post(p, b.dump(), "application/json"));
}

void ServerClient::post_live_frame(std::string_view b) {
    if (b.size() > 2 * 1024 * 1024) return;
    std::lock_guard lock(mu_);
    response(impl_->client.Post("/api/stream/frame", b.data(), b.size(), "image/jpeg"));
}

HttpResponse ServerClient::post_recording(std::string_view b,
                                          const std::optional<std::string>& source) {
    httplib::UploadFormDataItems items = {{"video", std::string(b), "clip.webm", "video/webm"}};
    if (source) items.push_back({"source", *source, "", ""});
    std::lock_guard lock(mu_);
    return response(impl_->client.Post("/api/recordings", items));
}

std::vector<SegmentInfo> segments_by_mtime(const std::filesystem::path& dir) {
    std::vector<SegmentInfo> segs;
    std::error_code ec;
    std::filesystem::directory_iterator it(dir, ec);
    if (ec == std::errc::no_such_file_or_directory) return segs;
    if (ec) throw std::filesystem::filesystem_error("list segments", dir, ec);
    for (const auto& e : it) {
        if (!e.path().filename().string().ends_with(".webm")) continue;
        struct stat st{};
        if (::stat(e.path().c_str(), &st)) {
            if (errno == ENOENT) continue;
            throw std::system_error(errno, std::generic_category(), e.path().string());
        }
        segs.push_back({e.path().filename().string(),
                        static_cast<double>(st.st_mtim.tv_sec) + st.st_mtim.tv_nsec / 1e9});
    }
    std::stable_sort(segs.begin(), segs.end(),
                     [](const auto& a, const auto& b) { return a.mtime < b.mtime; });
    return segs;
}

Clips::Clips(ClipOptions o, ServerClient& c) : options_(std::move(o)), client_(c) {}
void Clips::set_timing(double age, double secs) {
    std::lock_guard lock(mu_);
    options_.segment_max_age_sec = age;
    options_.clip_segment_secs = secs;
}

std::optional<std::string> Clips::upload_segment(std::optional<std::string> name,
                                                 const std::optional<std::string>& source,
                                                 double now) {
    if (!name) {
        auto segs = segments_by_mtime(options_.seg_dir);
        if (segs.size() >= 2) {
            const auto& s = segs[segs.size() - 2];
            double age;
            {
                std::lock_guard lock(mu_);
                age = options_.segment_max_age_sec;
            }
            if (now - s.mtime > age)
                log("  (latest finished segment " + s.name + " is stale — skipping upload)");
            else
                name = s.name;
        }
    }
    if (!name) {
        log("  (no finished segment yet — skipping upload)");
        return {};
    }
    std::ifstream in(options_.seg_dir / *name, std::ios::binary);
    if (!in) {
        log("  segment " + *name + " unreadable (" + std::strerror(errno) + ") — skipping upload");
        return {};
    }
    std::ostringstream data;
    data << in.rdbuf();
    if (in.bad()) {
        log("  segment " + *name + " unreadable (read error) — skipping upload");
        return {};
    }
    auto r = client_.post_recording(data.str(), source);
    if (r.status == 507) {
        std::lock_guard lock(mu_);
        if (!budget_blocked_) {
            budget_blocked_ = true;
            nlohmann::json reason;
            try {
                reason = r.json().value("reason", nlohmann::json());
            } catch (const nlohmann::json::exception&) {
            }
            log(std::string("  ") +
                (source == "clipping" ? "Clipping stopped" : "Motion capture refused") +
                " by the server: " + (truthy(reason) ? display(reason) : "budget reached"));
        }
        return {};
    }
    if (r.status != 200) {
        log("  upload failed: " + std::to_string(r.status));
        return {};
    }
    {
        std::lock_guard lock(mu_);
        budget_blocked_ = false;
    }
    return r.json().at("filename").get<std::string>();
}

void Clips::upload_and_alert(const Alert& a, double now) {
    auto f = upload_segment({}, {}, now);
    if (!f) return;
    log("  ↑ Uploaded clip " + *f + " → suggested " + a.label + " (review in Label Studio)");
    try {
        client_.post_json("/api/labels/" + *f + "/suggest", {{"label", a.label}});
    } catch (const std::exception&) {
    }  // Suggestions are best effort in the Python agent.
    auto result =
        client_
            .post_json("/api/notify",
                       {{"label", a.label}, {"confidence", a.confidence}, {"filename", *f}})
            .json();
    auto reason = result.value("reason", nlohmann::json());
    if (!truthy(reason)) reason = result.value("error", nlohmann::json());
    log("  ✉ Email: " +
        (truthy(result.value("sent", nlohmann::json())) ? "sent" : display(reason)));
}

void Clips::upload_motion_clip(double m, double now) {
    auto f = upload_segment({}, "motion", now);
    if (f)
        log("  ↑ Motion clip " + *f + " saved unlabeled (motion=" + motion_text(m) +
            ") — review in Label Studio");
}

void Clips::queue_clip(double m) {
    auto segs = segments_by_mtime(options_.seg_dir);
    if (segs.empty()) {
        log("  (no segment being written yet — skipping clip)");
        return;
    }
    const auto& s = segs.back();
    std::lock_guard lock(mu_);
    if (std::any_of(pending_.begin(), pending_.end(),
                    [&](const auto& p) { return p.name == s.name; }))
        return;
    auto key = std::pair{s.name, rounded(s.mtime)};
    if (std::find(clipped_.begin(), clipped_.end(), key) != clipped_.end()) return;
    pending_.push_back({s.name, s.mtime, m, next_pending_id_++});
    log("  ⧗ Clip queued (motion=" + motion_text(m) + ") — waiting for " + s.name + " to close");
}

void Clips::drain_once(double now) {
    std::lock_guard drain_lock(drain_mu_);
    std::vector<Pending> pending;
    double settle;
    {
        std::lock_guard lock(mu_);
        pending = pending_;
        settle = options_.clip_segment_secs * 1.5;
    }
    auto segs = segments_by_mtime(options_.seg_dir);
    for (const auto& p : pending) {
        auto s =
            std::find_if(segs.begin(), segs.end(), [&](const auto& s) { return s.name == p.name; });
        bool vanished = s == segs.end();
        if (!vanished && !(segs.back().mtime > s->mtime || now - s->mtime > settle)) continue;
        {
            std::lock_guard lock(mu_);
            auto entry = std::find_if(pending_.begin(), pending_.end(),
                                      [&](const auto& e) { return e.id == p.id; });
            if (entry == pending_.end()) continue;
            // Consume before network I/O. Failed uploads stay consumed, while
            // clips queued during this pass retain their own identities.
            pending_.erase(entry);
            if (!vanished) clipped_.emplace_back(p.name, rounded(s->mtime));
        }
        if (vanished) {
            log("  (queued segment " + p.name + " vanished — dropping clip)");
            continue;
        }
        try {
            auto f = upload_segment(p.name, "clipping", now);
            if (f)
                log("  ↑ Clip " + *f + " saved unlabeled (motion=" + motion_text(p.motion) +
                    ") — review in Label Studio");
        } catch (const std::exception& e) {
            log("  clip drain error: " + std::string(e.what()));
        }
    }
    std::lock_guard lock(mu_);
    if (clipped_.size() > 500) clipped_.clear();
}

void Clips::prune(double now, double age) {
    try {
        for (const auto& s : segments_by_mtime(options_.seg_dir))
            if (now - s.mtime > age) std::filesystem::remove(options_.seg_dir / s.name);
    } catch (const std::system_error&) {
        // Pruning is best effort, as in the Python background loop.
    }
}

void Clips::clear_queue() {
    std::lock_guard lock(mu_);
    pending_.clear();
    clipped_.clear();
}

void Clips::reset_budget_block() {
    std::lock_guard lock(mu_);
    budget_blocked_ = false;
}

bool Clips::has_pending() const {
    std::lock_guard lock(mu_);
    return !pending_.empty();
}
}  // namespace bunnycam
