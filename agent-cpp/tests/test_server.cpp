#include <fcntl.h>
#include <httplib.h>
#include <sys/stat.h>

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <condition_variable>
#include <future>
#include <iostream>
#include <thread>

#include "server.hpp"
#include "support.hpp"

using namespace bunnycam;
using nlohmann::json;
namespace {
struct Stub {
    httplib::Server server;
    std::thread thread;
    std::string url;
    void start() {
        int port = server.bind_to_any_port("127.0.0.1");
        REQUIRE(port > 0);
        url = "http://127.0.0.1:" + std::to_string(port);
        thread = std::thread([this] { server.listen_after_bind(); });
        server.wait_until_ready();
    }
    ~Stub() {
        server.stop();
        if (thread.joinable()) thread.join();
    }
};
struct Fixture {
    Stub stub;
    std::filesystem::path dir;
    std::vector<std::string> uploads;
    std::vector<std::string> sources;
    std::vector<std::string> paths;
    std::atomic<int> status{200};
    std::atomic<bool> allow{true};
    json webhook;
    std::string secret;
    std::mutex mu;
    std::mutex handler_mu;
    std::function<httplib::Server::HandlerResponse(const httplib::Request&, httplib::Response&)>
        before;
    void before_request(decltype(before) handler) {
        std::lock_guard lock(handler_mu);
        before = std::move(handler);
    }
    auto uploaded() {
        std::lock_guard lock(mu);
        return uploads;
    }
    auto sent_sources() {
        std::lock_guard lock(mu);
        return sources;
    }
    auto sent_paths() {
        std::lock_guard lock(mu);
        return paths;
    }
    auto hook_body() {
        std::lock_guard lock(mu);
        return webhook;
    }
    auto hook_secret() {
        std::lock_guard lock(mu);
        return secret;
    }
    explicit Fixture(bool serve = true) {
        char pattern[] = "/tmp/bunny-server-XXXXXX";
        auto path = mkdtemp(pattern);
        if (!path) throw std::runtime_error("mkdtemp failed");
        dir = path;
        stub.server.Post("/api/recordings", [this](const auto& req, auto& res) {
            std::lock_guard lock(mu);
            auto file = req.form.get_file("video");
            CHECK(file.filename == "clip.webm");
            CHECK(file.content_type == "video/webm");
            CHECK(req.get_header_value("X-Agent-Token") == "token");
            uploads.push_back(file.content);
            sources.push_back(req.form.has_field("source") ? req.form.get_field("source") : "");
            res.status = status.load();
            res.set_content(json{{"filename", "saved.webm"}, {"reason", "disk floor"}}.dump(),
                            "application/json");
        });
        stub.server.Post(R"(/api/labels/.*|/api/notify|/api/notify/gate)", [this](const auto& req,
                                                                                  auto& res) {
            std::lock_guard lock(mu);
            paths.push_back(req.path);
            if (req.path == "/api/notify")
                CHECK(json::parse(req.body) ==
                      json{{"label", "rearing"}, {"confidence", 95}, {"filename", "saved.webm"}});
            else
                CHECK(json::parse(req.body) == json{{"label", "rearing"}});
            res.set_content(json{{"allow", allow.load()}, {"sent", true}}.dump(),
                            "application/json");
        });
        stub.server.Post("/hook", [this](const auto& req, auto& res) {
            std::lock_guard lock(mu);
            webhook = json::parse(req.body);
            secret = req.get_header_value("X-BunnyCam-Secret");
            CHECK_FALSE(req.has_header("X-Agent-Token"));
            res.status = 503;  // HTTP failures still count as a delivered webhook.
        });
        stub.server.set_pre_routing_handler([this](const auto& req, auto& res) {
            decltype(before) handler;
            {
                std::lock_guard lock(handler_mu);
                handler = before;
            }
            return handler ? handler(req, res) : httplib::Server::HandlerResponse::Unhandled;
        });
        if (serve) stub.start();
    }
    ~Fixture() {
        stub.server.stop();
        if (stub.thread.joinable()) stub.thread.join();
        std::filesystem::remove_all(dir);
    }
    void file(const std::string& name, double mtime) {
        std::ofstream(dir / name) << name;
        auto sec = static_cast<time_t>(mtime);
        timespec ts[2] = {{sec, static_cast<long>((mtime - sec) * 1e9)},
                          {sec, static_cast<long>((mtime - sec) * 1e9)}};
        REQUIRE(utimensat(AT_FDCWD, (dir / name).c_str(), ts, 0) == 0);
    }
    ClipOptions options() const {
        ClipOptions o;
        o.seg_dir = dir;
        return o;
    }
};
struct Logs {
    std::ostringstream out;
    std::streambuf* old = std::cout.rdbuf(out.rdbuf());
    ~Logs() { std::cout.rdbuf(old); }
    std::size_t count(const std::string& s) const {
        auto text = out.str();
        std::size_t n = 0, p = 0;
        while ((p = text.find(s, p)) != std::string::npos) {
            ++n;
            p += s.size();
        }
        return n;
    }
};
}  // namespace

TEST_CASE("server verifies authentication and returns HTTP failures", "[server]") {
    Stub s;
    std::atomic<int> status{200};
    std::mutex auth_mu;
    json auth = {{"authed", true}};
    s.server.Get("/api/auth/check", [&](const auto& req, auto& res) {
        CHECK(req.get_header_value("X-Agent-Token") == "token");
        std::lock_guard lock(auth_mu);
        res.status = status.load();
        res.set_content(auth.dump(), "application/json");
    });
    s.start();
    REQUIRE_THROWS_WITH(ServerClient(s.url, ""),
                        "AGENT_TOKEN not set in data/.env — the agent cannot authenticate");
    ServerClient c(s.url, "token");
    REQUIRE_NOTHROW(c.verify());
    {
        std::lock_guard lock(auth_mu);
        auth = {{"authed", false}};
    }
    REQUIRE_THROWS_WITH(c.verify(),
                        "AGENT_TOKEN rejected by server — check data/.env on both sides");
    {
        std::lock_guard lock(auth_mu);
        auth = {{"authed", ""}};
    }
    REQUIRE_THROWS(c.verify());
    {
        std::lock_guard lock(auth_mu);
        auth = {{"authed", "yes"}};
    }
    REQUIRE_NOTHROW(c.verify());
    status = 401;
    REQUIRE(c.get("/api/auth/check").status == 401);
    REQUIRE_THROWS(c.verify());
    s.server.stop();
    s.thread.join();
    REQUIRE_THROWS_AS(c.get("/api/auth/check"), std::runtime_error);
}
TEST_CASE("server JSON endpoints and live frame cap", "[server]") {
    Stub s;
    std::atomic<int> frames{0};
    s.server.Post("/api/stream/frame", [&](const auto& req, auto& res) {
        ++frames;
        CHECK(req.get_header_value("Content-Type") == "image/jpeg");
        CHECK(req.get_header_value("X-Agent-Token") == "token");
        res.status = 500;
    });
    s.server.Post(R"(/api/motion|/api/predictions)", [](const auto& req, auto& res) {
        CHECK(req.get_header_value("X-Agent-Token") == "token");
        CHECK(json::parse(req.body) == json{{"level", 5}});
        res.set_content("{}", "application/json");
    });
    s.start();
    ServerClient c(s.url, "token");
    c.post_live_frame(std::string(2 * 1024 * 1024, 'a'));
    c.post_live_frame(std::string(2 * 1024 * 1024 + 1, 'a'));
    CHECK(frames == 1);
    CHECK(c.post_json("/api/motion", {{"level", 5}}).status == 200);
    CHECK(c.post_json("/api/predictions", {{"level", 5}}).status == 200);
}
TEST_CASE("clip targets in-progress segment and deduplicates moving mtimes", "[server]") {
    Fixture f;
    ServerClient client(f.stub.url, "token");
    Clips c(f.options(), client);
    f.file("old.webm", 980);
    f.file("active.webm", 994);
    c.queue_clip(4.2);
    c.queue_clip(5.1);
    c.drain_once(1000);
    CHECK(c.has_pending());
    CHECK(f.uploaded().empty());
    f.file("active.webm", 998);
    c.queue_clip(6);
    c.drain_once(1000);
    CHECK(f.uploaded().empty());
    f.file("next.webm", 1000);
    c.drain_once(1000);
    CHECK(f.uploaded() == std::vector<std::string>{"active.webm"});
    CHECK(f.sent_sources() == std::vector<std::string>{"clipping"});
    CHECK_FALSE(c.has_pending());
    std::filesystem::remove(f.dir / "next.webm");
    c.queue_clip(7);
    CHECK_FALSE(c.has_pending());
}
TEST_CASE("stalled segment settles once and restart name reuse queues again", "[server]") {
    Fixture f;
    ServerClient client(f.stub.url, "token");
    Clips c(f.options(), client);
    f.file("same.webm", 970);
    c.queue_clip(5);
    f.file("same.webm", 975);
    c.drain_once(1000);
    c.queue_clip(5);
    c.drain_once(1000);
    CHECK(f.uploaded().size() == 1);
    CHECK_FALSE(c.has_pending());
    f.file("same.webm", 999);
    c.queue_clip(5);
    CHECK(c.has_pending());
    c.drain_once(1000);
    CHECK(f.uploaded().size() == 1);
    c.drain_once(1012);
    CHECK(f.uploaded().size() == 2);
}
TEST_CASE("active segment never uploads early and vanished segment drops", "[server]") {
    Fixture f;
    ServerClient client(f.stub.url, "token");
    Clips c(f.options(), client);
    f.file("a.webm", 1000);
    c.queue_clip(5);
    for (int i = 0; i < 3; ++i) {
        f.file("a.webm", 1100 + i);
        c.drain_once(1100 + i);
    }
    CHECK(f.uploaded().empty());
    CHECK(c.has_pending());
    std::filesystem::remove(f.dir / "a.webm");
    c.drain_once(1200);
    CHECK_FALSE(c.has_pending());
    CHECK(f.uploaded().empty());
}
TEST_CASE("latest finished segment requires two files and skips stale footage", "[server]") {
    Fixture f;
    ServerClient client(f.stub.url, "token");
    Clips c(f.options(), client);
    c.upload_motion_clip(5, 1000);
    f.file("a.webm", 960);
    c.upload_motion_clip(5, 1000);
    f.file("b.webm", 999);
    c.upload_motion_clip(5, 1000);
    CHECK(f.uploaded().empty());
    f.file("a.webm", 980);
    c.upload_motion_clip(5, 1000);
    CHECK(f.uploaded() == std::vector<std::string>{"a.webm"});
}
TEST_CASE("motion source alert endpoints and budget episodes", "[server]") {
    Fixture f;
    ServerClient client(f.stub.url, "token");
    Clips c(f.options(), client);
    f.file("a.webm", 990);
    f.file("b.webm", 999);
    c.upload_motion_clip(5, 1000);
    c.upload_and_alert({"rearing", 95}, 1000);
    CHECK(f.sent_sources() == std::vector<std::string>{"motion", ""});
    CHECK(f.sent_paths() ==
          std::vector<std::string>{"/api/labels/saved.webm/suggest", "/api/notify"});
    Logs logs;
    f.status = 507;
    c.upload_motion_clip(5, 1000);
    c.upload_motion_clip(5, 1000);
    CHECK(logs.count("Motion capture refused by the server: disk floor") == 1);
    f.status = 200;
    c.upload_motion_clip(5, 1000);
    f.status = 507;
    c.upload_motion_clip(5, 1000);
    CHECK(logs.count("Motion capture refused by the server: disk floor") == 2);
    c.reset_budget_block();
    c.upload_motion_clip(5, 1000);
    CHECK(logs.count("Motion capture refused") == 3);
}
TEST_CASE("webhook gate secret timestamp and normalized recording URL", "[server]") {
    Fixture f;
    ServerClient client(f.stub.url + "/", "token");
    auto options = f.options();
    options.webhook_url = f.stub.url + "/hook";
    options.webhook_secret = "secret";
    Clips c(options, client);
    f.file("a.webm", 990);
    f.file("b.webm", 999);
    f.allow = false;
    c.upload_and_alert({"rearing", 95}, 1000);
    CHECK(f.hook_body().is_null());
    f.allow = true;
    c.upload_and_alert({"rearing", 95}, 1000);
    CHECK(f.sent_paths() ==
          std::vector<std::string>{"/api/labels/saved.webm/suggest", "/api/notify/gate",
                                   "/api/labels/saved.webm/suggest", "/api/notify/gate"});
    CHECK(f.hook_secret() == "secret");
    CHECK(f.hook_body() == json{{"label", "rearing"},
                                {"confidence", 95},
                                {"filename", "saved.webm"},
                                {"clipUrl", f.stub.url + "/recordings/saved.webm"},
                                {"at", "1970-01-01T00:16:40Z"}});
}
TEST_CASE("prune only removes webm older than cutoff", "[server]") {
    Fixture f(false);
    ServerClient client("http://127.0.0.1:1", "token");
    Clips c(f.options(), client);
    f.file("old.webm", 399.125);
    f.file("edge.webm", 400);
    f.file("new.webm", 401);
    f.file("keep.txt", 100);
    CHECK(segments_by_mtime(f.dir).front().mtime == 399.125);
    c.prune(1000);
    CHECK_FALSE(std::filesystem::exists(f.dir / "old.webm"));
    CHECK(std::filesystem::exists(f.dir / "edge.webm"));
    CHECK(std::filesystem::exists(f.dir / "new.webm"));
    CHECK(std::filesystem::exists(f.dir / "keep.txt"));
    CHECK(segments_by_mtime(f.dir / "missing").empty());
}
TEST_CASE("drain consumes attempted clips after a transport failure and continues", "[server]") {
    Fixture f;
    ServerClient client(f.stub.url, "token");
    Clips c(f.options(), client);
    for (int i = 0; i < 3; ++i) {
        f.file(std::to_string(i) + ".webm", 970 + i);
        c.queue_clip(5);
    }
    std::atomic<int> attempts{0};
    f.before_request([&](const auto& req, auto& res) {
        if (req.path == "/api/recordings" && ++attempts == 2) {
            res.set_content_provider(
                20, "application/json",
                [](std::size_t, std::size_t, httplib::DataSink&) { return false; });
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    // The second response closes before its promised body reaches the client.
    Logs logs;
    c.drain_once(1000);
    c.drain_once(1000);
    CHECK(attempts == 3);
    CHECK(f.uploaded().size() == 2);
    CHECK_FALSE(c.has_pending());
    CHECK(logs.count("clip drain error:") == 1);
    f.file("failed.webm", 1001);
    c.queue_clip(5);
    f.stub.server.stop();
    f.stub.thread.join();
    c.drain_once(1020);
    CHECK_FALSE(c.has_pending());
    c.queue_clip(5);
    CHECK_FALSE(c.has_pending());
    CHECK(logs.count("clip drain error:") == 2);
    REQUIRE_THROWS(c.upload_motion_clip(5, 1000));
    REQUIRE_THROWS(c.upload_and_alert({"rearing", 95}, 1000));
}
TEST_CASE("clip queued during an upload survives the drain snapshot", "[server]") {
    Fixture f;
    ServerClient client(f.stub.url, "token");
    Clips c(f.options(), client);
    f.file("a.webm", 970);
    c.queue_clip(5);
    std::promise<void> entered, resume;
    auto resumed = resume.get_future();
    f.before_request([&](const auto& req, auto&) {
        if (req.path == "/api/recordings") {
            entered.set_value();
            resumed.wait();
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    auto drain = std::async(std::launch::async, [&] { c.drain_once(1000); });
    entered.get_future().wait();
    f.file("b.webm", 999);
    c.queue_clip(6);
    resume.set_value();
    drain.get();
    CHECK(c.has_pending());
    CHECK(f.uploaded() == std::vector<std::string>{"a.webm"});
    f.before_request({});
    c.drain_once(1020);
    CHECK_FALSE(c.has_pending());
    CHECK(f.uploaded() == std::vector<std::string>{"a.webm", "b.webm"});
}
