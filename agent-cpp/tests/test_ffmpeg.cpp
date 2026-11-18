#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <format>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "ffmpeg.hpp"
#include "motion.hpp"
#include "support.hpp"

namespace {

using namespace std::chrono_literals;
using bunnycam::CaptureConfig;
using bunnycam::CaptureTiming;
using bunnycam::FfmpegCapture;
using Clock = FfmpegCapture::Clock;
namespace fs = std::filesystem;

struct TempDir {
    fs::path path;
    TempDir() {
        static int count = 0;
        path = fs::temp_directory_path() / std::format("bunnycam-ffmpeg-{}-{}", getpid(), count++);
        fs::create_directories(path);
    }
    ~TempDir() { fs::remove_all(path); }
};

// The fake ffmpeg scripts ignore their args (except segdir.sh), so only the
// path and seg_dir matter.
CaptureConfig fake_config(const char* script, const fs::path& seg_dir) {
    CaptureConfig config;
    config.ffmpeg_path =
        (bunnycam::test::kRepoDir / "agent-cpp/tests/fake_ffmpeg" / script).string();
    config.camera_format = "lavfi";
    config.camera_input = "unused";
    config.seg_dir = seg_dir;
    return config;
}

// What the handlers saw, readable from the test thread.
struct Seen {
    std::mutex mu;
    std::vector<int> frames;  // first byte of each handled frame
    std::vector<std::size_t> frame_sizes;
    std::vector<Clock::time_point> demuxed;
    std::vector<std::string> jpegs;

    void frame(const std::vector<std::uint8_t>& bytes, Clock::time_point demuxed_at) {
        std::lock_guard lock(mu);
        frames.push_back(bytes.at(0));
        frame_sizes.push_back(bytes.size());
        demuxed.push_back(demuxed_at);
    }
    void jpeg(std::string bytes) {
        std::lock_guard lock(mu);
        jpegs.push_back(std::move(bytes));
    }
    std::size_t frame_count() {
        std::lock_guard lock(mu);
        return frames.size();
    }
    std::size_t jpeg_count() {
        std::lock_guard lock(mu);
        return jpegs.size();
    }

    FfmpegCapture::FrameHandler on_frame() {
        return [this](std::vector<std::uint8_t> bytes, Clock::time_point at) { frame(bytes, at); };
    }
    FfmpegCapture::JpegHandler on_jpeg() {
        return [this](std::string bytes) { jpeg(std::move(bytes)); };
    }
};

template <class Pred>
bool eventually(Pred pred, std::chrono::milliseconds timeout = 5s) {
    const auto deadline = Clock::now() + timeout;
    while (!pred()) {
        if (Clock::now() > deadline) return false;
        std::this_thread::sleep_for(5ms);
    }
    return true;
}

// The test process has no children left: nothing running and nothing unreaped.
bool no_children() { return waitpid(-1, nullptr, WNOHANG) == -1 && errno == ECHILD; }

const CaptureTiming kFast{.restart_delay = 300ms, .stop_timeout = 2s};

}  // namespace

TEST_CASE("build_args matches the Python golden for every config") {
    const auto cases = bunnycam::test::read_json(bunnycam::test::kFixtures / "ffmpeg_args.json");
    REQUIRE(cases.size() == 4);
    for (const auto& c : cases) {
        INFO(c["name"].get<std::string>());
        CaptureConfig config;
        config.ffmpeg_path = "/usr/bin/ffmpeg";
        config.camera_format = c["camera_format"];
        config.camera_input = c["camera_input"];
        config.camera_size = c["camera_size"];
        config.camera_fps = c["camera_fps"];
        config.frame_interval = c["frame_interval"];
        config.segment_secs = c["segment_secs"];
        config.seg_dir = c["segment_dir"].get<std::string>();
        CHECK(bunnycam::build_args(config) == c["args"].get<std::vector<std::string>>());
    }
}

TEST_CASE("a slow handler does not stall the reader") {
    TempDir tmp;
    Seen seen;
    std::atomic<int> entered = 0;
    FfmpegCapture capture(
        fake_config("frames.sh", tmp.path),
        [&](std::vector<std::uint8_t> bytes, Clock::time_point at) {
            ++entered;
            std::this_thread::sleep_for(200ms);
            seen.frame(bytes, at);
        },
        seen.on_jpeg(), kFast);
    capture.start();

    // 20 frames at 200ms each would take 4s if the reader waited on the handler.
    REQUIRE(eventually([&] { return capture.frames_dropped() + entered == 20; }, 2s));
    REQUIRE(eventually([&] { return seen.frame_count() == static_cast<std::size_t>(entered); }));
    capture.stop_and_wait();

    CHECK(capture.frames_dropped() > 0);
    CHECK(seen.frames.size() < 20);
    CHECK(capture.frames_dropped() + seen.frames.size() == 20);
    CHECK(seen.frames.back() == 20);
    for (auto size : seen.frame_sizes) CHECK(size == bunnycam::kFrameBytes);
}

TEST_CASE("a handler exception does not kill the consumer") {
    TempDir tmp;
    Seen seen;
    FfmpegCapture capture(
        fake_config("two_frames.sh", tmp.path),
        [&](std::vector<std::uint8_t> bytes, Clock::time_point at) {
            if (bytes[0] == 1) throw std::runtime_error("boom");
            seen.frame(bytes, at);
        },
        seen.on_jpeg(), kFast);
    capture.start();
    REQUIRE(eventually([&] { return seen.frame_count() == 1; }));
    capture.stop_and_wait();
    CHECK(seen.frames == std::vector<int>{2});
}

TEST_CASE("JPEGs on fd 3 reach the handler intact") {
    TempDir tmp;
    Seen seen;
    FfmpegCapture capture(fake_config("jpegs.sh", tmp.path), seen.on_frame(), seen.on_jpeg(),
                          kFast);
    capture.start();
    REQUIRE(eventually([&] { return seen.jpeg_count() == 3; }));
    capture.stop_and_wait();
    // The JPEG with no EOI is discarded at the 2 MiB cap and the stream
    // resyncs on the next SOI.
    CHECK(seen.jpegs == std::vector<std::string>{"\xFF\xD8"
                                                 "AAAA\xFF\xD9",
                                                 "\xFF\xD8"
                                                 "BB\xFF\xD9",
                                                 "\xFF\xD8"
                                                 "CC\xFF\xD9"});
}

TEST_CASE("fd 3 reaches the child when the pipe's write end is fd 3 in the parent") {
    // With stdin closed and fd 3 free, the first pipe start() creates (the
    // fd-3 pipe) lands on fds 0 and 3, so the child needs dup2(3, 3).
    struct FreeFds {
        int saved_stdin = dup(0);
        int saved_fd3 = fcntl(3, F_DUPFD_CLOEXEC, 10);
        FreeFds() {
            close(0);
            if (saved_fd3 != -1) close(3);
        }
        ~FreeFds() {
            dup2(saved_stdin, 0);
            close(saved_stdin);
            if (saved_fd3 != -1) {
                dup2(saved_fd3, 3);
                close(saved_fd3);
            }
        }
    } free_fds;
    REQUIRE(fcntl(0, F_GETFD) == -1);
    REQUIRE(fcntl(3, F_GETFD) == -1);

    TempDir tmp;
    Seen seen;
    FfmpegCapture capture(fake_config("ready.sh", tmp.path), seen.on_frame(), seen.on_jpeg(),
                          kFast);
    capture.start();
    CHECK(eventually([&] { return seen.jpeg_count() == 1; }));
    capture.stop_and_wait();
}

TEST_CASE("the child starts with SIGTERM unblocked") {
    // The agent blocks SIGTERM in every thread for its sigwait thread.
    sigset_t term, old;
    sigemptyset(&term);
    sigaddset(&term, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &term, &old);

    TempDir tmp;
    Seen seen;
    FfmpegCapture capture(fake_config("ready.sh", tmp.path), seen.on_frame(), seen.on_jpeg(),
                          {.restart_delay = 300ms, .stop_timeout = 5s});
    capture.start();
    REQUIRE(eventually([&] { return seen.jpeg_count() == 1; }));
    const auto started = Clock::now();
    capture.stop_and_wait();
    CHECK(Clock::now() - started < 2s);
    pthread_sigmask(SIG_SETMASK, &old, nullptr);
}

TEST_CASE("a child that ignores SIGTERM is killed after stop_timeout and reaped") {
    TempDir tmp;
    Seen seen;
    FfmpegCapture capture(fake_config("ignore_term.sh", tmp.path), seen.on_frame(), seen.on_jpeg(),
                          {.restart_delay = 300ms, .stop_timeout = 300ms});
    capture.start();
    REQUIRE(eventually([&] { return seen.jpeg_count() == 1; }));
    const auto started = Clock::now();
    capture.stop_and_wait();
    const auto took = Clock::now() - started;
    CHECK(took >= 300ms);
    CHECK(took < 3s);
    CHECK(no_children());
}

TEST_CASE("capture restarts after an unexpected exit") {
    TempDir tmp;
    Seen seen;
    std::atomic<bool> busy = false;
    std::atomic<bool> overlapped = false;
    Clock::time_point first_done;
    FfmpegCapture capture(
        fake_config("exit_after_frame.sh", tmp.path),
        [&](std::vector<std::uint8_t> bytes, Clock::time_point at) {
            if (busy.exchange(true)) overlapped = true;
            // The first handler outlives its process (which exits 200ms after
            // its frame), so the restart has to wait for it.
            const bool first = seen.frame_count() == 0;
            if (first) std::this_thread::sleep_for(500ms);
            seen.frame(bytes, at);
            if (first) first_done = Clock::now();
            busy = false;
        },
        seen.on_jpeg(), kFast);
    capture.start();
    REQUIRE(eventually([&] { return seen.frame_count() >= 2; }));
    capture.stop_and_wait();
    CHECK_FALSE(overlapped);
    CHECK(seen.demuxed[1] - first_done >= kFast.restart_delay);
    CHECK(seen.frames[1] == 1);
}

TEST_CASE("a failed respawn is retried") {
    TempDir tmp;
    Seen seen;
    const fs::path seg_dir = tmp.path / "segments";
    FfmpegCapture capture(fake_config("exit_after_frame.sh", seg_dir), seen.on_frame(),
                          seen.on_jpeg(), {.restart_delay = 200ms, .stop_timeout = 2s});
    capture.start();
    REQUIRE(eventually([&] { return seen.frame_count() == 1; }));

    // A file where the segment directory goes makes every respawn throw.
    fs::remove_all(seg_dir);
    std::ofstream(seg_dir).put('x');
    std::this_thread::sleep_for(1s);
    CHECK(seen.frame_count() == 1);

    fs::remove(seg_dir);
    CHECK(eventually([&] { return seen.frame_count() >= 2; }, 3s));
    capture.stop_and_wait();
}

TEST_CASE("stop during start leaves no child") {
    TempDir tmp;
    for (int i = 0; i < 20; ++i) {
        Seen seen;
        FfmpegCapture capture(fake_config("ready.sh", tmp.path), seen.on_frame(), seen.on_jpeg(),
                              kFast);
        std::jthread starter([&] { capture.start(); });
        capture.stop_and_wait();
        starter.join();
        REQUIRE(no_children());
        capture.start();  // A stopped capture never spawns again.
        REQUIRE(no_children());
    }
}

TEST_CASE("stop_and_wait is safe twice and from two threads") {
    TempDir tmp;
    Seen seen;
    FfmpegCapture capture(fake_config("ready.sh", tmp.path), seen.on_frame(), seen.on_jpeg(),
                          kFast);
    capture.start();
    REQUIRE(eventually([&] { return seen.jpeg_count() == 1; }));
    {
        std::jthread a([&] { capture.stop_and_wait(); });
        std::jthread b([&] { capture.stop_and_wait(); });
    }
    capture.stop_and_wait();
    CHECK(no_children());
}

TEST_CASE("the destructor stops a running capture") {
    TempDir tmp;
    Seen seen;
    {
        FfmpegCapture capture(fake_config("ready.sh", tmp.path), seen.on_frame(), seen.on_jpeg(),
                              kFast);
        capture.start();
        REQUIRE(eventually([&] { return seen.jpeg_count() == 1; }));
    }
    CHECK(no_children());
}

TEST_CASE("start throws when ffmpeg cannot be spawned") {
    TempDir tmp;
    Seen seen;
    CaptureConfig config = fake_config("ready.sh", tmp.path);
    config.ffmpeg_path = (tmp.path / "no-such-ffmpeg").string();
    FfmpegCapture capture(config, seen.on_frame(), seen.on_jpeg(), kFast);
    CHECK_THROWS_AS(capture.start(), std::system_error);
    CHECK(no_children());
}

TEST_CASE("the segment directory exists before ffmpeg starts") {
    TempDir tmp;
    Seen seen;
    FfmpegCapture capture(fake_config("segdir.sh", tmp.path / "a" / "b" / "segments"),
                          seen.on_frame(), seen.on_jpeg(), kFast);
    capture.start();
    REQUIRE(eventually([&] { return seen.jpeg_count() == 1; }));
    capture.stop_and_wait();
    CHECK(seen.jpegs[0] == "\xFF\xD8ok\xFF\xD9");
}

TEST_CASE("real ffmpeg delivers raw frames and JPEGs") {
    if (!fs::exists("/usr/bin/ffmpeg")) SKIP("/usr/bin/ffmpeg not installed");
    TempDir tmp;
    Seen seen;
    CaptureConfig config;
    config.ffmpeg_path = "/usr/bin/ffmpeg";
    config.camera_format = "lavfi";
    config.camera_input = "testsrc2=size=640x360:rate=15";
    config.frame_interval = 0.5;
    config.segment_secs = 2;
    config.seg_dir = tmp.path;
    FfmpegCapture capture(config, seen.on_frame(), seen.on_jpeg());
    capture.start();
    CHECK(eventually([&] { return seen.frame_count() >= 2 && seen.jpeg_count() >= 5; }, 6s));
    capture.stop_and_wait();
    CHECK(no_children());
    for (auto size : seen.frame_sizes) CHECK(size == bunnycam::kFrameBytes);
    for (const auto& jpeg : seen.jpegs) {
        CHECK(jpeg.starts_with("\xFF\xD8"));
        CHECK(jpeg.ends_with("\xFF\xD9"));
    }
}
