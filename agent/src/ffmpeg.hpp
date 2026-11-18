// One ffmpeg process with three outputs (raw 224x224 frames on stdout,
// rolling WebM segments, MJPEG live view on fd 3), supervised and restarted
// after an unexpected exit.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace bunnycam {

struct CaptureConfig {
    std::string ffmpeg_path;
    std::string camera_format;  // v4l2 | lavfi
    std::string camera_input;
    std::string camera_size;
    std::string camera_fps;
    double frame_interval = 1.5;
    int segment_secs = 12;
    std::filesystem::path seg_dir;
};

// Golden-tested against FfmpegCapture._build_args (tests/fixtures/ffmpeg_args.json).
std::vector<std::string> build_args(const CaptureConfig& config);

struct CaptureTiming {
    std::chrono::milliseconds restart_delay{5000};
    std::chrono::milliseconds stop_timeout{5000};
};

class FfmpegCapture {
public:
    using Clock = std::chrono::steady_clock;
    // Runs on the frame consumer thread, one call at a time. Frames that
    // arrive while it runs are dropped except the newest. demuxed_at is when
    // the reader finished the frame.
    using FrameHandler =
        std::function<void(std::vector<std::uint8_t> frame, Clock::time_point demuxed_at)>;
    // Runs on the live-view thread, once per complete JPEG from fd 3.
    using JpegHandler = std::function<void(std::string jpeg)>;

    FfmpegCapture(CaptureConfig config, FrameHandler on_frame, JpegHandler on_jpeg,
                  CaptureTiming timing = {});
    ~FfmpegCapture();  // stop_and_wait()

    FfmpegCapture(const FfmpegCapture&) = delete;
    FfmpegCapture& operator=(const FfmpegCapture&) = delete;

    // Creates seg_dir, spawns ffmpeg, and starts the reader, consumer and
    // supervisor threads. Throws if the spawn fails.
    void start();

    // Pause: SIGTERM ffmpeg, SIGKILL after stop_timeout, reap, then join every
    // thread including a frame handler that is mid-call. No restart follows.
    // Safe to call more than once and while start() is running.
    void stop_and_wait();

    std::uint64_t frames_dropped() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace bunnycam
