#include "ffmpeg.hpp"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/eventfd.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <format>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#include "log.hpp"
#include "motion.hpp"

extern char** environ;

namespace bunnycam {
namespace {

// Recorded segments run at a constant 15fps against real timestamps.
constexpr int kSegmentFps = 15;

// Frames are dropped silently by design (see read_frames), which is exactly
// how a Pi falling behind the frame budget would hide. Log a line per N drops
// so a sustained backlog is visible in the log without spamming it.
constexpr std::uint64_t kDropLogEvery = 100;

// A live-view stream that loses an EOI would otherwise buffer forever.
constexpr std::size_t kMaxJpegBytes = 2 * 1024 * 1024;
constexpr std::string_view kSoi = "\xFF\xD8";
constexpr std::string_view kEoi = "\xFF\xD9";

// ffmpeg stderr with no newline would otherwise buffer forever.
constexpr std::size_t kMaxStderrLine = 4096;

std::system_error errno_error(const char* what) {
    return std::system_error(errno, std::generic_category(), what);
}

class Fd {
public:
    Fd() = default;
    explicit Fd(int fd) : fd_(fd) {}
    Fd(Fd&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
    Fd& operator=(Fd&& other) noexcept {
        reset(std::exchange(other.fd_, -1));
        return *this;
    }
    ~Fd() { reset(); }

    int get() const { return fd_; }
    void reset(int fd = -1) {
        if (fd_ >= 0) ::close(fd_);
        fd_ = fd;
    }

private:
    int fd_ = -1;
};

struct Pipe {
    Fd read;
    Fd write;
};

Pipe make_pipe() {
    int fds[2];
    if (pipe2(fds, O_CLOEXEC) == -1) throw errno_error("pipe2");
    Pipe pipe{Fd(fds[0]), Fd(fds[1])};
    // The child gets each write end through dup2 onto fd 1, 2 or 3. A write
    // end already sitting on 0-3 breaks that: dup2(3, 3) is a no-op, and
    // whether posix_spawn then clears FD_CLOEXEC depends on the glibc
    // version, and a write end on 1 or 2 would be clobbered by an earlier
    // dup2. Above 3, every dup2 is a real one, which always clears it.
    if (pipe.write.get() <= 3) {
        const int moved = fcntl(pipe.write.get(), F_DUPFD_CLOEXEC, 4);
        if (moved == -1) throw errno_error("fcntl");
        pipe.write.reset(moved);
    }
    return pipe;
}

std::string_view trim(std::string_view s) {
    constexpr std::string_view ws = " \t\r\n\v\f";
    const auto begin = s.find_first_not_of(ws);
    if (begin == std::string_view::npos) return {};
    return s.substr(begin, s.find_last_not_of(ws) - begin + 1);
}

// One spawned ffmpeg and the threads bound to its streams. The threads are
// declared last, so they are stopped and joined before the fds close.
struct Process {
    struct Frame {
        std::vector<std::uint8_t> bytes;
        FfmpegCapture::Clock::time_point demuxed_at;
    };

    pid_t pid = -1;
    Fd stdout_fd;
    Fd stderr_fd;
    Fd mjpeg_fd;
    // Never read, so once a stop writes to it every reader's poll sees it.
    Fd wake;

    // Single-slot handoff between the stdout reader and the frame consumer.
    std::mutex slot_mu;
    std::condition_variable_any slot_cv;
    std::optional<Frame> slot;

    std::jthread stdout_reader;
    std::jthread mjpeg_reader;
    std::jthread stderr_reader;
    std::jthread consumer;
};

// Fills buf from fd. Returns 0 at EOF and when stop is requested.
std::size_t read_some(const std::stop_token& stop, Process& proc, int fd, void* buf,
                      std::size_t len) {
    std::stop_callback wake_on_stop(stop, [&] { eventfd_write(proc.wake.get(), 1); });
    pollfd fds[2] = {{fd, POLLIN, 0}, {proc.wake.get(), POLLIN, 0}};
    for (;;) {
        if (poll(fds, 2, -1) == -1) {
            if (errno == EINTR) continue;
            throw errno_error("poll");
        }
        if (fds[1].revents != 0) return 0;
        const ssize_t n = ::read(fd, buf, len);
        if (n >= 0) return static_cast<std::size_t>(n);
        if (errno != EINTR) throw errno_error("read");
    }
}

}  // namespace

std::vector<std::string> build_args(const CaptureConfig& config) {
    // -y: the segment muxer does not need it. With seg-00000.webm already
    // present and no -y, ffmpeg n9.0.2 replaced it without asking. A plain
    // file output in the same test asked to overwrite, read EOF from stdin
    // (/dev/null), and refused to write. -y keeps any future plain-file
    // output from failing on that prompt.
    std::vector<std::string> args = {config.ffmpeg_path, "-y", "-hide_banner", "-loglevel",
                                     "error"};
    if (config.camera_format == "lavfi") {
        // Synthetic source for testing without a camera (see the README's
        // Clipping Mode section). A real device paces itself; lavfi
        // generates frames as fast as the CPU allows, which makes the fps=
        // filters and -segment_time meaningless. -re paces it to realtime.
        args.push_back("-re");
    }
    args.insert(args.end(), {"-f", config.camera_format});
    if (config.camera_format == "v4l2") {
        // v4l2 selects the device-side format with -input_format. Without
        // this the driver picks its own default, and on a UVC webcam that is
        // the largest raw mode it advertises (3840x2160 yuyv422 at 1fps),
        // which dequeues corrupted buffers and yields no frames at all.
        args.insert(args.end(), {"-input_format", "mjpeg", "-video_size", config.camera_size,
                                 "-framerate", config.camera_fps});
    }
    args.insert(args.end(), {"-i", config.camera_input});

    // Output 1: raw frames for inference.
    const std::string fps = std::format("{:.4f}", 1.0 / config.frame_interval);
    args.insert(args.end(), {"-vf", "fps=" + fps + ",scale=224:224", "-f", "rawvideo", "-pix_fmt",
                             "rgb24", "pipe:1"});

    // Output 2: rolling recorded segments (constant 15fps against real timestamps).
    // -g pins the keyframe interval to exactly one segment's worth of
    // frames. The segment muxer can only cut on a keyframe, and libvpx's
    // default GOP is much longer than -segment_time, so without this every
    // segment overshoots to the next keyframe: 12s of requested segment
    // came out as ~12.5s, and a shorter -segment_time was ignored almost
    // entirely. Clip length is a real setting (Clipping Mode advertises 8s
    // clips), so it has to be honored.
    // -threads 1 is a large CPU win, not a throttle. libvpx defaults to a
    // thread per core and splits each frame into horizontal tiles. At
    // 640x360 the tiles are small enough that the per-frame join costs more
    // than the encoding it parallelizes. Measured over 450 frames at this
    // exact bitrate and resolution: 23.3 CPU-seconds at the default, 5.0 at
    // -threads 1, 4.7x less CPU for a byte-for-byte comparable output, and
    // lower wall-clock time too, so the realtime deadline gets no tighter.
    // This matters on the Pi, where the agent shares four cores with the
    // server and the tunnel, and where there is no hardware encoder to fall
    // back to (BCM2712 dropped the legacy codec blocks entirely), so every
    // recorded frame is encoded in software.
    const std::string segment_frames =
        std::to_string(std::max(1, config.segment_secs * kSegmentFps));
    const std::string segment_fps = std::to_string(kSegmentFps);
    args.insert(args.end(), {"-c:v",
                             "libvpx",
                             "-deadline",
                             "realtime",
                             "-cpu-used",
                             "8",
                             "-threads",
                             "1",
                             "-b:v",
                             "1M",
                             "-vf",
                             "fps=" + segment_fps + ",scale=640:360",
                             "-r",
                             segment_fps,
                             "-an",
                             "-g",
                             segment_frames,
                             "-keyint_min",
                             segment_frames,
                             "-f",
                             "segment",
                             "-segment_time",
                             std::to_string(config.segment_secs),
                             "-reset_timestamps",
                             "1",
                             (config.seg_dir / "seg-%05d.webm").string()});

    // Output 3: live-view MJPEG on fd 3, split into JPEGs by read_mjpeg.
    args.insert(args.end(), {"-vf", "fps=12,scale=640:360", "-q:v", "5", "-f", "mjpeg", "pipe:3"});
    return args;
}

struct FfmpegCapture::Impl {
    CaptureConfig config;
    FrameHandler on_frame;
    JpegHandler on_jpeg;
    CaptureTiming timing;
    std::atomic<std::uint64_t> frames_dropped{0};

    std::mutex mu;
    std::condition_variable cv;
    bool stopped = false;  // Never cleared: a stopped capture is replaced, not restarted.
    pid_t live_pid = -1;   // The current child until it exits. Signal it only under mu.

    // start() assigns supervisor and stop_and_wait joins it.
    std::mutex lifecycle_mu;
    std::jthread supervisor;

    // Caller holds mu and has checked stopped.
    std::unique_ptr<Process> spawn_locked() {
        std::filesystem::create_directories(config.seg_dir);
        const std::vector<std::string> args = build_args(config);
        std::vector<char*> argv;
        for (const auto& arg : args) argv.push_back(const_cast<char*>(arg.c_str()));
        argv.push_back(nullptr);

        // The fd-3 pipe comes first so the test that closes stdin can land its
        // write end on fd 3.
        Pipe mjpeg = make_pipe();
        Pipe out = make_pipe();
        Pipe err = make_pipe();
        Fd wake(eventfd(0, EFD_CLOEXEC));
        if (wake.get() == -1) throw errno_error("eventfd");

        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0);
        posix_spawn_file_actions_adddup2(&actions, out.write.get(), 1);
        posix_spawn_file_actions_adddup2(&actions, err.write.get(), 2);
        posix_spawn_file_actions_adddup2(&actions, mjpeg.write.get(), 3);
        // An inheritable fd from elsewhere in the agent, such as an HTTP
        // socket, must not stay open in ffmpeg.
        posix_spawn_file_actions_addclosefrom_np(&actions, 4);

        // The child inherits the spawning thread's signal mask, and the agent
        // blocks SIGTERM for its sigwait thread. A blocked SIGTERM would make
        // every stop wait out stop_timeout and SIGKILL ffmpeg mid-segment.
        // SIGPIPE goes back to default too.
        posix_spawnattr_t attr;
        posix_spawnattr_init(&attr);
        sigset_t empty;
        sigemptyset(&empty);
        posix_spawnattr_setsigmask(&attr, &empty);
        sigset_t defaults;
        sigemptyset(&defaults);
        sigaddset(&defaults, SIGPIPE);
        posix_spawnattr_setsigdefault(&attr, &defaults);
        posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF);

        pid_t pid;
        const int rc = posix_spawnp(&pid, argv[0], &actions, &attr, argv.data(), environ);
        posix_spawnattr_destroy(&attr);
        posix_spawn_file_actions_destroy(&actions);
        if (rc != 0) throw std::system_error(rc, std::generic_category(), "posix_spawn " + args[0]);

        live_pid = pid;
        try {
            auto proc = std::make_unique<Process>();
            proc->pid = pid;
            proc->stdout_fd = std::move(out.read);
            proc->stderr_fd = std::move(err.read);
            proc->mjpeg_fd = std::move(mjpeg.read);
            proc->wake = std::move(wake);
            // Each reader gets its own process's streams, never the current
            // one: a reader that followed a restart would split the new
            // stream with the new reader and hand inference misaligned frames.
            Process& p = *proc;
            p.stdout_reader = std::jthread([this, &p](std::stop_token st) { read_frames(st, p); });
            p.mjpeg_reader = std::jthread([this, &p](std::stop_token st) { read_mjpeg(st, p); });
            p.stderr_reader = std::jthread([this, &p](std::stop_token st) { read_stderr(st, p); });
            p.consumer = std::jthread([this, &p](std::stop_token st) { process_frames(st, p); });
            log(std::format("✓ Camera capture started ({}: {})", config.camera_format,
                            config.camera_input));
            return proc;
        } catch (...) {
            abandon_locked(pid);
            throw;
        }
    }

    // Kills and reaps a child that no supervisor took over. Caller holds mu.
    void abandon_locked(pid_t pid) {
        kill(pid, SIGKILL);
        while (waitpid(pid, nullptr, 0) == -1 && errno == EINTR) {
        }
        live_pid = -1;
    }

    // Drains stdout continuously and hands off only the newest frame.
    //
    // This loop must never wait on the frame handler. ffmpeg writes raw
    // frames, the recorded segments and the live view from one process, so if
    // this loop stops reading while inference runs, the unread backlog fills
    // the pipe buffer, ffmpeg blocks on its write to pipe:1, and recording and
    // the dashboard stream stall along with classification.
    void read_frames(const std::stop_token& stop, Process& p) {
        for (;;) {
            std::vector<std::uint8_t> frame(kFrameBytes);
            std::size_t filled = 0;
            while (filled < kFrameBytes) {
                const std::size_t n = read_some(stop, p, p.stdout_fd.get(), frame.data() + filled,
                                                kFrameBytes - filled);
                if (n == 0) return;
                filled += n;
            }
            const auto demuxed_at = Clock::now();
            {
                std::lock_guard lock(p.slot_mu);
                if (p.slot) {
                    // The handler is still busy with the previous frame.
                    // Overwrite it: the freshest frame is the useful one.
                    const std::uint64_t dropped = ++frames_dropped;
                    if (dropped % kDropLogEvery == 0) {
                        log(std::format("ffmpeg: dropped {} frames (inference behind cadence)",
                                        dropped));
                    }
                }
                p.slot = Process::Frame{std::move(frame), demuxed_at};
            }
            p.slot_cv.notify_one();
        }
    }

    void process_frames(const std::stop_token& stop, Process& p) {
        for (;;) {
            Process::Frame frame;
            {
                std::unique_lock lock(p.slot_mu);
                if (!p.slot_cv.wait(lock, stop, [&] { return p.slot.has_value(); })) return;
                frame = std::move(*p.slot);
                p.slot.reset();
            }
            try {
                on_frame(std::move(frame.bytes), frame.demuxed_at);
            } catch (const std::exception& err) {
                // One bad frame must not kill the consumer and silently end
                // all classification for the life of the process.
                log(std::string("frame handler error: ") + err.what());
            }
        }
    }

    void read_mjpeg(const std::stop_token& stop, Process& p) {
        std::string buf;
        std::size_t eoi_from = kSoi.size();
        char chunk[65536];
        while (const std::size_t n = read_some(stop, p, p.mjpeg_fd.get(), chunk, sizeof chunk)) {
            buf.append(chunk, n);
            for (;;) {
                const auto soi = buf.find(kSoi);
                if (soi == std::string::npos) {
                    // Keep a trailing 0xFF: it may be the first half of an SOI.
                    buf.erase(0, buf.size() - (buf.ends_with('\xFF') ? 1 : 0));
                    break;
                }
                buf.erase(0, soi);
                const auto eoi = buf.find(kEoi, eoi_from);
                if (eoi == std::string::npos) {
                    eoi_from = std::max(kSoi.size(), buf.size() - 1);
                    break;
                }
                std::string jpeg = buf.substr(0, eoi + kEoi.size());
                buf.erase(0, jpeg.size());
                eoi_from = kSoi.size();
                try {
                    on_jpeg(std::move(jpeg));
                } catch (const std::exception& err) {
                    // A dead fd-3 reader stalls ffmpeg as surely as a dead
                    // stdout reader, so keep reading.
                    log(std::string("live frame handler error: ") + err.what());
                }
            }
            if (buf.size() > kMaxJpegBytes) {
                log("ffmpeg: live-view JPEG over 2 MiB with no EOI, resyncing");
                buf.clear();
                eoi_from = kSoi.size();
            }
        }
    }

    void read_stderr(const std::stop_token& stop, Process& p) {
        std::string pending;
        char chunk[4096];
        auto emit = [](std::string_view line) {
            const std::string_view msg = trim(line);
            if (!msg.empty()) log(std::format("ffmpeg: {}", msg));
        };
        while (const std::size_t n = read_some(stop, p, p.stderr_fd.get(), chunk, sizeof chunk)) {
            pending.append(chunk, n);
            std::size_t start = 0;
            for (std::size_t nl; (nl = pending.find('\n', start)) != std::string::npos;) {
                emit(std::string_view(pending).substr(start, nl - start));
                start = nl + 1;
            }
            pending.erase(0, start);
            while (pending.size() >= kMaxStderrLine) {
                emit(std::string_view(pending).substr(0, kMaxStderrLine));
                pending.erase(0, kMaxStderrLine);
            }
        }
        emit(pending);
    }

    // Waits for the child to exit and reaps it. Returns the exit code, or the
    // negated signal number, as Python's returncode does.
    int wait_exit(pid_t pid) {
        siginfo_t info{};
        // WNOWAIT leaves a zombie until live_pid is cleared, so the pid
        // cannot be reused while stop_and_wait may still signal it.
        while (waitid(P_PID, static_cast<id_t>(pid), &info, WEXITED | WNOWAIT) == -1) {
            if (errno != EINTR) throw errno_error("waitid");
        }
        {
            std::lock_guard lock(mu);
            live_pid = -1;
        }
        cv.notify_all();
        while (waitpid(pid, nullptr, 0) == -1) {
            if (errno != EINTR) throw errno_error("waitpid");
        }
        return info.si_code == CLD_EXITED ? info.si_status : -info.si_status;
    }

    void supervise(std::unique_ptr<Process> proc) {
        const double delay_secs = std::chrono::duration<double>(timing.restart_delay).count();
        for (;;) {
            const int code = wait_exit(proc->pid);
            // Tear the threads down before anything spawns the next process,
            // or a consumer parked in on_frame resumes against it.
            proc.reset();
            std::unique_lock lock(mu);
            if (stopped) {
                log("Capture stopped — monitoring is OFF");
                return;
            }
            log(std::format("ffmpeg exited ({}) — restarting in {}s...", code, delay_secs));
            while (!proc) {
                if (cv.wait_for(lock, timing.restart_delay, [&] { return stopped; })) return;
                try {
                    proc = spawn_locked();
                } catch (const std::system_error& err) {
                    // Keep retrying so a transient failure doesn't leave
                    // capture dead until the next monitor toggle.
                    log(std::format("ffmpeg restart failed: {}", err.what()));
                }
            }
        }
    }
};

FfmpegCapture::FfmpegCapture(CaptureConfig config, FrameHandler on_frame, JpegHandler on_jpeg,
                             CaptureTiming timing)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = std::move(config);
    impl_->on_frame = std::move(on_frame);
    impl_->on_jpeg = std::move(on_jpeg);
    impl_->timing = timing;
}

FfmpegCapture::~FfmpegCapture() { stop_and_wait(); }

void FfmpegCapture::start() {
    Impl& impl = *impl_;
    std::lock_guard lifecycle(impl.lifecycle_mu);
    if (impl.supervisor.joinable()) return;  // Already running.
    std::lock_guard lock(impl.mu);
    if (impl.stopped) return;
    std::unique_ptr<Process> proc = impl.spawn_locked();
    const pid_t pid = proc->pid;
    try {
        impl.supervisor = std::jthread(
            [&impl, proc = std::move(proc)]() mutable { impl.supervise(std::move(proc)); });
    } catch (...) {
        impl.abandon_locked(pid);
        throw;
    }
}

void FfmpegCapture::stop_and_wait() {
    Impl& impl = *impl_;
    {
        std::lock_guard lock(impl.mu);
        impl.stopped = true;
        if (impl.live_pid != -1) kill(impl.live_pid, SIGTERM);
    }
    impl.cv.notify_all();
    // Spawns check stopped under mu, so no child can start after this point.
    // A concurrent start() that spawned first hands its child to the
    // supervisor before releasing lifecycle_mu.
    std::lock_guard lifecycle(impl.lifecycle_mu);
    {
        std::unique_lock lock(impl.mu);
        if (!impl.cv.wait_for(lock, impl.timing.stop_timeout,
                              [&] { return impl.live_pid == -1; })) {
            kill(impl.live_pid, SIGKILL);
        }
    }
    if (impl.supervisor.joinable()) impl.supervisor.join();
}

std::uint64_t FfmpegCapture::frames_dropped() const { return impl_->frames_dropped; }

}  // namespace bunnycam
