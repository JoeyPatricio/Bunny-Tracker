"""Port of capture.mjs's startCapture(): ffmpeg spawn + frame demuxing.

Same camera config and dshow-specific args as capture.mjs, same 5s
auto-restart on exit. Two deliberate differences from the JS version:

1. The MJPEG live-view stream is NOT a third piped output (`pipe:3`). Node's
   libuv implements an undocumented Windows CRT protocol (populating
   STARTUPINFO's reserved fields with a serialized fd-inheritance table) to
   hand a child process an inherited handle beyond the standard 0/1/2 —
   Python's subprocess module has no equivalent, and reimplementing that
   low-level protocol by hand is fragile and not worth it here. Instead,
   ffmpeg writes the live-view frame to a single continuously-overwritten
   JPEG file (`-f image2 -update 1`), which capture.py polls by mtime. Same
   ~12fps cadence, same result for the dashboard, no extra pipe plumbing —
   and it works unchanged on the eventual Pi target too.
2. The raw-frame stdout scanner uses a bytearray with an in-place delete
   (`del pending[:n]`) rather than repeated `bytes` concatenation, avoiding
   the same quadratic-reassembly shape flagged for the MJPEG parser in
   fixes.md 4.5b (there is no MJPEG pipe parser to port here anymore, since
   the live-view frame is now file-based, but the raw-frame buffer has the
   identical risk and is fixed the same way).
3. Reading stdout and running inference are two separate tasks joined by a
   single-slot handoff, rather than one loop that awaits the frame handler
   inline. See _read_frames for why awaiting inline stalls ffmpeg outright
   (fixes.md 2.1).
"""
import asyncio
import os
import time
from pathlib import Path

FRAME_SIZE = 224 * 224 * 3
RESTART_DELAY_SEC = 5

# Recorded segments run at a constant 15fps against real timestamps.
SEGMENT_FPS = 15

# Frames are dropped silently by design (see _read_frames), which is exactly
# how a Pi falling behind the frame budget would hide. Log a line per N drops
# so a sustained backlog is visible in the pm2 log without spamming it.
DROP_LOG_EVERY = 100


def _log_task_exception(task) -> None:
    """gather()'s future used to be assigned and never awaited, so a crash in
    any reader vanished into the discarded future and the agent carried on
    looking healthy. Surface it instead.

    CancelledError is not a failure here. The readers catch cancellation and
    return normally, which leaves the gather future holding a CancelledError as
    its exception rather than marked cancelled, so task.cancelled() alone is not
    enough to filter out an ordinary stop or restart.
    """
    if task.cancelled():
        return
    err = task.exception()
    if err is not None and not isinstance(err, asyncio.CancelledError):
        print(f"capture task failed: {err!r}")


class FfmpegCapture:
    def __init__(self, *, ffmpeg_path: str, camera_format: str, camera_input: str,
                 camera_size: str, camera_fps: str, frame_interval: float,
                 segment_secs: int, seg_dir: Path, live_frame_path: Path,
                 on_frame, on_stopped=None):
        self.ffmpeg_path = ffmpeg_path
        self.camera_format = camera_format
        self.camera_input = camera_input
        self.camera_size = camera_size
        self.camera_fps = camera_fps
        self.frame_interval = frame_interval
        self.segment_secs = segment_secs
        self.seg_dir = seg_dir
        self.live_frame_path = live_frame_path
        self.on_frame = on_frame  # async callback(frame_bytes)
        self.on_stopped = on_stopped

        self.proc: asyncio.subprocess.Process | None = None
        self.paused = False
        self._readers: asyncio.Future | None = None
        self._supervisor: asyncio.Future | None = None
        # Single-slot handoff between the stdout reader and the frame handler.
        self._newest_frame: bytes | None = None
        self._frame_ready: asyncio.Event | None = None
        self.frames_dropped = 0

    def _build_args(self) -> list[str]:
        # -y: auto-overwrite the live-view JPEG on restart. Without it, ffmpeg
        # prompts on stdin for confirmation when the file from a previous run
        # still exists; stdin is DEVNULL, so it hangs indefinitely holding the
        # camera open rather than failing fast.
        args = [self.ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error"]
        if self.camera_format == "lavfi":
            # Synthetic source for testing without a camera (see the README's
            # Clipping Mode section). A real device paces itself; lavfi
            # generates frames as fast as the CPU allows, which makes the fps=
            # filters and -segment_time meaningless. -re paces it to realtime.
            args += ["-re"]
        args += ["-f", self.camera_format]
        if self.camera_format == "dshow":
            args += ["-rtbufsize", "100M", "-vcodec", "mjpeg",
                      "-video_size", self.camera_size, "-framerate", self.camera_fps]
        elif self.camera_format == "v4l2":
            # Same intent as the dshow branch, different spelling: v4l2 selects
            # the device-side format with -input_format, not -vcodec. Without
            # this the driver picks its own default, and on a UVC webcam that
            # is the largest raw mode it advertises — 3840x2160 yuyv422 at 1fps,
            # which dequeues corrupted buffers and yields no frames at all.
            args += ["-input_format", "mjpeg",
                      "-video_size", self.camera_size, "-framerate", self.camera_fps]
        args += ["-i", self.camera_input]
        # Output 1: raw frames for inference.
        fps = f"{(1 / self.frame_interval):.4f}"
        args += ["-vf", f"fps={fps},scale=224:224", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        # Output 2: rolling recorded segments (constant 15fps against real timestamps).
        # -g pins the keyframe interval to exactly one segment's worth of
        # frames. The segment muxer can only cut on a keyframe, and libvpx's
        # default GOP is much longer than -segment_time, so without this every
        # segment overshoots to the next keyframe — 12s of requested segment
        # came out as ~12.5s, and a shorter -segment_time was ignored almost
        # entirely. Clip length is a real setting now (Clipping Mode advertises
        # 8s clips), so it has to be honored.
        segment_frames = max(1, int(self.segment_secs) * SEGMENT_FPS)
        args += [
            "-c:v", "libvpx", "-deadline", "realtime", "-cpu-used", "8",
            "-b:v", "1M", "-vf", f"fps={SEGMENT_FPS},scale=640:360", "-r", str(SEGMENT_FPS), "-an",
            "-g", str(segment_frames), "-keyint_min", str(segment_frames),
            "-f", "segment", "-segment_time", str(self.segment_secs), "-reset_timestamps", "1",
            str(self.seg_dir / "seg-%05d.webm"),
        ]
        # Output 3: live-view frame, continuously overwritten (see module docstring).
        args += ["-vf", "fps=12,scale=640:360", "-q:v", "5", "-f", "image2", "-update", "1",
                  str(self.live_frame_path)]
        return args

    async def start(self) -> None:
        self.seg_dir.mkdir(parents=True, exist_ok=True)
        args = self._build_args()
        proc = await asyncio.create_subprocess_exec(
            *args, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        self.proc = proc
        print(f"✓ Camera capture started ({self.camera_format}: {self.camera_input})")

        # Drop anything left over from the previous process: a frame captured
        # before a restart describes a scene that is seconds stale.
        self._newest_frame = None
        self._frame_ready = asyncio.Event()

        # The stream objects are passed in, never re-read from self.proc inside
        # the loops. _watch_exit rebinds self.proc on restart, and a reader that
        # re-read it each iteration would start consuming the NEW process's
        # stdout while the new reader did too, splitting the byte stream between
        # them and handing inference misaligned garbage (fixes.md 2.2).
        self._readers = asyncio.gather(
            self._read_frames(proc.stdout),
            self._read_stderr(proc.stderr),
            self._process_frames(),
        )
        self._readers.add_done_callback(_log_task_exception)
        # Deliberately NOT part of the gather above: _watch_exit cancels that
        # gather on restart, and a task cannot cleanly cancel itself.
        self._supervisor = asyncio.ensure_future(self._watch_exit(proc))
        self._supervisor.add_done_callback(_log_task_exception)

    async def _read_frames(self, stdout: asyncio.StreamReader) -> None:
        """Drain stdout continuously and hand off only the newest frame.

        This loop must never await the frame handler. ffmpeg writes raw frames,
        the recorded segments and the live-view JPEG from one process, so if
        this loop stops reading while inference runs, the unread backlog fills
        the OS pipe buffer, ffmpeg blocks on its write to pipe:1, and recording
        and the dashboard stream stall along with classification. The JS
        original dropped frames for the same reason; the port awaited the
        handler inline, which made its `busy` flag unreachable (fixes.md 2.1).
        """
        pending = bytearray()
        try:
            while True:
                chunk = await stdout.read(65536)
                if not chunk:
                    break
                pending += chunk
                while len(pending) >= FRAME_SIZE:
                    frame = bytes(pending[:FRAME_SIZE])
                    del pending[:FRAME_SIZE]  # in-place, not a fresh concat — see module docstring
                    if self._newest_frame is not None:
                        # The handler is still busy with the previous frame.
                        # Overwrite it: the freshest frame is the useful one.
                        self.frames_dropped += 1
                        if self.frames_dropped % DROP_LOG_EVERY == 0:
                            print(f"ffmpeg: dropped {self.frames_dropped} frames (inference behind cadence)")
                    self._newest_frame = frame
                    self._frame_ready.set()
        except asyncio.CancelledError:
            pass

    async def _process_frames(self) -> None:
        """Consume the single-slot handoff, one frame at a time."""
        try:
            while True:
                await self._frame_ready.wait()
                self._frame_ready.clear()
                frame, self._newest_frame = self._newest_frame, None
                if frame is None:
                    continue
                try:
                    await self.on_frame(frame)
                except asyncio.CancelledError:
                    raise
                except Exception as err:
                    # One bad frame must not kill the consumer and silently
                    # end all classification for the life of the process.
                    print("frame handler error:", err)
        except asyncio.CancelledError:
            pass

    async def _read_stderr(self, stderr: asyncio.StreamReader) -> None:
        try:
            while True:
                line = await stderr.readline()
                if not line:
                    break
                msg = line.decode(errors="replace").strip()
                if msg:
                    print("ffmpeg:", msg)
        except asyncio.CancelledError:
            pass

    async def _stop_readers(self) -> None:
        readers, self._readers = self._readers, None
        if readers is None:
            return
        readers.cancel()
        try:
            await readers
        except asyncio.CancelledError:
            pass
        except Exception as err:
            print(f"capture task failed during shutdown: {err!r}")

    async def _watch_exit(self, proc: asyncio.subprocess.Process) -> None:
        returncode = await proc.wait()
        # Tear the readers down BEFORE anything can rebind self.proc, or a
        # reader parked in on_frame resumes against the next process.
        await self._stop_readers()
        if self.proc is proc:
            self.proc = None
        if self.paused:
            print("Capture stopped — monitoring is OFF")
            if self.on_stopped:
                self.on_stopped()
            return
        print(f"ffmpeg exited ({returncode}) — restarting in {RESTART_DELAY_SEC}s...")
        await asyncio.sleep(RESTART_DELAY_SEC)
        if not self.paused:
            await self.start()

    def stop(self) -> None:
        self.paused = True
        if self.proc:
            self.proc.terminate()


class LiveFramePoller:
    """Polls the continuously-overwritten live-view JPEG file for changes
    (replaces the piped-MJPEG scanner — see ffmpeg_io module docstring)."""

    def __init__(self, path: Path, on_new_frame, poll_interval: float = 0.08):
        self.path = path
        self.on_new_frame = on_new_frame  # async callback(jpeg_bytes)
        self.poll_interval = poll_interval
        self._last_mtime = 0.0
        self._task: asyncio.Task | None = None
        self._running = False

    async def _loop(self) -> None:
        while self._running:
            try:
                mtime = os.path.getmtime(self.path)
                if mtime != self._last_mtime:
                    self._last_mtime = mtime
                    data = await asyncio.to_thread(self.path.read_bytes)
                    if data:
                        await self.on_new_frame(data)
            except OSError:
                pass  # file not written yet
            await asyncio.sleep(self.poll_interval)

    def start(self) -> None:
        self._running = True
        self._task = asyncio.ensure_future(self._loop())

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
