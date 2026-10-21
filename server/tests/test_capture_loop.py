"""Regression gate for agent/ffmpeg_io.py.

Drives FfmpegCapture's reader/consumer tasks directly against a real
asyncio.StreamReader instead of spawning ffmpeg, so this runs anywhere with no
camera and no ffmpeg binary. Same shape as test_motion_parity.py: plain main(),
exit 1 on failure.

Usage (from server/, runtime venv):
    .venv/Scripts/python.exe -m tests.test_capture_loop
"""
import asyncio
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import capture, ffmpeg_io
from agent.ffmpeg_io import FRAME_SIZE, FfmpegCapture
from tests.checks import check, finish


def make_capture(on_frame) -> FfmpegCapture:
    """A capture object wired for direct task driving. start() is never called,
    so none of the ffmpeg/camera settings matter here."""
    return FfmpegCapture(
        ffmpeg_path="ffmpeg", camera_format="dshow", camera_input="video=none",
        camera_size="1280x720", camera_fps="15", frame_interval=1.5,
        segment_secs=12, seg_dir=Path("."), live_frame_path=Path("live.jpg"),
        on_frame=on_frame,
    )


def frame_bytes(marker: int) -> bytes:
    """A full frame whose first byte identifies it."""
    return bytes([marker]) + b"\x00" * (FRAME_SIZE - 1)


class FakeProc:
    def __init__(self, returncode: int = 1):
        self.returncode = returncode

    async def wait(self) -> int:
        return self.returncode


async def test_slow_handler_does_not_stall_reader() -> None:
    """The reader must keep draining stdout while on_frame runs."""
    handled: list[int] = []

    async def slow_on_frame(frame: bytes) -> None:
        await asyncio.sleep(0.05)
        handled.append(frame[0])

    cap = make_capture(slow_on_frame)
    cap._frame_ready = asyncio.Event()

    total = 10
    reader = asyncio.StreamReader()
    for i in range(total):
        reader.feed_data(frame_bytes(i))
    reader.feed_eof()

    consumer = asyncio.ensure_future(cap._process_frames())
    # 10 frames at 50ms each would be 500ms if the reader awaited the handler
    # inline. It must finish in a small fraction of that.
    started = asyncio.get_running_loop().time()
    await asyncio.wait_for(cap._read_frames(reader), timeout=2.0)
    elapsed = asyncio.get_running_loop().time() - started

    check("2.1 reader drains stdout without waiting on the handler", elapsed < 0.15, f"took {elapsed:.3f}s")
    check("2.1 frames were dropped rather than queued", cap.frames_dropped > 0, f"dropped {cap.frames_dropped}")

    # Let the consumer finish whatever it last picked up.
    await asyncio.sleep(0.15)
    consumer.cancel()
    await asyncio.gather(consumer, return_exceptions=True)

    check(
        "2.1 handler saw far fewer frames than were read",
        len(handled) < total,
        f"handled {len(handled)} of {total}",
    )
    check(
        "2.1 dropped + handled accounts for every frame read",
        cap.frames_dropped + len(handled) == total,
        f"{cap.frames_dropped} dropped + {len(handled)} handled != {total}",
    )
    check(
        "2.1 the newest frame is the one that survives",
        handled and handled[-1] == total - 1,
        f"last handled {handled[-1] if handled else None}, expected {total - 1}",
    )


async def test_reader_is_bound_to_its_own_stream() -> None:
    """The loop must not re-read self.proc each iteration."""
    handled: list[int] = []

    async def on_frame(frame: bytes) -> None:
        handled.append(frame[0])

    cap = make_capture(on_frame)
    cap._frame_ready = asyncio.Event()

    original = asyncio.StreamReader()
    consumer = asyncio.ensure_future(cap._process_frames())
    task = asyncio.ensure_future(cap._read_frames(original))

    original.feed_data(frame_bytes(1))
    await asyncio.sleep(0.02)

    # Simulate a restart rebinding self.proc underneath a running reader.
    replacement = asyncio.StreamReader()
    replacement.feed_data(frame_bytes(99))
    cap.proc = type("P", (), {"stdout": replacement, "stderr": replacement})()

    original.feed_data(frame_bytes(2))
    original.feed_eof()
    await asyncio.wait_for(task, timeout=2.0)
    await asyncio.sleep(0.05)
    consumer.cancel()
    await asyncio.gather(consumer, return_exceptions=True)

    check(
        "2.2 reader stayed on its own stream after self.proc was rebound",
        99 not in handled and handled == [1, 2],
        f"handled {handled}",
    )


async def test_restart_cancels_readers_first() -> None:
    """Readers must be torn down before start() rebinds state."""
    cap = make_capture(lambda frame: asyncio.sleep(0))
    order: list[str] = []
    parked = asyncio.Event()

    async def never_ends() -> None:
        try:
            parked.set()
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            order.append("readers cancelled")
            raise

    cap._readers = asyncio.gather(never_ends())
    await parked.wait()

    async def fake_start() -> None:
        order.append("start")

    cap.start = fake_start
    saved_delay = ffmpeg_io.RESTART_DELAY_SEC
    ffmpeg_io.RESTART_DELAY_SEC = 0
    try:
        await asyncio.wait_for(cap._watch_exit(FakeProc()), timeout=2.0)
    finally:
        ffmpeg_io.RESTART_DELAY_SEC = saved_delay

    check(
        "2.2 readers cancelled before restart",
        order == ["readers cancelled", "start"],
        f"order was {order}",
    )
    check("2.2 reader handle cleared", cap._readers is None)


async def test_stale_proc_not_cleared_on_restart() -> None:
    """A late exit from an OLD process must not null out the NEW self.proc."""
    cap = make_capture(lambda frame: asyncio.sleep(0))
    cap.paused = True
    old_proc = FakeProc()
    new_proc = FakeProc()
    cap.proc = new_proc
    await asyncio.wait_for(cap._watch_exit(old_proc), timeout=2.0)
    check("2.2 old process exit left the current proc alone", cap.proc is new_proc)


async def test_handler_error_does_not_kill_consumer() -> None:
    """One bad frame must not end classification for the process lifetime."""
    seen: list[int] = []

    async def flaky(frame: bytes) -> None:
        if frame[0] == 0:
            raise ValueError("boom")
        seen.append(frame[0])

    cap = make_capture(flaky)
    cap._frame_ready = asyncio.Event()
    consumer = asyncio.ensure_future(cap._process_frames())

    for marker in (0, 1):
        cap._newest_frame = frame_bytes(marker)
        cap._frame_ready.set()
        await asyncio.sleep(0.02)

    alive = not consumer.done()
    consumer.cancel()
    await asyncio.gather(consumer, return_exceptions=True)

    check("2.1 consumer survived a handler exception", alive)
    check("2.1 frame after the failing one was still handled", seen == [1], f"seen {seen}")


def make_agent():
    with patch.object(capture, "AgentClient", return_value=SimpleNamespace(post=AsyncMock())):
        agent = capture.Agent()
    agent.decision_loop = capture.AgentDecisionLoop()
    return agent


async def test_blocking_inference_keeps_reader_and_timer_running() -> None:
    """Synchronous inference must leave the event loop free to drain the pipe."""
    agent = make_agent()
    entered, release = threading.Event(), threading.Event()
    original = agent.decision_loop.handle_frame

    def blocking(frame):
        entered.set()
        if not release.wait(3):
            raise TimeoutError("test did not release inference")
        return original(frame)

    agent.decision_loop.handle_frame = blocking
    cap = make_capture(agent._on_frame)
    cap._frame_ready = asyncio.Event()
    started = asyncio.get_running_loop().time()
    consumer = asyncio.create_task(cap._process_frames())
    cap._newest_frame = frame_bytes(0)
    cap._frame_ready.set()
    try:
        check("synchronous worker started", await asyncio.to_thread(entered.wait, 1))
        reader = asyncio.StreamReader()
        for marker in range(1, 10):
            reader.feed_data(frame_bytes(marker))
        reader.feed_eof()
        await asyncio.wait_for(cap._read_frames(reader), 1)
        await asyncio.sleep(0.01)
        elapsed = asyncio.get_running_loop().time() - started
        check("timer and reader finish promptly during synchronous inference", elapsed < 0.5,
              f"event loop stalled for {elapsed:.3f}s")
        check("reader drops stale frames during synchronous inference", cap.frames_dropped == 8)
    finally:
        release.set()
        consumer.cancel()
        await consumer


async def test_config_waits_for_serialized_decisions() -> None:
    """Live config must wait for inference and the preceding decision's effects."""
    agent = make_agent()
    agent.decision_loop.clipping = True
    agent.decision_loop.clip_cooldown_sec = 0
    agent.decision_loop.motion = SimpleNamespace(motion_level=lambda frame: {
        "level": 9, "hotCells": 1, "lighting": False,
    })
    agent._queue_clip = AsyncMock()
    entered, release = threading.Event(), threading.Event()
    original = agent.decision_loop.handle_frame
    calls = []

    def blocking(frame):
        calls.append(frame[0, 0, 0])
        entered.set()
        if not release.wait(3):
            raise TimeoutError("test did not release inference")
        return original(frame)

    agent.decision_loop.handle_frame = blocking
    first = asyncio.create_task(agent._on_frame(frame_bytes(0)))
    second = update = None
    try:
        check("first decision entered the worker", await asyncio.to_thread(entered.wait, 1))
        second = asyncio.create_task(agent._on_frame(frame_bytes(1)))
        update = asyncio.create_task(agent._apply_monitor_state({
            "enabled": True, "clipping": True, "clippingMotionThreshold": 9, "clippingDryRun": True,
        }))
        await asyncio.sleep(0.02)
        check("a second frame cannot mutate decision state concurrently", calls == [0])
        check("config does not change under a running decision", not update.done()
              and agent.decision_loop.clip_threshold == capture.CLIPPING_MOTION_THRESHOLD)
        release.set()
        await asyncio.gather(first, second, update)
        check("queued decisions run serially", calls == [0, 1])
        check("live settings apply after preceding decisions", agent.decision_loop.clip_threshold == 9
              and agent.decision_loop.dry_run)
        posts = agent.client.post.call_args_list
        check("completed decisions publish using their original dry-run setting",
              len(posts) == 2 and all(call.kwargs["json"]["captured"] for call in posts)
              and agent._queue_clip.call_count == 2)
    finally:
        release.set()
        await asyncio.gather(*(task for task in (first, second, update) if task), return_exceptions=True)


async def test_cancelled_worker_finishes_before_monitor_reset() -> None:
    """Monitor stop and mode flip join cancelled workers before resetting state."""
    for state, worker_fails in (
        ({"enabled": False}, False),
        ({"enabled": True, "clipping": True}, False),
        ({"enabled": False}, True),
    ):
        agent = make_agent()
        entered, release = threading.Event(), threading.Event()
        original = agent.decision_loop.handle_frame
        old_motion = agent.decision_loop.motion
        agent.start_capture = AsyncMock()

        def blocking(frame, *, agent=agent, entered=entered, release=release,
                     original=original, worker_fails=worker_fails):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test did not release inference")
            result = original(frame)
            agent.decision_loop.motion_streak = 100
            agent.decision_loop.streak_count = 100
            agent.decision_loop.streak_scores["rearing"] = 100
            agent.decision_loop.last_capture_at = 100
            if worker_fails:
                raise ValueError("inference failed during shutdown")
            result["action"] = "motion_capture"
            return result

        agent.decision_loop.handle_frame = blocking
        agent._upload_motion_clip = AsyncMock()
        cap = make_capture(agent._on_frame)
        agent.ffmpeg = cap
        cap._frame_ready = asyncio.Event()
        cap._newest_frame = frame_bytes(0)
        cap._frame_ready.set()
        cap._readers = asyncio.gather(cap._process_frames())
        readers = cap._readers
        update = None
        try:
            check("worker entered before monitor change", await asyncio.to_thread(entered.wait, 1))
            update = asyncio.create_task(agent._apply_monitor_state(state))
            await asyncio.sleep(0.02)
            # Repeated cancellation must still leave the worker joined.
            readers.cancel()
            await asyncio.sleep(0.01)
            check("monitor waits for a cancelled worker", not update.done())
            check("reset has not raced the old worker", agent.decision_loop.motion is old_motion)
            release.set()
            await asyncio.wait_for(update, 2)
            check("reset follows the worker's last mutation", agent.decision_loop.motion is not old_motion
                  and agent.decision_loop.motion_streak == 0 and agent.decision_loop.streak_count == 0
                  and not agent.decision_loop.streak_scores and agent.decision_loop.last_capture_at == 0)
            check("cancelled decision cannot upload", not agent._upload_motion_clip.called)
            check("restart follows joined reset", agent.start_capture.call_count == int(state["enabled"]))
        finally:
            release.set()
            if update:
                await asyncio.gather(update, return_exceptions=True)
            await cap._stop_readers()


async def test_resistant_child_is_killed_and_reaped() -> None:
    """A child ignoring terminate must exit before agent stop returns."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)",
        stdout=asyncio.subprocess.PIPE,
    )
    cap = make_capture(lambda frame: asyncio.sleep(0))
    cap.proc = proc
    agent = make_agent()
    agent.ffmpeg = cap
    try:
        await asyncio.wait_for(proc.stdout.readline(), 2)
        cap._supervisor = asyncio.create_task(cap._watch_exit(proc))
        with patch.object(ffmpeg_io, "STOP_TIMEOUT_SEC", 0.05), \
                patch.object(proc, "kill", wraps=proc.kill) as kill_child:
            await asyncio.wait_for(agent._stop_capture(), 2)
        check("resistant child was force-killed", kill_child.call_count == 1
              and proc.returncode is not None and proc.returncode != 0)
        check("old process and supervisor are finished", cap.proc is None and cap._supervisor.done())
        check("killed child was reaped", await proc.wait() == proc.returncode)
    finally:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()


async def test_cancelled_stop_still_reaps_child() -> None:
    """Shutdown cancellation cannot abandon a child waiting for forced kill."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)",
        stdout=asyncio.subprocess.PIPE,
    )
    cap = make_capture(lambda frame: asyncio.sleep(0))
    cap.proc = proc
    agent = make_agent()
    agent.ffmpeg = cap
    try:
        await asyncio.wait_for(proc.stdout.readline(), 2)
        cap._supervisor = asyncio.create_task(cap._watch_exit(proc))
        with patch.object(ffmpeg_io, "STOP_TIMEOUT_SEC", 0.05):
            stopping = asyncio.create_task(agent._stop_capture())
            await asyncio.sleep(0.01)
            stopping.cancel()
            result = await asyncio.wait_for(asyncio.gather(stopping, return_exceptions=True), 2)
        check("stop preserves cancellation after cleanup", isinstance(result[0], asyncio.CancelledError))
        check("cancelled stop still kills and reaps the child", proc.returncode is not None
              and cap._supervisor.done() and agent.ffmpeg is None)
    finally:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()


async def test_replacement_waits_for_child_exit() -> None:
    """Starting replacement capture must finish the old capture's shutdown."""
    agent = make_agent()
    old_capture = make_capture(agent._on_frame)
    agent.ffmpeg = old_capture
    released = asyncio.Event()

    async def stop():
        await asyncio.sleep(0.01)
        released.set()

    async def start():
        check("replacement starts after old capture was reaped and joined", released.is_set())

    old_capture.stop_and_wait = stop
    replacement = SimpleNamespace(start=AsyncMock(side_effect=start))
    with patch.object(capture, "FfmpegCapture", return_value=replacement), \
            patch.object(capture, "LiveFramePoller"):
        await agent.start_capture()
    check("agent retains the replacement capture", agent.ffmpeg is replacement)


async def test_failed_restart_allows_later_start() -> None:
    """An expected restart OS error must not prevent cleanup or a later retry."""
    agent = make_agent()
    cap = make_capture(agent._on_frame)
    agent.ffmpeg = cap
    proc = FakeProc()
    cap.proc = proc
    cap.start = AsyncMock(side_effect=PermissionError("segment directory not writable"))
    with patch.object(ffmpeg_io, "RESTART_DELAY_SEC", 0):
        cap._supervisor = asyncio.create_task(cap._watch_exit(proc))
        await asyncio.wait({cap._supervisor})
    check("auto-restart failed with the expected OS error",
          isinstance(cap._supervisor.exception(), PermissionError) and cap.proc is None)

    parked, cancelled = asyncio.Event(), asyncio.Event()

    async def reader():
        parked.set()
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()

    cap._readers = asyncio.gather(reader())
    await parked.wait()
    replacement = SimpleNamespace(start=AsyncMock())
    with patch.object(capture, "FfmpegCapture", return_value=replacement), \
            patch.object(capture, "LiveFramePoller"), patch("builtins.print") as log:
        await agent.start_capture()
    check("failed supervisor still joins remaining readers", cancelled.is_set() and cap._readers is None)
    check("expected restart failure is logged", any("ffmpeg restart failed" in call.args[0]
          for call in log.call_args_list))
    check("later capture start retries successfully", agent.ffmpeg is replacement
          and replacement.start.await_count == 1)


async def test_unexpected_supervisor_failure_still_cleans_readers() -> None:
    """Unrelated supervisor failures remain visible and still clean readers."""
    agent = make_agent()
    cap = make_capture(agent._on_frame)
    agent.ffmpeg = cap

    async def fail():
        raise ValueError("unexpected supervisor failure")

    cap._supervisor = asyncio.create_task(fail())
    await asyncio.wait({cap._supervisor})
    cap._stop_readers = AsyncMock()
    raised = False
    with patch.object(capture, "FfmpegCapture") as replacement:
        try:
            await agent.start_capture()
        except ValueError as err:
            raised = str(err) == "unexpected supervisor failure"
    check("unrelated supervisor failure is propagated", raised)
    check("reader cleanup runs when supervisor raises", cap._stop_readers.await_count == 1)
    check("failed cleanup cannot start a replacement", not replacement.called and agent.ffmpeg is cap)


async def test_main_closes_client_when_stop_fails() -> None:
    """A shutdown error must not skip closing the HTTP client."""
    agent = make_agent()
    agent.run = AsyncMock()
    agent._stop_capture = AsyncMock(side_effect=ValueError("shutdown failed"))
    agent.client.close = AsyncMock()
    raised = False
    with patch.object(capture, "Agent", return_value=agent), \
            patch.object(asyncio.get_running_loop(), "add_signal_handler"):
        try:
            await capture.main()
        except ValueError as err:
            raised = str(err) == "shutdown failed"
    check("main preserves shutdown failures", raised)
    check("main closes its HTTP client despite shutdown failure", agent.client.close.await_count == 1)


async def test_capture_creates_both_transient_directories() -> None:
    """FFmpeg receives existing segment and live-frame parent directories."""
    with tempfile.TemporaryDirectory(prefix="bunny-capture-") as tmp:
        cap = make_capture(lambda frame: asyncio.sleep(0))
        cap.seg_dir = Path(tmp) / "segments" / "nested"
        cap.live_frame_path = Path(tmp) / "live" / "nested" / "frame.jpg"

        async def spawn(*args, **kwargs):
            check("segment directory exists before spawning", cap.seg_dir.is_dir())
            check("live-frame parent exists before spawning", cap.live_frame_path.parent.is_dir())
            check("FFmpeg arguments use the configured paths", str(cap.live_frame_path) in args
                  and str(cap.seg_dir / "seg-%05d.webm") in args)
            raise OSError("test camera unavailable")

        with patch.object(asyncio, "create_subprocess_exec", side_effect=spawn):
            try:
                await cap.start()
            except OSError:
                pass


async def main_async() -> None:
    for fn in (
        test_slow_handler_does_not_stall_reader,
        test_reader_is_bound_to_its_own_stream,
        test_restart_cancels_readers_first,
        test_stale_proc_not_cleared_on_restart,
        test_handler_error_does_not_kill_consumer,
        test_blocking_inference_keeps_reader_and_timer_running,
        test_config_waits_for_serialized_decisions,
        test_cancelled_worker_finishes_before_monitor_reset,
        test_resistant_child_is_killed_and_reaped,
        test_cancelled_stop_still_reaps_child,
        test_replacement_waits_for_child_exit,
        test_failed_restart_allows_later_start,
        test_unexpected_supervisor_failure_still_cleans_readers,
        test_main_closes_client_when_stop_fails,
        test_capture_creates_both_transient_directories,
    ):
        print(f"\n{fn.__doc__.splitlines()[0]}")
        await fn()


def main() -> None:
    asyncio.run(main_async())
    finish("capture loop")


if __name__ == "__main__":
    main()
