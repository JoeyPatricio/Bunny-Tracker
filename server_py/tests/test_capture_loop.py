"""Regression gate for notes/fixes.md installment 2 (agent/ffmpeg_io.py).

Drives FfmpegCapture's reader/consumer tasks directly against a real
asyncio.StreamReader instead of spawning ffmpeg, so this runs anywhere with no
camera and no ffmpeg binary. Same shape as test_motion_parity.py: plain main(),
exit 1 on failure.

Usage (from server_py/, runtime venv):
    .venv/Scripts/python.exe -m tests.test_capture_loop
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import ffmpeg_io
from agent.ffmpeg_io import FRAME_SIZE, FfmpegCapture

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")
        failures.append(name)


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
    """fixes.md 2.1: the reader must keep draining stdout while on_frame runs."""
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
    """fixes.md 2.2: the loop must not re-read self.proc each iteration."""
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
    """fixes.md 2.2: readers must be torn down before start() rebinds state."""
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


async def main_async() -> None:
    for fn in (
        test_slow_handler_does_not_stall_reader,
        test_reader_is_bound_to_its_own_stream,
        test_restart_cancels_readers_first,
        test_stale_proc_not_cleared_on_restart,
        test_handler_error_does_not_kill_consumer,
    ):
        print(f"\n{fn.__doc__.splitlines()[0]}")
        await fn()


def main() -> None:
    asyncio.run(main_async())
    if failures:
        print(f"\nFAIL: {len(failures)} check(s) failed")
        sys.exit(1)
    print("\nPASS: all capture loop checks passed")


if __name__ == "__main__":
    main()
