"""Regression gate for server robustness fixes.

3.1 is checked two ways. The structural check calls the real listing handler
and asserts the scan ran on a worker thread, never the event-loop thread; that
is the actual claim and it cannot be flaky. The timing check then measures
event-loop lag while the scan runs, against a deliberately-blocking control
doing the identical work inline. Its timer floor and batch size are measured on
whatever machine this runs on, and if even the capped batch cannot stall the
loop it skips rather than fails: a measurement that cannot be made is not a
regression.

Nothing reads or writes under server/: every check repoints the router's
RECORDINGS_DIR/SEGMENTS_DIR globals at a temp directory seeded with synthetic
files, so this runs on a fresh clone with no footage.

Usage (from server_py/, runtime venv):
    .venv/Scripts/python.exe -m tests.test_server_robustness
"""
import asyncio
import io
import math
import shutil
import statistics
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.routers import recordings as rec
from tests.checks import check, finish, skip

# Synthetic recordings, not the real archive. scandir's cost is per directory
# entry, not per byte, so zero-byte files measure the same as real clips; and a
# temp dir means this runs on a fresh clone or a CI box with no footage at all.
SYNTHETIC_RECORDINGS = 500

# How long the blocking control has to stall the loop, as a multiple of this
# machine's own timer floor, before the comparison means anything. And a cap on
# how many scans the control may run, so a very fast disk turns into a skip
# rather than a test that spins for a minute.
STALL_FLOORS = 8
MAX_SCAN_BATCH = 5000


@contextmanager
def seeded_recordings(count: int):
    """Point the router at a temp dir holding `count` empty .webm files plus
    one non-webm decoy, restoring the real path afterwards."""
    tmp = Path(tempfile.mkdtemp(prefix="bunnyscan-"))
    saved = rec.RECORDINGS_DIR
    try:
        for i in range(count):
            (tmp / f"recording-2026-01-01T00-{i // 60:02d}-{i % 60:02d}-{i:03d}Z.webm").touch()
        (tmp / "notes.txt").touch()
        rec.RECORDINGS_DIR = tmp
        yield tmp
    finally:
        rec.RECORDINGS_DIR = saved
        shutil.rmtree(tmp, ignore_errors=True)


async def measure_lag(work) -> float:
    """Max gap between heartbeat ticks while `work` runs concurrently."""
    gaps: list[float] = []
    done = asyncio.Event()
    loop = asyncio.get_running_loop()

    async def heartbeat():
        last = loop.time()
        while not done.is_set():
            await asyncio.sleep(0.005)
            now = loop.time()
            gaps.append(now - last)
            last = now

    hb = asyncio.ensure_future(heartbeat())
    await asyncio.sleep(0.05)  # let the heartbeat settle
    await work()
    done.set()
    await hb
    return max(gaps) if gaps else 0.0


async def timer_floor() -> float:
    """The heartbeat's own max gap with nothing else running. Windows' default
    timer resolution puts this near 15.6ms; Linux is nearer the 5ms sleep. Any
    stall this test wants to detect has to clear it by a wide margin or the
    measurement is just reporting timer granularity."""
    async def idle():
        await asyncio.sleep(0.2)
    return await measure_lag(idle)


async def test_listing_runs_off_the_loop_thread() -> None:
    """The directory walk must not run on the event-loop thread."""
    # The structural form of 3.1, and the one that cannot be flaky: call the
    # real handler and record which thread the scan ran on.
    loop_thread = threading.get_ident()
    seen: list[int] = []
    real_scan = rec._scan_recordings

    def spy():
        seen.append(threading.get_ident())
        return real_scan()

    with seeded_recordings(3), patch.object(rec, "_scan_recordings", spy), \
            patch.object(rec, "is_authed", lambda _req: True):
        result = await rec.list_recordings(request=None)

    check("3.1 handler scanned the recordings dir", len(seen) == 1, f"{len(seen)} scan call(s)")
    check("3.1 scan ran off the event-loop thread", loop_thread not in seen)
    check("3.1 handler returned the listing", isinstance(result, dict) and len(result.get("recordings", [])) == 3, str(result)[:200])


async def test_listing_does_not_block_the_loop() -> None:
    """And the loop stays responsive while it runs."""
    floor = await timer_floor()
    target = floor * STALL_FLOORS

    with seeded_recordings(SYNTHETIC_RECORDINGS):
        rec._scan_recordings()  # warm the dentry cache before timing anything
        samples = []
        for _ in range(5):
            t0 = time.perf_counter()
            rec._scan_recordings()
            samples.append(time.perf_counter() - t0)
        per_scan = statistics.median(samples)
        batch_size = min(MAX_SCAN_BATCH, max(1, math.ceil(target / per_scan)))
        print(
            f"  (timer floor {floor * 1000:.1f}ms, one scan of {SYNTHETIC_RECORDINGS} "
            f"files {per_scan * 1000:.2f}ms, batch {batch_size})"
        )

        def batch():
            for _ in range(batch_size):
                rec._scan_recordings()

        async def blocking():
            batch()  # the pre-fix shape: straight on the loop

        async def off_thread():
            await asyncio.to_thread(batch)

        blocking_lag = await measure_lag(blocking)
        async_lag = await measure_lag(off_thread)

    print(f"  blocking lag {blocking_lag * 1000:.1f}ms vs off-thread lag {async_lag * 1000:.1f}ms")

    # Guard against a false pass: if the control did not actually stall the
    # loop well past the timer floor, the comparison below proves nothing. That
    # is a measurement this machine cannot make, not a regression, so it skips.
    # The structural check above still covers 3.1 either way.
    if blocking_lag < floor * 4:
        skip(
            "3.1 off-thread listing keeps the loop responsive",
            f"blocking control only reached {blocking_lag * 1000:.1f}ms against a "
            f"{floor * 1000:.1f}ms timer floor, even at {batch_size} scans",
        )
        return
    check(
        "3.1 off-thread listing keeps the loop responsive",
        async_lag < blocking_lag / 3,
        f"{async_lag * 1000:.1f}ms is not comfortably under a third of {blocking_lag * 1000:.1f}ms",
    )


async def test_scan_recordings_shape() -> None:
    """The rewritten scandir helper must return what the old iterdir did."""
    with seeded_recordings(5):
        items = await asyncio.to_thread(rec._scan_recordings)
    check("3.1 every seeded clip listed", len(items) == 5, f"got {len(items)}")
    if not items:
        return
    first = items[0]
    check("3.1 keys unchanged", set(first) == {"filename", "createdAt", "size"}, f"got {sorted(first)}")
    check("3.1 only webm listed", all(i["filename"].endswith(".webm") for i in items))
    check("3.1 sizes are real", all(isinstance(i["size"], int) and i["size"] >= 0 for i in items))
    check("3.1 createdAt is an ISO millis string", first["createdAt"].endswith("Z"), first["createdAt"])


async def test_upload_and_grab_in_a_temp_dir() -> None:
    """Round-trip upload and grab with the module globals repointed."""
    tmp = Path(tempfile.mkdtemp(prefix="bunnyrec-"))
    saved = (rec.RECORDINGS_DIR, rec.SEGMENTS_DIR)
    try:
        rec.RECORDINGS_DIR = tmp / "recordings"
        rec.SEGMENTS_DIR = tmp / "segments"
        rec.RECORDINGS_DIR.mkdir()
        rec.SEGMENTS_DIR.mkdir()

        payload = b"\x1a\x45\xdf\xa3fake webm" * 500

        class FakeUpload:
            content_type = "video/webm"
            file = io.BytesIO(payload)

        result = await rec.upload_recording(FakeUpload())
        written = rec.RECORDINGS_DIR / result["filename"]
        check("3.1 upload wrote the file", written.is_file())
        check("3.1 upload wrote every byte", written.read_bytes() == payload)
        check("3.1 upload reported the real size", result["size"] == len(payload), f"{result['size']} vs {len(payload)}")

        # Two segments: grab must take the second-newest, since ffmpeg is still
        # writing the newest.
        (rec.SEGMENTS_DIR / "seg-00000.webm").write_bytes(b"older")
        time.sleep(0.02)
        (rec.SEGMENTS_DIR / "seg-00001.webm").write_bytes(b"newest, still being written")
        grabbed = await rec.grab_segment()
        check("3.1 grab returned a filename", isinstance(grabbed, dict) and "filename" in grabbed, str(grabbed))
        if isinstance(grabbed, dict) and "filename" in grabbed:
            body = (rec.RECORDINGS_DIR / grabbed["filename"]).read_bytes()
            check("3.1 grab took the second-newest segment", body == b"older", f"got {body!r}")
    finally:
        rec.RECORDINGS_DIR, rec.SEGMENTS_DIR = saved
        shutil.rmtree(tmp, ignore_errors=True)

    check("3.1 module globals restored", rec.RECORDINGS_DIR == saved[0] and rec.SEGMENTS_DIR == saved[1])


async def test_config_survives_a_bad_env_value() -> None:
    """A typo used to raise at import and take the server down."""
    cases = [
        ("not-a-number", "unparseable"),
        ("", "empty"),
        ("10 minutes", "trailing text"),
        ("nan", "NaN"),
        ("inf", "infinity"),
    ]
    for raw, label in cases:
        try:
            value = config._numberish(raw, name="NOTIFY_COOLDOWN_MINUTES", default=10)
            check(f"3.2 {label} falls back to the default", value == 10, f"got {value!r}")
        except Exception as err:
            check(f"3.2 {label} falls back to the default", False, f"raised {err!r}")

    # Good values must still behave exactly as before.
    check("3.2 whole numbers stay int", config._numberish("15", name="X", default=10) == 15)
    check("3.2 int type preserved", isinstance(config._numberish("15", name="X", default=10), int))
    check("3.2 fractional values stay float", config._numberish("2.5", name="X", default=10) == 2.5)
    check("3.2 float-looking whole numbers become int", config._numberish("10.0", name="X", default=10) == 10)
    check("3.2 the live value is usable arithmetic", isinstance(config.NOTIFY_COOLDOWN_MINUTES * 60, (int, float)))


async def main_async() -> None:
    for fn in (
        test_listing_runs_off_the_loop_thread,
        test_listing_does_not_block_the_loop,
        test_scan_recordings_shape,
        test_upload_and_grab_in_a_temp_dir,
        test_config_survives_a_bad_env_value,
    ):
        print(f"\n{fn.__doc__.splitlines()[0]}")
        await fn()


def main() -> None:
    asyncio.run(main_async())
    finish("server robustness")


if __name__ == "__main__":
    main()
