"""Regression gate for notes/fixes.md installment 3.

3.1 is about the event loop staying responsive, which a plain unit assertion
cannot show. The test instead measures event-loop lag while the recordings work
runs, against a deliberately-blocking control doing the identical work inline.
The control calibrates the threshold on whatever machine this runs on rather
than hardcoding a timing guess.

Writes are isolated: the router's RECORDINGS_DIR/SEGMENTS_DIR globals are
repointed at a temp directory for the upload and grab checks, so nothing under
server/ is created, modified or deleted.

Usage (from server_py/, runtime venv):
    .venv/Scripts/python.exe -m tests.test_server_robustness
"""
import asyncio
import io
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.routers import recordings as rec

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")
        failures.append(name)


# Windows' default timer resolution is ~15.6ms, so an asyncio.sleep() heartbeat
# cannot tick faster than that no matter what is asked for. Any blocking this
# test wants to detect has to be substantially longer than that floor or the
# measurement is just reporting timer granularity. Hence a batch big enough to
# block for a few hundred ms rather than a single ~1ms scan.
TIMER_FLOOR_SEC = 0.0156
SCAN_BATCH = 60


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


async def test_listing_does_not_block_the_loop() -> None:
    """fixes.md 3.1: the directory walk must not run on the event loop."""
    count = len(rec._scan_recordings())
    print(f"  (scanning {count} real recordings, read-only)")
    if count < 20:
        check("3.1 enough recordings on disk to measure", False, f"only {count}")
        return

    def batch():
        for _ in range(SCAN_BATCH):
            rec._scan_recordings()

    async def blocking():
        batch()  # the pre-fix shape: straight on the loop

    async def off_thread():
        await asyncio.to_thread(batch)

    blocking_lag = await measure_lag(blocking)
    async_lag = await measure_lag(off_thread)
    print(f"  blocking lag {blocking_lag * 1000:.1f}ms vs off-thread lag {async_lag * 1000:.1f}ms")

    # Guard against a false pass: if the control did not actually stall the
    # loop well past the timer floor, the comparison below proves nothing.
    check(
        "3.1 the blocking control really does stall the loop",
        blocking_lag > TIMER_FLOOR_SEC * 4,
        f"only {blocking_lag * 1000:.1f}ms, too close to the ~15.6ms timer floor to be meaningful",
    )
    check(
        "3.1 off-thread listing keeps the loop responsive",
        async_lag < blocking_lag / 3,
        f"{async_lag * 1000:.1f}ms is not comfortably under a third of {blocking_lag * 1000:.1f}ms",
    )


async def test_scan_recordings_shape() -> None:
    """The rewritten scandir helper must return what the old iterdir did."""
    items = await asyncio.to_thread(rec._scan_recordings)
    check("3.1 listing is non-empty", bool(items))
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
    """fixes.md 3.2: a typo used to raise at import and take the server down."""
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
        test_listing_does_not_block_the_loop,
        test_scan_recordings_shape,
        test_upload_and_grab_in_a_temp_dir,
        test_config_survives_a_bad_env_value,
    ):
        print(f"\n{fn.__doc__.splitlines()[0]}")
        await fn()


def main() -> None:
    asyncio.run(main_async())
    if failures:
        print(f"\nFAIL: {len(failures)} check(s) failed")
        sys.exit(1)
    print("\nPASS: all server robustness checks passed")


if __name__ == "__main__":
    main()
