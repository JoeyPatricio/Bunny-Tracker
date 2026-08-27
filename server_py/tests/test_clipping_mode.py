"""Gate for Clipping Mode — the bulk unlabeled-clip harvester.

Runs with no camera, no ffmpeg and no ONNX model, which is the point: the whole
feature exists so clips can be harvested when there is no usable model, so the
test must prove the loop works with none loaded.

Writes are isolated: routers/recordings.py's RECORDINGS_DIR/SEGMENTS_DIR globals
are repointed at a temp directory for the budget checks, exactly as
tests/test_server_robustness.py does, and monitor.json is never written.

Usage (from server_py/, runtime venv):
    .venv/Scripts/python.exe -m tests.test_clipping_mode
"""
import asyncio
import io
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.capture import Agent, AgentDecisionLoop
from app.lib import clip_budget
from app.routers import monitor as mon
from app.routers import recordings as rec

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")
        failures.append(name)


def fixture_frames() -> np.ndarray:
    return np.load(FIXTURES_DIR / "motion_fixture_frames.npy")


def replay(loop: AgentDecisionLoop, frames, *, step: float) -> list[dict]:
    """Feed frames at a fixed cadence, returning every decision."""
    return [loop.handle_frame(frame, now=i * step) for i, frame in enumerate(frames)]


class ExplodingBackbone:
    """Any call means inference ran when it should not have."""

    def embed(self, pixel_values):
        raise AssertionError("the model was invoked during Clipping Mode")


# ── 1. Model-free ───────────────────────────────────────────────────────────

async def test_runs_with_no_model() -> None:
    """Clipping Mode must run with no ONNX artifacts loaded at all."""
    frames = fixture_frames()
    loop = AgentDecisionLoop(None, None, clip_threshold=2.5, clip_streak=1, clip_cooldown_sec=8)
    loop.apply_clipping_config(clipping=True)

    try:
        decisions = replay(loop, frames, step=0.5)
        ok = True
    except Exception as err:
        decisions, ok = [], False
        check("no-model loop never raises", False, repr(err))
    if ok:
        check("no-model loop never raises", True)

    check("predictor is not built without a model", loop.predictor is None)
    check("never warm without a model", all(not d["warm"] for d in decisions))
    check("no behavior alert is ever emitted", all(not d["behaviorAlert"] for d in decisions))
    check("no action is ever 'alert'", all(d["action"] != "alert" for d in decisions))
    check("nothing is published to the prediction feed",
          all(not d["publishPrediction"] for d in decisions))
    check("some clips are captured", any(d["action"] == "clip" for d in decisions))


async def test_model_is_not_invoked_while_clipping() -> None:
    """With a model loaded, Clipping Mode must still not run inference."""
    frames = fixture_frames()[:12]
    loop = AgentDecisionLoop(ExplodingBackbone(), object(), clip_streak=1)
    loop.apply_clipping_config(clipping=True)
    try:
        replay(loop, frames, step=0.5)
        check("model is never invoked while clipping", True)
    except AssertionError as err:
        check("model is never invoked while clipping", False, str(err))


# ── 2. Yield, threshold and cooldown ────────────────────────────────────────

async def test_clipping_harvests_more_than_the_alert_path() -> None:
    """The whole point: many more clips than the motion-capture side path."""
    frames = fixture_frames()

    baseline = AgentDecisionLoop(None, None)
    default_events = sum(1 for d in replay(baseline, frames, step=1.5)
                         if d["action"] == "motion_capture")

    harvest = AgentDecisionLoop(None, None, clip_threshold=2.5, clip_streak=1, clip_cooldown_sec=8)
    harvest.apply_clipping_config(clipping=True)
    clip_events = sum(1 for d in replay(harvest, frames, step=0.5) if d["action"] == "clip")

    check("clipping harvests strictly more than the default path",
          clip_events > default_events, f"{clip_events} clips vs {default_events} motion captures")


async def test_sensitivity_changes_the_yield() -> None:
    """A lower threshold must capture at least as much, and generally more."""
    frames = fixture_frames()

    def yield_at(threshold: float) -> int:
        loop = AgentDecisionLoop(None, None, clip_threshold=threshold, clip_streak=1,
                                 clip_cooldown_sec=0)
        loop.apply_clipping_config(clipping=True)
        return sum(1 for d in replay(loop, frames, step=0.5) if d["action"] == "clip")

    insensitive, sensitive = yield_at(8.0), yield_at(0.4)
    check("a lower threshold captures more", sensitive > insensitive,
          f"{sensitive} at 0.4 vs {insensitive} at 8.0")


async def test_cooldown_is_respected() -> None:
    """No two clips may land closer together than the cooldown."""
    cooldown = 8.0
    loop = AgentDecisionLoop(None, None, clip_threshold=0.4, clip_streak=1,
                             clip_cooldown_sec=cooldown)
    loop.apply_clipping_config(clipping=True)

    times = [i * 0.5 for i, d in enumerate(replay(loop, fixture_frames(), step=0.5))
             if d["action"] == "clip"]
    gaps = [b - a for a, b in zip(times, times[1:])]
    check("clips are captured at all", len(times) >= 2, f"only {len(times)}")
    check("no clip lands inside the cooldown", all(g >= cooldown for g in gaps),
          f"gaps {gaps}")


async def test_live_retuning_needs_no_restart() -> None:
    """apply_clipping_config must take effect on the very next frame."""
    frames = fixture_frames()
    loop = AgentDecisionLoop(None, None, clip_threshold=8.0, clip_streak=1, clip_cooldown_sec=0)
    loop.apply_clipping_config(clipping=True)

    half = len(frames) // 2
    before = sum(1 for d in replay(loop, frames[:half], step=0.5) if d["action"] == "clip")

    changed = loop.apply_clipping_config(clipping=True, threshold=0.4, cooldown=0)
    check("retuning without a mode change reports no restart needed", changed is False)
    check("the new threshold is live", loop.clip_threshold == 0.4)

    after = sum(1 for d in replay(loop, frames[half:], step=0.5) if d["action"] == "clip")
    check("retuning changes the capture rate mid-run", after > before,
          f"{after} after vs {before} before")


async def test_mode_transition_resets_state() -> None:
    """Flipping the mode must report a restart and clear stale streaks."""
    loop = AgentDecisionLoop(None, None)
    loop.motion_streak = 3
    loop.last_label = "zoomies"

    check("turning clipping on reports a restart is needed",
          loop.apply_clipping_config(clipping=True) is True)
    check("motion streak is cleared across the transition", loop.motion_streak == 0)
    check("stale label is cleared across the transition", loop.last_label is None)
    check("turning clipping off reports a restart is needed",
          loop.apply_clipping_config(clipping=False) is True)


# ── 3. The clip lands on the segment the motion happened in ─────────────────

def _fake_agent(seg_dir: Path):
    """An Agent with only the fields the clip queue touches. Built without
    __init__ so it needs no AGENT_TOKEN, no server and no camera."""
    agent = Agent.__new__(Agent)
    agent.seg_dir = seg_dir
    agent.pending_clips = []
    agent.clipped_segments = set()
    agent.clip_segment_secs = 8
    agent.clip_budget_blocked = False
    agent.uploaded = []

    async def fake_upload(seg_name=None, source=None):
        agent.uploaded.append((seg_name, source))
        return f"recording-clip-{seg_name}"

    agent._upload_segment = fake_upload
    return agent


def _write_segment(seg_dir: Path, name: str, mtime: float) -> None:
    path = seg_dir / name
    path.write_bytes(b"fake webm " + name.encode())
    import os
    os.utime(path, (mtime, mtime))


async def test_clip_targets_the_in_progress_segment() -> None:
    """The clip must be the segment being written, not the one before it."""
    tmp = Path(tempfile.mkdtemp(prefix="bunnyseg-"))
    try:
        now = time.time()
        _write_segment(tmp, "seg-00000.webm", now - 20)
        _write_segment(tmp, "seg-00001.webm", now - 2)   # ffmpeg is writing this
        agent = _fake_agent(tmp)

        await agent._queue_clip(4.2)
        check("the in-progress segment is queued",
              [c["name"] for c in agent.pending_clips] == ["seg-00001.webm"],
              str(agent.pending_clips))

        # Still the newest file, so still being written — nothing to upload yet.
        await agent._drain_pending_clips_once()
        check("nothing uploads while the segment is still open", agent.uploaded == [],
              str(agent.uploaded))
        check("the clip stays queued", len(agent.pending_clips) == 1)

        # A newer segment appears: the target is closed.
        _write_segment(tmp, "seg-00002.webm", now)
        await agent._drain_pending_clips_once()
        check("the event's own segment is uploaded once closed",
              agent.uploaded == [("seg-00001.webm", "clipping")], str(agent.uploaded))
        check("the queue is drained", agent.pending_clips == [])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def test_one_clip_per_segment() -> None:
    """Two events inside one segment must produce a single clip."""
    tmp = Path(tempfile.mkdtemp(prefix="bunnyseg-"))
    try:
        now = time.time()
        _write_segment(tmp, "seg-00000.webm", now - 20)
        _write_segment(tmp, "seg-00001.webm", now - 2)
        agent = _fake_agent(tmp)

        await agent._queue_clip(4.2)
        await agent._queue_clip(5.1)
        check("a second event in the same segment does not queue twice",
              len(agent.pending_clips) == 1, str(agent.pending_clips))

        _write_segment(tmp, "seg-00002.webm", now)
        await agent._drain_pending_clips_once()

        # A later event landing back in the segment that was already harvested
        # must not produce a second copy of the same footage.
        (tmp / "seg-00002.webm").unlink()
        await agent._queue_clip(6.0)
        check("an already-clipped segment is never re-queued",
              agent.pending_clips == [], str(agent.pending_clips))
        check("exactly one upload", len(agent.uploaded) == 1, str(agent.uploaded))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def test_moving_mtime_does_not_duplicate_a_clip() -> None:
    """A segment being written has a MOVING mtime.

    Regression: the pending queue keyed on (name, mtime), so two events a few
    seconds apart inside one segment carried different keys, both queued, and
    the same footage was uploaded twice. Caught by a live capture run, not by a
    fixture with frozen timestamps.
    """
    tmp = Path(tempfile.mkdtemp(prefix="bunnyseg-"))
    try:
        now = time.time()
        _write_segment(tmp, "seg-00000.webm", now - 6)
        agent = _fake_agent(tmp)

        await agent._queue_clip(9.2)
        # ffmpeg writes more of the same segment; its mtime advances.
        _write_segment(tmp, "seg-00000.webm", now - 2)
        await agent._queue_clip(9.0)
        check("a moving mtime does not queue the same segment twice",
              len(agent.pending_clips) == 1, str(agent.pending_clips))

        _write_segment(tmp, "seg-00001.webm", now)
        await agent._drain_pending_clips_once()
        check("the segment is uploaded exactly once",
              agent.uploaded == [("seg-00000.webm", "clipping")], str(agent.uploaded))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def test_stalled_capture_settles_and_does_not_duplicate() -> None:
    """When capture stops, no newer segment ever appears.

    The queued clip must still be uploaded once the file stops changing (it is
    finalized, not half-written), and a later event landing on that same
    settled segment must not upload it a second time. The dedupe key has to be
    the SETTLED mtime for that guard to fire at all.
    """
    tmp = Path(tempfile.mkdtemp(prefix="bunnyseg-"))
    try:
        agent = _fake_agent(tmp)
        # Old enough to count as settled (clip_segment_secs * 1.5 = 12s).
        stale = time.time() - 30
        _write_segment(tmp, "seg-00000.webm", stale)

        await agent._queue_clip(5.0)
        await agent._drain_pending_clips_once()
        check("a settled segment is uploaded even with no newer file",
              agent.uploaded == [("seg-00000.webm", "clipping")], str(agent.uploaded))

        await agent._queue_clip(5.5)
        await agent._drain_pending_clips_once()
        check("a settled segment is never harvested twice",
              len(agent.uploaded) == 1, str(agent.uploaded))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def test_open_segment_is_never_uploaded_early() -> None:
    """A file ffmpeg is actively writing must never be uploaded, however long
    the clip has been queued. The close check is a property of the FILE, not of
    how long we have been waiting."""
    tmp = Path(tempfile.mkdtemp(prefix="bunnyseg-"))
    try:
        agent = _fake_agent(tmp)
        _write_segment(tmp, "seg-00000.webm", time.time())
        await agent._queue_clip(5.0)

        for _ in range(3):
            # Simulate ffmpeg still writing: the mtime keeps advancing.
            _write_segment(tmp, "seg-00000.webm", time.time())
            await agent._drain_pending_clips_once()

        check("an actively-written segment is never uploaded", agent.uploaded == [],
              str(agent.uploaded))
        check("the clip is still queued", len(agent.pending_clips) == 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def test_ffmpeg_restart_does_not_collide() -> None:
    """ffmpeg renumbers from seg-00000 on restart, so names repeat. The dedupe
    key carries the mtime for exactly this reason."""
    tmp = Path(tempfile.mkdtemp(prefix="bunnyseg-"))
    try:
        now = time.time()
        _write_segment(tmp, "seg-00000.webm", now - 30)
        _write_segment(tmp, "seg-00001.webm", now - 22)
        agent = _fake_agent(tmp)
        await agent._queue_clip(4.0)
        _write_segment(tmp, "seg-00002.webm", now - 20)
        await agent._drain_pending_clips_once()
        check("first run uploaded", len(agent.uploaded) == 1, str(agent.uploaded))

        # ffmpeg restarts: numbering goes back to 0 with fresh mtimes.
        for name in ("seg-00000.webm", "seg-00001.webm", "seg-00002.webm"):
            (tmp / name).unlink()
        _write_segment(tmp, "seg-00000.webm", now - 2)
        _write_segment(tmp, "seg-00001.webm", now - 1)
        await agent._queue_clip(4.0)
        check("a same-named segment from a new ffmpeg run still queues",
              len(agent.pending_clips) == 1, str(agent.pending_clips))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def test_vanished_segment_is_dropped() -> None:
    """A pruned segment must not wedge the queue."""
    tmp = Path(tempfile.mkdtemp(prefix="bunnyseg-"))
    try:
        now = time.time()
        _write_segment(tmp, "seg-00000.webm", now - 20)
        _write_segment(tmp, "seg-00001.webm", now - 2)
        agent = _fake_agent(tmp)
        await agent._queue_clip(4.0)
        (tmp / "seg-00001.webm").unlink()
        await agent._drain_pending_clips_once()
        check("a vanished segment is dropped from the queue", agent.pending_clips == [])
        check("nothing was uploaded for it", agent.uploaded == [], str(agent.uploaded))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── 4. Budget ───────────────────────────────────────────────────────────────

class FakeUpload:
    content_type = "video/webm"

    def __init__(self, payload: bytes = b"\x1a\x45\xdf\xa3fake webm"):
        self.file = io.BytesIO(payload)


async def test_budget_stops_clipping_but_never_other_uploads() -> None:
    """The cap counts unlabeled clips only, and only refuses harvest uploads."""
    tmp = Path(tempfile.mkdtemp(prefix="bunnybudget-"))
    saved_dirs = (rec.RECORDINGS_DIR, rec.SEGMENTS_DIR)
    saved_state = rec.read_monitor_state
    saved_stop = rec.stop_clipping
    saved_labels = clip_budget.read_labels

    labeled: dict[str, str] = {}
    stopped: list[str] = []

    async def fake_state():
        return {**mon.DEFAULTS, "clipping": True, "clippingMaxUnlabeled": 2, "clippingMinFreeGb": 0}

    async def fake_stop(reason):
        stopped.append(reason)

    async def fake_labels():
        return dict(labeled)

    try:
        rec.RECORDINGS_DIR = tmp / "recordings"
        rec.SEGMENTS_DIR = tmp / "segments"
        rec.RECORDINGS_DIR.mkdir()
        rec.SEGMENTS_DIR.mkdir()
        rec.read_monitor_state = fake_state
        rec.stop_clipping = fake_stop
        clip_budget.read_labels = fake_labels

        # filename_timestamp() has millisecond resolution, so uploads issued
        # in the same millisecond would share a name and overwrite each other.
        # Real clips are a cooldown apart; space these out so the count is real.
        async def upload(**kwargs):
            await asyncio.sleep(0.002)
            return await rec.upload_recording(FakeUpload(), **kwargs)

        first = await upload(source="clipping")
        second = await upload(source="clipping")
        check("harvest uploads use the recording-clip- prefix",
              first["filename"].startswith("recording-clip-"), first["filename"])
        check("clips under the cap are accepted", "filename" in second)

        third = await upload(source="clipping")
        check("the clip over the cap is refused with 507",
              getattr(third, "status_code", None) == 507, str(third))
        check("nothing was written for the refused clip",
              len(list(rec.RECORDINGS_DIR.glob("*.webm"))) == 2)
        check("the server stopped clipping itself", len(stopped) == 1, str(stopped))
        check("the stop reason names the cap", "cap" in (stopped[0] or ""), str(stopped))

        # A non-harvest upload is never refused, however full the queue is.
        manual = await upload()
        check("a normal upload is never refused by the budget", "filename" in manual, str(manual))
        check("a normal upload keeps the plain prefix",
              manual.get("filename", "").startswith("recording-2"), str(manual))

        # The cap counts the whole unlabeled backlog, manual clips included:
        # three clips are now on disk and none are labeled, so labeling just one
        # still leaves the queue at the cap.
        labeled[first["filename"]] = "normal"
        blocked_still = await upload(source="clipping")
        check("labeling one of three still leaves the backlog at the cap",
              getattr(blocked_still, "status_code", None) == 507, str(blocked_still))

        # Labeling a second one drops the backlog below the cap and harvesting
        # can resume — the point of counting unlabeled clips rather than all of
        # them: working the queue in Label Studio is what buys more room.
        labeled[manual["filename"]] = "normal"
        fourth = await upload(source="clipping")
        check("working the labeling queue frees room to harvest again",
              isinstance(fourth, dict) and "filename" in fourth, str(fourth))
    finally:
        rec.RECORDINGS_DIR, rec.SEGMENTS_DIR = saved_dirs
        rec.read_monitor_state = saved_state
        rec.stop_clipping = saved_stop
        clip_budget.read_labels = saved_labels
        shutil.rmtree(tmp, ignore_errors=True)

    check("module globals restored", rec.RECORDINGS_DIR == saved_dirs[0])


async def test_disk_floor_blocks() -> None:
    """A free-disk floor above the real free space must block."""
    tmp = Path(tempfile.mkdtemp(prefix="bunnybudget-"))
    try:
        usage = await clip_budget.recordings_usage(
            recordings_dir=tmp, max_unlabeled=0, min_free_gb=10_000_000, labeled=set())
        check("an unreachable disk floor blocks", usage["blocked"] is True, str(usage))
        check("the disk reason is reported", "disk" in (usage["reason"] or ""), str(usage))

        usage = await clip_budget.recordings_usage(
            recordings_dir=tmp, max_unlabeled=0, min_free_gb=0, labeled=set())
        check("both limits disabled never blocks", usage["blocked"] is False, str(usage))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def test_unreadable_labels_do_not_stop_the_harvest() -> None:
    """A corrupt labels.json must not read as "nothing is labeled".

    JsonStore.read() raises on a corrupt or mid-write file by design. Counting
    the whole archive as unlabeled would blow through the cap, 507 the next
    clip and auto-stop harvesting with a false "cap reached" banner — killed by
    a transient read error. The disk floor still applies.
    """
    tmp = Path(tempfile.mkdtemp(prefix="bunnybudget-"))
    saved = clip_budget.read_labels
    try:
        for i in range(5):
            (tmp / f"recording-clip-{i}.webm").write_bytes(b"x")

        async def broken_labels():
            raise ValueError("labels.json is corrupt, refusing to overwrite")

        clip_budget.read_labels = broken_labels
        usage = await clip_budget.recordings_usage(
            recordings_dir=tmp, max_unlabeled=2, min_free_gb=0)
        check("a corrupt labels.json does not block the harvest",
              usage["blocked"] is False, str(usage))
        check("the unreadable state is reported", usage["labelsReadable"] is False, str(usage))

        # The disk floor is an independent measurement and must still bite.
        usage = await clip_budget.recordings_usage(
            recordings_dir=tmp, max_unlabeled=2, min_free_gb=10_000_000)
        check("the disk floor still applies when labels are unreadable",
              usage["blocked"] is True and "disk" in (usage["reason"] or ""), str(usage))
    finally:
        clip_budget.read_labels = saved
        shutil.rmtree(tmp, ignore_errors=True)


async def test_missing_recordings_dir_is_not_fatal() -> None:
    """The wiped-data case: nothing here may raise on a missing directory."""
    usage = await clip_budget.recordings_usage(
        recordings_dir=Path("/nonexistent/bunny/recordings"),
        max_unlabeled=500, min_free_gb=0, labeled=set())
    check("a missing recordings dir reads as empty, not an error",
          usage["count"] == 0 and usage["unlabeled"] == 0, str(usage))
    check("an empty archive does not block", usage["blocked"] is False, str(usage))


# ── 5. Monitor state ────────────────────────────────────────────────────────

async def test_sensitivity_maps_to_a_threshold() -> None:
    """The 1-10 preset must land on the documented band, not a raw 0-100."""
    check("sensitivity 1 is the least sensitive",
          mon.threshold_for_sensitivity(1) == mon.SENSITIVITY_THRESHOLDS[0])
    check("sensitivity 10 is the most sensitive",
          mon.threshold_for_sensitivity(10) == mon.SENSITIVITY_THRESHOLDS[-1])
    check("higher sensitivity always means a lower threshold",
          all(a > b for a, b in zip(mon.SENSITIVITY_THRESHOLDS, mon.SENSITIVITY_THRESHOLDS[1:])))
    check("out-of-range values clamp rather than raise",
          mon.threshold_for_sensitivity(0) == mon.SENSITIVITY_THRESHOLDS[0]
          and mon.threshold_for_sensitivity(99) == mon.SENSITIVITY_THRESHOLDS[-1])
    check("every preset is inside the useful band",
          all(0 < t <= 10 for t in mon.SENSITIVITY_THRESHOLDS))


async def test_public_monitor_hides_the_harvest_config() -> None:
    """GET /api/monitor is public for the demo page; clipping config is not."""
    class FakeRequest:
        headers: dict = {}
        cookies: dict = {}

    saved = mon.is_authed
    try:
        mon.is_authed = lambda request: False
        public = await mon.get_monitor(FakeRequest())
        check("public callers see only the two public fields",
              set(public) == set(mon.PUBLIC_FIELDS), str(sorted(public)))
        check("clipping state is not public", "clipping" not in public)
        check("the stop reason is not public", "clippingStoppedReason" not in public)

        mon.is_authed = lambda request: True
        private = await mon.get_monitor(FakeRequest())
        check("authed callers see the clipping config",
              "clipping" in private and "clippingMotionThreshold" in private,
              str(sorted(private)))
    finally:
        mon.is_authed = saved


async def main_async() -> None:
    for fn in (
        test_runs_with_no_model,
        test_model_is_not_invoked_while_clipping,
        test_clipping_harvests_more_than_the_alert_path,
        test_sensitivity_changes_the_yield,
        test_cooldown_is_respected,
        test_live_retuning_needs_no_restart,
        test_mode_transition_resets_state,
        test_clip_targets_the_in_progress_segment,
        test_one_clip_per_segment,
        test_moving_mtime_does_not_duplicate_a_clip,
        test_stalled_capture_settles_and_does_not_duplicate,
        test_open_segment_is_never_uploaded_early,
        test_ffmpeg_restart_does_not_collide,
        test_vanished_segment_is_dropped,
        test_budget_stops_clipping_but_never_other_uploads,
        test_disk_floor_blocks,
        test_unreadable_labels_do_not_stop_the_harvest,
        test_missing_recordings_dir_is_not_fatal,
        test_sensitivity_maps_to_a_threshold,
        test_public_monitor_hides_the_harvest_config,
    ):
        print(f"\n{fn.__doc__.splitlines()[0]}")
        await fn()


def main() -> None:
    asyncio.run(main_async())
    if failures:
        print(f"\nFAIL: {len(failures)} check(s) failed")
        sys.exit(1)
    print("\nPASS: all clipping mode checks passed")


if __name__ == "__main__":
    main()
