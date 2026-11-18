"""Server half of Clipping Mode: the upload budget and the monitor controls.

The agent half (decision loop, clip queue, segment selection) is tested in
agent/tests/test_decision.cpp and agent/tests/test_server.cpp.

Writes are isolated: routers/recordings.py's RECORDINGS_DIR/SEGMENTS_DIR globals
are repointed at a temp directory for the budget checks, exactly as
tests/test_server_robustness.py does, and monitor.json is never written.

Usage (from server/, runtime venv):
    .venv/bin/python tests/test_clipping_mode.py
"""
import asyncio
import io
import shutil
import sys
import tempfile
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.lib import clip_budget
from app.routers import monitor as mon
from app.routers import recordings as rec
from tests.checks import check, finish

# ── 1. Budget ───────────────────────────────────────────────────────────────

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
        labeled[first["filename"]] = "resting_lying"
        blocked_still = await upload(source="clipping")
        check("labeling one of three still leaves the backlog at the cap",
              getattr(blocked_still, "status_code", None) == 507, str(blocked_still))

        # Labeling a second one drops the backlog below the cap and harvesting
        # can resume — the point of counting unlabeled clips rather than all of
        # them: working the queue in Label Studio is what buys more room.
        labeled[manual["filename"]] = "resting_lying"
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


async def test_low_disk_also_stops_normal_motion_uploads() -> None:
    """Clipping's low-disk auto-stop must not hand the disk to normal motion.

    Once clipping stops, the agent is back in normal mode, where motion alone
    triggers uploads. Those are tagged "motion" and must hit the same disk
    floor, but not the backlog cap, and must resume once space is freed.
    """
    tmp = Path(tempfile.mkdtemp(prefix="bunnybudget-"))
    saved_dirs = (rec.RECORDINGS_DIR, rec.SEGMENTS_DIR)
    saved_state = rec.read_monitor_state
    saved_stop = rec.stop_clipping
    saved_labels = clip_budget.read_labels

    # 10 PB floor: unreachable on any real disk, so the floor is breached.
    state = {**mon.DEFAULTS, "clipping": True, "clippingMaxUnlabeled": 0,
             "clippingMinFreeGb": 10_000_000}
    stopped: list[str] = []

    async def fake_state():
        return dict(state)

    async def fake_stop(reason):
        stopped.append(reason)
        state.update(clipping=False, clippingStoppedReason=reason)

    async def no_labels():
        return {}

    def written() -> int:
        return len(list(rec.RECORDINGS_DIR.glob("*.webm")))

    try:
        rec.RECORDINGS_DIR = tmp / "recordings"
        rec.SEGMENTS_DIR = tmp / "segments"
        rec.RECORDINGS_DIR.mkdir()
        rec.SEGMENTS_DIR.mkdir()
        rec.read_monitor_state = fake_state
        rec.stop_clipping = fake_stop
        clip_budget.read_labels = no_labels

        async def upload(**kwargs):
            await asyncio.sleep(0.002)
            return await rec.upload_recording(FakeUpload(), **kwargs)

        clip = await upload(source="clipping")
        check("a low-disk clipping upload is refused with 507",
              getattr(clip, "status_code", None) == 507, str(clip))
        check("clipping auto-stopped on low disk",
              state["clipping"] is False and "disk" in (stopped[0] if stopped else ""), str(stopped))

        motion = await upload(source="motion")
        check("a motion upload after the auto-stop is refused with 507",
              getattr(motion, "status_code", None) == 507, str(motion))
        check("nothing was written below the floor", written() == 0)
        check("a motion refusal does not touch monitor state", len(stopped) == 1, str(stopped))

        alert = await upload()
        check("an untagged alert upload is still accepted below the floor",
              isinstance(alert, dict) and "filename" in alert, str(alert))

        # Space freed: motion resumes by itself, clipping stays off until re-armed.
        state["clippingMinFreeGb"] = 0
        motion = await upload(source="motion")
        check("motion uploads resume once the floor clears",
              isinstance(motion, dict) and motion["filename"].startswith("recording-2"), str(motion))
        check("clipping stays stopped until re-armed", state["clipping"] is False)

        # The backlog cap is clipping's alone: a full queue never blocks motion.
        state["clippingMaxUnlabeled"] = 1
        capped = await upload(source="clipping")
        check("the backlog cap still refuses clipping",
              getattr(capped, "status_code", None) == 507, str(capped))
        motion = await upload(source="motion")
        check("the backlog cap never refuses a motion upload",
              isinstance(motion, dict) and "filename" in motion, str(motion))
    finally:
        rec.RECORDINGS_DIR, rec.SEGMENTS_DIR = saved_dirs
        rec.read_monitor_state = saved_state
        rec.stop_clipping = saved_stop
        clip_budget.read_labels = saved_labels
        shutil.rmtree(tmp, ignore_errors=True)


async def test_unknown_upload_source_is_rejected() -> None:
    """A misspelled source tag must not bypass the disk floor as an untagged upload."""
    tmp = Path(tempfile.mkdtemp(prefix="bunnybudget-"))
    saved_dirs = (rec.RECORDINGS_DIR, rec.SEGMENTS_DIR)
    try:
        rec.RECORDINGS_DIR = tmp / "recordings"
        rec.SEGMENTS_DIR = tmp / "segments"
        for source in ("Motion", "clip"):
            res = await rec.upload_recording(FakeUpload(), source=source)
            check(f"source {source!r} is refused with 400",
                  getattr(res, "status_code", None) == 400, str(res))
        check("nothing was written for an unknown source", not rec.RECORDINGS_DIR.exists())
    finally:
        rec.RECORDINGS_DIR, rec.SEGMENTS_DIR = saved_dirs
        shutil.rmtree(tmp, ignore_errors=True)


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


# ── 2. Monitor state ────────────────────────────────────────────────────────

async def test_sensitivity_maps_to_a_threshold() -> None:
    """The 1-10 preset must land on the documented band, not a raw 0-100."""
    check("sensitivity 1 is the least sensitive",
          mon.threshold_for_sensitivity(1) == mon.SENSITIVITY_THRESHOLDS[0])
    check("sensitivity 10 is the most sensitive",
          mon.threshold_for_sensitivity(10) == mon.SENSITIVITY_THRESHOLDS[-1])
    check("higher sensitivity always means a lower threshold",
          all(a > b for a, b in pairwise(mon.SENSITIVITY_THRESHOLDS)))
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
        test_budget_stops_clipping_but_never_other_uploads,
        test_low_disk_also_stops_normal_motion_uploads,
        test_unknown_upload_source_is_rejected,
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
    finish("clipping mode")


if __name__ == "__main__":
    main()
