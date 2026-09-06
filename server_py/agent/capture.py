"""Port of handleFrame() (Phase 3) plus the boot/runtime wiring around it
(Phase 5): mkdir segments/, verify AGENT_TOKEN, load ONNX models, check
/api/monitor, startCapture() if enabled, pollMonitor every 5s, pruneSegments
every 60s. Same log line shapes as capture.mjs (a diagnostic-parity contract
with the live system, useful when comparing agent logs side by side).
"""

import asyncio
import os
import time
from pathlib import Path

import httpx
import imageio_ffmpeg
import numpy as np
from dotenv import load_dotenv

from agent.client import AgentClient
from agent.ffmpeg_io import FfmpegCapture, LiveFramePoller
from inference.backbone import Backbone
from inference.predictor import Predictor
from inference.temporal_head import TemporalHead
from shared.frames import normalize
from shared.labels import LABELS
from shared.motion import MotionDetector

# Defaults match capture.mjs's env-var defaults.
EMBED_WINDOW = 8
INTEREST_THRESH = 60
MOTION_FLOOR = 1.0
ALERT_STREAK = 3
MOTION_CAPTURE = 2.5
MOTION_STREAK = 3
CAPTURE_COOLDOWN_SEC = 30

# ── Clipping Mode ───────────────────────────────────────────────────────────
# Bulk harvesting of unlabeled clips for hand-labeling, not alerting. The model
# is off (see AgentDecisionLoop.__init__), so a much denser frame cadence is
# free — motion_level() is a numpy pool-and-diff over a 56x56 grid — and a
# 1-frame streak can catch a binky, which the 3-frame/1.5s alerting cadence
# cannot see at all.
CLIPPING_MOTION_THRESHOLD = 2.5  # overridden live by monitor.json
CLIPPING_STREAK = 1  # fire on the first frame over the threshold
CLIPPING_COOLDOWN_SEC = 8  # == segment length: at most one clip per segment

# The resting baseline. Every other class is, by definition, worth a look —
# which is why the alert decision is binary and `interest_score` below scores
# it directly instead of routing it through argmax (see notes below).
NORMAL_LABEL = "normal"
NORMAL_INDEX = LABELS.index(NORMAL_LABEL)


def interest_score(probs: np.ndarray) -> float:
    """Score one softmax window 0-100: how much is this worth an email?

    Called once per classified frame; the returned score is compared against
    AGENT_INTEREST_THRESHOLD to decide whether the frame is an alert candidate.
    `probs` is the full class distribution in LABELS order, summing to 1;
    probs[NORMAL_INDEX] is the resting baseline.

    Why this is not argmax: the head can be genuinely torn between grooming and
    yawning while still being confident it is not resting. Argmax throws that
    agreement away and the old gate then read the frame as boring. On the
    deployed model's own val set that cost 10 of 26 real events.
    """
    # Seed with 0.0, not probs[NORMAL_INDEX]: seeding with the value being
    # excluded turns "ignore normal" into "must beat normal", which makes this
    # max(probs) and scores a confidently-resting window ~85.
    res = 0.0
    for i, p in enumerate(probs):
        if i == NORMAL_INDEX:
            continue
        if p > res:
            res = p
    return res * 100.0  # 0-100% confidence in the most likely non-normal class


# What _classify returns when there is no model to run. warm stays False, so
# `candidate` and therefore behavior_alert are structurally unreachable — that
# is what guarantees no email, webhook or label suggestion can fire while
# harvesting, rather than a flag someone can forget to check.
NO_PREDICTION = {
    "label": None,
    "confidence": 0,
    "windowFill": 0,
    # interest 0 keeps a model-free frame below every possible threshold, so
    # the alert path stays unreachable on this dict alone, without `warm`.
    "interest": 0.0,
    "alertLabel": None,
    "alertConfidence": 0,
}


class AgentDecisionLoop:
    def __init__(
        self,
        backbone=None,
        head=None,
        *,
        embed_window: int = EMBED_WINDOW,
        interest_thresh: float = INTEREST_THRESH,
        motion_floor: float = MOTION_FLOOR,
        alert_streak: int = ALERT_STREAK,
        motion_capture: float = MOTION_CAPTURE,
        motion_streak: int = MOTION_STREAK,
        capture_cooldown_sec: float = CAPTURE_COOLDOWN_SEC,
        clip_threshold: float = CLIPPING_MOTION_THRESHOLD,
        clip_streak: int = CLIPPING_STREAK,
        clip_cooldown_sec: float = CLIPPING_COOLDOWN_SEC,
        clip_run_model: bool = False,
    ):
        # backbone/head are optional: without them there is no predictor, and
        # the loop runs motion-only. That is both Clipping Mode's normal state
        # and the degraded mode the agent falls back to when the ONNX artifacts
        # are missing, so one code path covers both.
        self.predictor = (
            Predictor(backbone, head, embed_window) if (backbone and head) else None
        )
        self.motion = MotionDetector()
        self.embed_window = embed_window
        self.interest_thresh = interest_thresh
        self.motion_floor = motion_floor
        self.alert_streak = alert_streak
        self.motion_capture = motion_capture
        self.motion_streak_threshold = motion_streak
        self.capture_cooldown_sec = capture_cooldown_sec

        # Clipping Mode — all live-tunable from the dashboard via
        # apply_clipping_config(); none of these need a restart.
        self.clipping = False
        self.clip_threshold = clip_threshold
        self.clip_streak_threshold = clip_streak
        self.clip_cooldown_sec = clip_cooldown_sec
        self.clip_run_model = clip_run_model
        self.dry_run = False

        self.last_label: str | None = None
        # A streak is now counted over consecutive *interesting* frames, not
        # over a repeated label: the label is allowed to flicker between
        # grooming/yawn/standing while the "not resting" signal holds steady,
        # which is exactly the case the old label-matching streak dropped.
        # streak_scores accumulates alertConfidence per label across the streak
        # so the fired alert is named by the whole streak, not its last frame.
        self.streak_count = 0
        self.streak_scores: dict[str, float] = {}
        self.motion_streak = 0
        self.last_capture_at = 0.0

    def apply_clipping_config(
        self,
        *,
        clipping: bool,
        threshold: float | None = None,
        cooldown: float | None = None,
        streak: int | None = None,
        dry_run: bool = False,
    ) -> bool:
        """Push dashboard settings onto the loop. Returns True if the mode
        itself flipped, which is the only case the caller has to restart ffmpeg
        for (frame_interval and segment_secs are baked into its args); a pure
        threshold or cooldown change takes effect on the very next frame."""
        was_clipping = self.clipping
        self.clipping = bool(clipping)
        if threshold is not None:
            self.clip_threshold = float(threshold)
        if cooldown is not None:
            self.clip_cooldown_sec = float(cooldown)
        if streak is not None:
            self.clip_streak_threshold = max(1, int(streak))
        self.dry_run = bool(dry_run)

        if was_clipping != self.clipping:
            # Streaks counted under the old thresholds mean nothing under the
            # new ones, and the embedding window would blend across a gap in
            # which the camera was restarted.
            self.motion_streak = 0
            self.streak_count = 0
            self.streak_scores = {}
            self.last_label = None
            # last_capture_at is shared by both paths, and the two cooldowns
            # differ by a factor of four. Carrying it across the flip made a
            # just-fired 30s alert cooldown silently suppress the first ~28s of
            # harvesting — which reads as "Clipping Mode is broken" at exactly
            # the moment someone turns it on and watches.
            self.last_capture_at = 0.0
            if self.predictor is not None:
                self.predictor.buffer.clear()
        return was_clipping != self.clipping

    def _classify(self, pixel_frame: np.ndarray) -> dict:
        """Mirrors classify(): embed one frame, push to the ring buffer,
        classify the window's mean via the temporal head. windowFill lets
        callers gate on warm-up the same way capture.mjs does."""
        if self.predictor is None:
            return dict(NO_PREDICTION)
        probs = self.predictor.push_frame(normalize(pixel_frame))
        window_fill = len(self.predictor.buffer)
        if probs is None:
            return {**NO_PREDICTION, "windowFill": window_fill}
        idx = int(np.argmax(probs))

        # The best NON-baseline class. Once interest_score has decided a frame
        # is worth alerting on, something has to name it: /api/notify rejects
        # "normal" outright and a "normal" suggestion in Label Studio is no
        # help to a reviewer. This is the descriptive answer to "interesting
        # how?", deliberately separate from the gate above it.
        alert_idx = max(
            range(len(LABELS)),
            key=lambda k: -1.0 if k == NORMAL_INDEX else float(probs[k]),
        )
        return {
            "label": LABELS[idx],
            "confidence": round(float(probs[idx]) * 100),
            "windowFill": window_fill,
            "interest": float(interest_score(probs)),
            "alertLabel": LABELS[alert_idx],
            "alertConfidence": round(float(probs[alert_idx]) * 100),
        }

    def handle_frame(self, pixel_frame: np.ndarray, now: float | None = None) -> dict:
        """pixel_frame: [224,224,3] uint8 RGB, raw. Motion detection consumes
        it as-is; _classify() normalizes its own copy before embedding — same
        split as the JS version, where motionLevel() and embedFrame() both
        take the same raw frameBuf but only the classifier path normalizes."""
        now = time.time() if now is None else now

        motion = self.motion.motion_level(pixel_frame)

        # Clipping Mode skips inference entirely (unless the RUN_MODEL escape
        # hatch is set): the point is volume of unlabeled footage, and a model
        # that has no current training data behind it would only add noise to
        # the public prediction feed.
        run_model = self.predictor is not None and (
            not self.clipping or self.clip_run_model
        )
        pred = self._classify(pixel_frame) if run_model else dict(NO_PREDICTION)

        warm = pred["windowFill"] >= self.embed_window

        candidate = (
            not self.clipping
            and warm
            and pred["interest"] >= self.interest_thresh
            and motion["level"] >= self.motion_floor
        )

        if candidate:
            self.streak_count += 1
            self.streak_scores[pred["alertLabel"]] = (
                self.streak_scores.get(pred["alertLabel"], 0.0)
                + pred["alertConfidence"]
            )
        else:
            self.streak_count = 0
            self.streak_scores = {}

        behavior_alert = candidate and self.streak_count >= self.alert_streak

        # Name the alert from the whole streak rather than whichever frame
        # happened to trip the counter: summed confidence, so three frames of
        # weak "zoomies" outrank one strong flicker of "standing".
        alert = None
        if behavior_alert:
            best = max(self.streak_scores, key=lambda k: self.streak_scores[k])
            alert = {
                "label": best,
                "confidence": pred["alertConfidence"]
                if best == pred["alertLabel"]
                else round(self.streak_scores[best] / self.streak_count),
            }

        # Clipping Mode runs the same streak/cooldown machinery on its own,
        # much more permissive, numbers.
        threshold = self.clip_threshold if self.clipping else self.motion_capture
        streak_needed = (
            self.clip_streak_threshold
            if self.clipping
            else self.motion_streak_threshold
        )
        cooldown = (
            self.clip_cooldown_sec if self.clipping else self.capture_cooldown_sec
        )

        self.motion_streak = (
            self.motion_streak + 1 if motion["level"] >= threshold else 0
        )
        motion_event = self.motion_streak >= streak_needed

        capture_ready = (now - self.last_capture_at) >= cooldown

        # Nothing reaches the public prediction feed while harvesting — with no
        # model there is no label to publish anyway, and under RUN_MODEL the
        # predictions are diagnostic only.
        publish_prediction = (
            not self.clipping and warm and pred["label"] != self.last_label
        )
        if publish_prediction:
            self.last_label = pred["label"]

        action = "none"
        if (behavior_alert or motion_event) and capture_ready:
            self.last_capture_at = now
            self.streak_count = 0
            self.streak_scores = {}
            self.motion_streak = 0
            if self.clipping:
                action = "clip"
            else:
                action = "alert" if behavior_alert else "motion_capture"

        return {
            "motion": motion,
            "prediction": pred,
            "warm": warm,
            "candidate": candidate,
            "behaviorAlert": behavior_alert,
            # The {label, confidence} the alert should be sent under, or None.
            # Shaped like `prediction` so _upload_and_alert takes either one.
            "alert": alert,
            "motionEvent": motion_event,
            "publishPrediction": publish_prediction,
            "threshold": threshold,
            "action": action,
        }


# ── Boot / runtime wiring (Phase 5) ─────────────────────────────────────────

SERVER_PY_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SERVER_PY_DIR.parent
SERVER_DIR = ROOT_DIR / "server"
MODELS_DIR = SERVER_PY_DIR / "models"

load_dotenv(SERVER_DIR / ".env")


def log(*args) -> None:
    print(time.strftime("%I:%M:%S %p"), *args)


class Agent:
    def __init__(self):
        self.camera_format = os.environ.get("CAMERA_FORMAT", "dshow")
        self.camera_input = os.environ.get("CAMERA_INPUT", "video=onn 4K Webcam")
        self.camera_size = os.environ.get("CAMERA_SIZE", "1280x720")
        self.camera_fps = os.environ.get("CAMERA_FPS", "30")
        self.frame_interval = float(os.environ.get("AGENT_FRAME_SECONDS", "1.5"))
        self.embed_window = int(os.environ.get("AGENT_EMBED_WINDOW", "8"))
        self.segment_secs = int(os.environ.get("AGENT_SEGMENT_SECONDS", "12"))
        self.segment_max_age_sec = self.segment_secs * 2.5
        self.interest_thresh = float(
            os.environ.get("AGENT_INTEREST_THRESHOLD", str(INTEREST_THRESH))
        )
        self.motion_floor = float(os.environ.get("AGENT_MOTION_FLOOR", "1"))
        self.alert_streak = int(os.environ.get("AGENT_ALERT_STREAK", "3"))
        self.motion_capture = float(os.environ.get("AGENT_MOTION_CAPTURE", "2.5"))
        self.motion_streak = int(os.environ.get("AGENT_MOTION_STREAK", "3"))
        self.capture_cooldown_sec = float(
            os.environ.get("AGENT_CAPTURE_COOLDOWN_SEC", "30")
        )

        # Clipping Mode. These are boot defaults only — monitor.json is
        # authoritative for threshold/cooldown/dry-run once the dashboard has
        # written it, and _poll_monitor_loop pushes those in every 5s.
        self.clip_frame_interval = float(
            os.environ.get("AGENT_CLIPPING_FRAME_SECONDS", "0.5")
        )
        self.clip_segment_secs = int(
            os.environ.get("AGENT_CLIPPING_SEGMENT_SECONDS", "8")
        )
        self.clip_streak = int(
            os.environ.get("AGENT_CLIPPING_STREAK", str(CLIPPING_STREAK))
        )
        self.clip_run_model = os.environ.get("AGENT_CLIPPING_RUN_MODEL", "0") not in (
            "",
            "0",
            "false",
            "False",
        )
        # Segments whose motion fired a clip, awaiting the segment closing.
        # Each entry: {"name", "mtime", "motion"}.
        self.pending_clips: list[dict] = []
        self.clipped_segments: set[tuple[str, float]] = set()
        self.clip_budget_blocked = False

        self.alert_webhook_url = os.environ.get("ALERT_WEBHOOK_URL")
        self.alert_webhook_secret = os.environ.get("ALERT_WEBHOOK_SECRET", "")

        self.seg_dir = SERVER_DIR / "agent" / "segments"
        # Deliberately NOT server/agent/ — this is the Python agent's own scratch
        # file, kept under server_py/ so it never collides with the Node agent's
        # segments directory if both existed on disk at once.
        self.live_frame_path = SERVER_PY_DIR / "agent" / "live_frame.jpg"

        self.client = AgentClient()
        self.decision_loop: AgentDecisionLoop | None = None
        self.ffmpeg: FfmpegCapture | None = None
        self.live_poller: LiveFramePoller | None = None
        self.paused = False
        self.last_capture_at = 0.0

    async def load_models(self) -> None:
        """Load the ONNX artifacts, but never die for want of them.

        The models directory is committed, yet a wipe, a half-finished export or
        a corrupt download all used to raise here — during run(), before
        anything else — which under pm2's autorestart just burned through
        max_restarts and left the agent dead. Motion detection needs no model at
        all, and Clipping Mode is entirely motion-driven, so a missing model
        should cost alerting and nothing else.
        """
        log("Loading ONNX models...")
        backbone = head = None
        try:
            backbone = Backbone(str(MODELS_DIR / "backbone_int8.onnx"))
            head = TemporalHead(str(MODELS_DIR / "head.onnx"))
        except Exception as err:
            backbone = head = None
            log(
                f"⚠ ONNX models unavailable ({err}) — running motion-only, alerts disabled."
            )
            log("  Clipping Mode still works; it never uses the model.")

        self.decision_loop = AgentDecisionLoop(
            backbone,
            head,
            embed_window=self.embed_window,
            interest_thresh=self.interest_thresh,
            motion_floor=self.motion_floor,
            alert_streak=self.alert_streak,
            motion_capture=self.motion_capture,
            motion_streak=self.motion_streak,
            capture_cooldown_sec=self.capture_cooldown_sec,
            clip_streak=self.clip_streak,
            clip_run_model=self.clip_run_model,
        )
        if backbone and head:
            log(f"✓ Models ready (backend: onnxruntime; classes: {', '.join(LABELS)})")

    async def _segments_by_mtime(self) -> list[dict]:
        if not self.seg_dir.exists():
            return []
        names = [f.name for f in self.seg_dir.iterdir() if f.name.endswith(".webm")]
        segs = [{"name": n, "mtime": (self.seg_dir / n).stat().st_mtime} for n in names]
        segs.sort(key=lambda s: s["mtime"])
        return segs

    async def _latest_finished_segment(self) -> str | None:
        segs = await self._segments_by_mtime()
        if len(segs) < 2:
            return None
        seg = segs[-2]  # newest file is still being written
        if time.time() - seg["mtime"] > self.segment_max_age_sec:
            log(f"  (latest finished segment {seg['name']} is stale — skipping upload)")
            return None
        return seg["name"]

    async def _upload_segment(
        self, seg_name: str | None = None, source: str | None = None
    ) -> str | None:
        """seg_name=None keeps the original behavior verbatim for the alert
        path: upload the latest FINISHED segment, i.e. the ~12s leading up to
        the event. Clipping Mode passes an explicit name instead (see
        _queue_clip) so the clip contains the motion rather than preceding it.

        `source` tags the upload server-side: "clipping" gets the
        recording-clip- prefix and is subject to the harvest budget.
        """
        if seg_name is None:
            seg_name = await self._latest_finished_segment()
        if not seg_name:
            log("  (no finished segment yet — skipping upload)")
            return None
        try:
            data = await asyncio.to_thread((self.seg_dir / seg_name).read_bytes)
        except OSError as err:
            log(f"  segment {seg_name} unreadable ({err}) — skipping upload")
            return None
        resp = await self.client.post(
            "/api/recordings",
            files={"video": ("clip.webm", data, "video/webm")},
            data={"source": source} if source else None,
        )
        if resp.status_code == 507:
            # Budget reached. The server has already flipped clipping off in
            # monitor.json; _poll_monitor_loop picks that up within 5s.
            if not self.clip_budget_blocked:
                self.clip_budget_blocked = True
                try:
                    reason = resp.json().get("reason")
                except Exception:
                    reason = None
                log(
                    f"  ⛔ Clipping stopped by the server: {reason or 'budget reached'}"
                )
            return None
        if resp.status_code != 200:
            log("  upload failed:", resp.status_code)
            return None
        return resp.json()["filename"]

    async def _notify_webhook(self, pred: dict, filename: str) -> None:
        if not self.alert_webhook_url:
            return
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    self.alert_webhook_url,
                    headers={"X-BunnyCam-Secret": self.alert_webhook_secret},
                    json={
                        "label": pred["label"],
                        "confidence": pred["confidence"],
                        "filename": filename,
                        "clipUrl": f"{self.client.client.base_url}/recordings/{filename}",
                        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                )
            log(f"  → n8n webhook notified ({pred['label']})")
        except Exception as err:
            log("  webhook failed:", err)

    async def _upload_and_alert(self, pred: dict) -> None:
        """`pred` is the decision's `alert` dict — {label, confidence} named by
        the whole streak, never the raw argmax. It is guaranteed non-"normal",
        which /api/notify's ALERT_LABELS requires and a Label Studio reviewer
        needs; the raw prediction can and often does still read "normal" on the
        very frame that fires, which is the point of the interest gate.
        """
        filename = await self._upload_segment()
        if not filename:
            return
        log(
            f"  ↑ Uploaded clip {filename} → suggested {pred['label']} (review in Label Studio)"
        )

        try:
            await self.client.post(
                f"/api/labels/{filename}/suggest", json={"label": pred["label"]}
            )
        except Exception:
            pass

        if self.alert_webhook_url:
            gate_resp = await self.client.post(
                "/api/notify/gate", json={"label": pred["label"]}
            )
            gate = (
                gate_resp.json()
                if gate_resp.status_code == 200
                else {"allow": False, "reason": "gate unreachable"}
            )
            if gate.get("allow"):
                await self._notify_webhook(pred, filename)
            else:
                log(f"  → webhook skipped ({gate.get('reason')})")
        else:
            resp = await self.client.post(
                "/api/notify",
                json={
                    "label": pred["label"],
                    "confidence": pred["confidence"],
                    "filename": filename,
                },
            )
            result = resp.json()
            log(
                f"  ✉ Email: {'sent' if result.get('sent') else result.get('reason') or result.get('error')}"
            )

    async def _upload_motion_clip(self, motion_level: float) -> None:
        filename = await self._upload_segment()
        if not filename:
            return
        log(
            f"  ↑ Motion clip {filename} saved unlabeled (motion={motion_level:.1f}) — review in Label Studio"
        )

    # ── Clipping Mode: defer to the segment the motion happened in ─────────

    async def _queue_clip(self, motion_level: float) -> None:
        """Note the segment ffmpeg is writing RIGHT NOW and queue it. The alert
        path uploads segs[-2], the latest finished one, which for an alert is
        useful pre-roll — but for harvesting it means hand-labeling the 8s
        BEFORE the motion. The in-progress segment is the one that contains it,
        so wait for it to close instead (_drain_pending_clips_loop).
        """
        segs = await self._segments_by_mtime()
        if not segs:
            log("  (no segment being written yet — skipping clip)")
            return
        target = segs[-1]
        # Pending entries dedupe by NAME. Only one segment is ever being
        # written, so its name identifies it — and its mtime is still moving,
        # which is why the key cannot include it here: two events a few seconds
        # apart in the same segment would otherwise carry different keys and
        # upload the same footage twice.
        if any(c["name"] == target["name"] for c in self.pending_clips):
            return
        # The already-uploaded set DOES carry the mtime, because ffmpeg
        # renumbers from seg-00000 on restart and a fresh segment must not be
        # mistaken for one already harvested. This is belt-and-braces: a segment
        # is only uploaded once it is no longer the one being written, so
        # _queue_clip cannot normally reach it again.
        key = (target["name"], round(target["mtime"], 3))
        if key in self.clipped_segments:
            return
        self.pending_clips.append(
            {
                "name": target["name"],
                "mtime": target["mtime"],
                "motion": motion_level,
            }
        )
        log(
            f"  ⧗ Clip queued (motion={motion_level:.1f}) — waiting for {target['name']} to close"
        )

    async def _drain_pending_clips_loop(self) -> None:
        while True:
            await asyncio.sleep(2)
            if not self.pending_clips:
                continue
            try:
                await self._drain_pending_clips_once()
            except Exception as err:
                log("  clip drain error:", err)

    async def _drain_pending_clips_once(self) -> None:
        """Upload each queued segment once ffmpeg has finished writing it.

        "Finished" = a strictly newer .webm exists in the directory. The
        fallback is age, for the case where capture stops before the next
        segment is ever created and the clip would otherwise be stranded.
        _prune_segments_loop's 600s cutoff is far longer than either window, so
        nothing gets deleted out from under this.

        Split out from the loop above so it can be driven directly by tests
        without waiting on the poll interval.
        """
        segs = await self._segments_by_mtime()
        by_name = {s["name"]: s for s in segs}
        newest_mtime = segs[-1]["mtime"] if segs else 0.0
        now = time.time()
        settled_after = self.clip_segment_secs * 1.5

        still_pending = []
        for clip in self.pending_clips:
            current = by_name.get(clip["name"])
            if current is None:
                # Pruned or wiped while waiting — drop it rather than spinning
                # on a file that will never come back.
                log(f"  (queued segment {clip['name']} vanished — dropping clip)")
                continue
            # Two ways a segment can be finished, and both must be checked
            # against the FILE, never against how long the clip has been
            # queued: a wall-clock timeout would happily upload a file ffmpeg
            # is still writing.
            #   1. A strictly newer file exists, so this is no longer the one
            #      being written. The normal case.
            #   2. Nothing has been written to it for well over a segment's
            #      length, so capture stopped and ffmpeg finalized it. Without
            #      this, the last clip before monitoring is turned off is
            #      stranded forever, since no newer segment ever appears.
            closed = (
                newest_mtime > current["mtime"]
                or (now - current["mtime"]) > settled_after
            )
            if not closed:
                still_pending.append(clip)
                continue

            # Key on the mtime the file settled at. Both branches above mean
            # the file has stopped changing, so a later _queue_clip that lands
            # on this same segment sees the identical mtime and the guard below
            # actually fires — keying on the moving, mid-write mtime made it
            # dead code.
            self.clipped_segments.add((clip["name"], round(current["mtime"], 3)))
            filename = await self._upload_segment(clip["name"], source="clipping")
            if filename:
                log(
                    f"  ↑ Clip {filename} saved unlabeled (motion={clip['motion']:.1f})"
                    f" — review in Label Studio"
                )

        self.pending_clips = still_pending

        # Keep the dedupe set from growing without bound over a long unattended
        # harvest; segment names cycle quickly anyway.
        if len(self.clipped_segments) > 500:
            self.clipped_segments.clear()

    async def _on_frame(self, frame_buf: bytes) -> None:
        pixel_frame = np.frombuffer(frame_buf, dtype=np.uint8).reshape(224, 224, 3)
        decision = self.decision_loop.handle_frame(pixel_frame)
        motion, pred = decision["motion"], decision["prediction"]

        if self.decision_loop.clipping:
            # Same log line shape, minus the classifier half that isn't running.
            log(
                f"frame  motion={motion['level']:.1f}  cells={motion['hotCells']}"
                f"{'  (scene-wide change, scored 0)' if motion['lighting'] else ''}"
                f"  →  clipping (threshold {decision['threshold']:.1f})"
                f"{('  ✂ WOULD CLIP' if self.decision_loop.dry_run else '  ✂ CLIP') if decision['action'] == 'clip' else ''}"
            )
            # Telemetry for the dashboard's tuning sparkline. Fire-and-forget
            # and exception-swallowed, exactly like the predictions post below:
            # a telemetry hiccup must never interrupt capture.
            try:
                await self.client.post(
                    "/api/motion",
                    json={
                        "level": motion["level"],
                        "hotCells": motion["hotCells"],
                        "lighting": motion["lighting"],
                        "threshold": decision["threshold"],
                        "wouldCapture": decision["action"] == "clip",
                        "captured": decision["action"] == "clip"
                        and not self.decision_loop.dry_run,
                    },
                )
            except Exception:
                pass
        else:
            warmup_tag = (
                f"  (warmup {pred['windowFill']}/{self.embed_window})"
                if not decision["warm"]
                else ""
            )
            log(
                f"frame  motion={motion['level']:.1f}  cells={motion['hotCells']}"
                f"{'  (scene-wide change, scored 0)' if motion['lighting'] else ''}"
                f"  →  {pred['label']} {pred['confidence']}%"
                f"  interest={pred['interest']:.0f}/{self.interest_thresh:.0f}{warmup_tag}"
                f"{'  ⚠ ALERT ' + decision['alert']['label'] if decision['behaviorAlert'] else '  ● MOTION' if decision['motionEvent'] else ''}"
            )

        if decision["publishPrediction"]:
            try:
                await self.client.post(
                    "/api/predictions",
                    json={"label": pred["label"], "confidence": pred["confidence"]},
                )
            except Exception:
                pass

        if decision["action"] == "alert":
            try:
                await self._upload_and_alert(decision["alert"])
            except Exception as err:
                log("  alert error:", err)
        elif decision["action"] == "motion_capture":
            try:
                await self._upload_motion_clip(motion["level"])
            except Exception as err:
                log("  capture error:", err)
        elif decision["action"] == "clip" and not self.decision_loop.dry_run:
            try:
                await self._queue_clip(motion["level"])
            except Exception as err:
                log("  clip error:", err)

    async def _on_live_frame(self, jpeg_bytes: bytes) -> None:
        try:
            await self.client.post(
                "/api/stream/frame",
                content=jpeg_bytes,
                headers={"Content-Type": "image/jpeg"},
            )
        except Exception:
            pass

    async def start_capture(self) -> None:
        # frame_interval and segment_secs are baked into ffmpeg's args, so
        # Clipping Mode's denser cadence and shorter segments only take effect
        # across a restart — which is why toggling the mode restarts capture
        # (see _poll_monitor_loop). Threshold and cooldown need no restart.
        clipping = self.decision_loop is not None and self.decision_loop.clipping
        frame_interval = self.clip_frame_interval if clipping else self.frame_interval
        segment_secs = self.clip_segment_secs if clipping else self.segment_secs
        self.segment_max_age_sec = segment_secs * 2.5

        self.ffmpeg = FfmpegCapture(
            ffmpeg_path=os.environ.get("FFMPEG_PATH")
            or imageio_ffmpeg.get_ffmpeg_exe(),
            camera_format=self.camera_format,
            camera_input=self.camera_input,
            camera_size=self.camera_size,
            camera_fps=self.camera_fps,
            frame_interval=frame_interval,
            segment_secs=segment_secs,
            seg_dir=self.seg_dir,
            live_frame_path=self.live_frame_path,
            on_frame=self._on_frame,
        )
        await self.ffmpeg.start()
        self.live_poller = LiveFramePoller(self.live_frame_path, self._on_live_frame)
        self.live_poller.start()

    async def _prune_segments_loop(self) -> None:
        cutoff_age = 600.0
        while True:
            await asyncio.sleep(60)
            try:
                now = time.time()
                for seg in await self._segments_by_mtime():
                    if now - seg["mtime"] > cutoff_age:
                        (self.seg_dir / seg["name"]).unlink(missing_ok=True)
            except Exception:
                pass

    async def _stop_capture(self) -> None:
        """Stop ffmpeg and wait for the child to actually exit before returning.

        Restarting capture for a mode change is a stop immediately followed by a
        start, unlike the monitor toggle where a human's next click is the gap.
        A force-started second ffmpeg while the first still holds the camera
        fails to open the device, so wait for the handle to be released.
        """
        if self.live_poller:
            self.live_poller.stop()
        if not self.ffmpeg:
            return
        ffmpeg, supervisor = self.ffmpeg, self.ffmpeg._supervisor
        proc = ffmpeg.proc
        ffmpeg.stop()
        if proc:
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass
        # proc.wait() returning is NOT enough. _watch_exit awaits the same
        # future in its own task and only then tears the readers down, so
        # without waiting for it a frame still sitting in the old handoff can
        # be delivered after _reset_decision_state() and seed the fresh
        # MotionDetector with a pre-restart scene — exactly the cross-gap
        # blending that reset exists to prevent. paused is set, so _watch_exit
        # stops the readers and returns instead of restarting.
        if supervisor is not None:
            try:
                await asyncio.wait_for(asyncio.shield(supervisor), timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass

    def _reset_decision_state(self) -> None:
        """Drop everything that describes the scene before a capture gap, so
        frames from either side of it can never blend."""
        self.decision_loop.motion = MotionDetector()
        if self.decision_loop.predictor is not None:
            self.decision_loop.predictor.buffer.clear()
        self.decision_loop.last_label = None
        self.decision_loop.motion_streak = 0

    async def _poll_monitor_loop(self) -> None:
        while True:
            await asyncio.sleep(5)
            try:
                resp = await self.client.get("/api/monitor")
                state = resp.json()
            except Exception:
                continue  # server unreachable — keep current state
            enabled = state.get("enabled", True)

            if not enabled and not self.paused:
                self.paused = True
                log("⏸ Monitoring turned OFF from dashboard — releasing camera")
                if self.ffmpeg:
                    self.ffmpeg.stop()
                if self.live_poller:
                    self.live_poller.stop()
                self._reset_decision_state()
                continue
            elif enabled and self.paused:
                self.paused = False
                log("▶ Monitoring turned ON from dashboard — starting capture")
                await self.start_capture()

            # Clipping Mode. Threshold/cooldown/dry-run land on the decision
            # loop directly and take effect on the next frame; only flipping
            # the mode itself needs a capture restart, because the frame
            # cadence and segment length are ffmpeg arguments.
            clipping = bool(state.get("clipping", False))
            mode_changed = self.decision_loop.apply_clipping_config(
                clipping=clipping,
                threshold=state.get("clippingMotionThreshold"),
                cooldown=state.get("clippingCooldownSec"),
                dry_run=bool(state.get("clippingDryRun", False)),
            )
            if not mode_changed:
                continue

            self.clip_budget_blocked = False
            if clipping:
                dry = (
                    " (dry run — nothing will be saved)"
                    if self.decision_loop.dry_run
                    else ""
                )
                log(
                    f"✂ Clipping Mode ON{dry} — motion ≥ {self.decision_loop.clip_threshold:.1f}, "
                    f"{self.clip_segment_secs}s clips, {self.decision_loop.clip_cooldown_sec:.0f}s cooldown. "
                    f"Alerts and predictions are paused."
                )
            else:
                reason = state.get("clippingStoppedReason")
                log(f"✂ Clipping Mode OFF{f' — {reason}' if reason else ''}")

            if not self.paused:
                await self._stop_capture()
                # Only now that no more frames can arrive. A frame already in
                # flight when the mode flipped can queue a clip naming a
                # segment from the OUTGOING ffmpeg run — and the incoming one
                # restarts numbering at seg-00000 and overwrites that file, so
                # the clip would be uploaded from footage it never saw.
                # Clearing before the stop left exactly that window open.
                self.pending_clips.clear()
                self.clipped_segments.clear()
                self._reset_decision_state()
                await self.start_capture()

    async def run(self) -> None:
        log("\U0001f407 BunnyCam Python Agent starting...")
        self.seg_dir.mkdir(parents=True, exist_ok=True)
        await self.client.verify()
        log("✓ AGENT_TOKEN verified")
        await self.load_models()

        try:
            resp = await self.client.get("/api/monitor")
            enabled = resp.json().get("enabled", True)
        except Exception:
            enabled = True

        if not enabled:
            self.paused = True
            log("⏸ Monitoring is OFF — camera idle until enabled from dashboard")
        else:
            await self.start_capture()

        await asyncio.gather(
            self._poll_monitor_loop(),
            self._prune_segments_loop(),
            self._drain_pending_clips_loop(),
        )


async def main():
    agent = Agent()
    run_task = asyncio.ensure_future(agent.run())

    def _shutdown():
        log("Shutdown signal received — stopping ffmpeg child before exit")
        if agent.ffmpeg:
            agent.ffmpeg.stop()
        if agent.live_poller:
            agent.live_poller.stop()
        run_task.cancel()

    loop = asyncio.get_running_loop()
    try:
        import signal

        loop.add_signal_handler(signal.SIGTERM, _shutdown)
        loop.add_signal_handler(signal.SIGINT, _shutdown)
    except (NotImplementedError, AttributeError):
        pass  # Windows Proactor loop doesn't support add_signal_handler; Ctrl+C still raises KeyboardInterrupt

    try:
        await run_task
    except (asyncio.CancelledError, KeyboardInterrupt):
        if agent.ffmpeg:
            agent.ffmpeg.stop()
        if agent.live_poller:
            agent.live_poller.stop()
        # Give the ffmpeg child a moment to actually exit before the process
        # tree tears down — a force-killed parent orphans it, leaving the
        # camera locked for the next attempt.
        if agent.ffmpeg and agent.ffmpeg.proc:
            try:
                await asyncio.wait_for(agent.ffmpeg.proc.wait(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass


if __name__ == "__main__":
    asyncio.run(main())
