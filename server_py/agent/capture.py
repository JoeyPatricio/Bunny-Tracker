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
CONF_THRESH = 70
MOTION_FLOOR = 1.0
ALERT_STREAK = 3
MOTION_CAPTURE = 2.5
MOTION_STREAK = 3
CAPTURE_COOLDOWN_SEC = 30


class AgentDecisionLoop:
    def __init__(self, backbone, head, *,
                 embed_window: int = EMBED_WINDOW,
                 conf_thresh: float = CONF_THRESH,
                 motion_floor: float = MOTION_FLOOR,
                 alert_streak: int = ALERT_STREAK,
                 motion_capture: float = MOTION_CAPTURE,
                 motion_streak: int = MOTION_STREAK,
                 capture_cooldown_sec: float = CAPTURE_COOLDOWN_SEC):
        self.predictor = Predictor(backbone, head, embed_window)
        self.motion = MotionDetector()
        self.embed_window = embed_window
        self.conf_thresh = conf_thresh
        self.motion_floor = motion_floor
        self.alert_streak = alert_streak
        self.motion_capture = motion_capture
        self.motion_streak_threshold = motion_streak
        self.capture_cooldown_sec = capture_cooldown_sec

        self.last_label: str | None = None
        self.streak_label: str | None = None
        self.streak_count = 0
        self.motion_streak = 0
        self.last_capture_at = 0.0

    def _classify(self, pixel_frame: np.ndarray) -> dict:
        """Mirrors classify(): embed one frame, push to the ring buffer,
        classify the window's mean via the temporal head. windowFill lets
        callers gate on warm-up the same way capture.mjs does."""
        probs = self.predictor.push_frame(normalize(pixel_frame))
        window_fill = len(self.predictor.buffer)
        if probs is None:
            return {"label": None, "confidence": 0, "windowFill": window_fill}
        idx = int(np.argmax(probs))
        return {"label": LABELS[idx], "confidence": round(float(probs[idx]) * 100), "windowFill": window_fill}

    def handle_frame(self, pixel_frame: np.ndarray, now: float | None = None) -> dict:
        """pixel_frame: [224,224,3] uint8 RGB, raw. Motion detection consumes
        it as-is; _classify() normalizes its own copy before embedding — same
        split as the JS version, where motionLevel() and embedFrame() both
        take the same raw frameBuf but only the classifier path normalizes."""
        now = time.time() if now is None else now

        motion = self.motion.motion_level(pixel_frame)
        pred = self._classify(pixel_frame)

        warm = pred["windowFill"] >= self.embed_window

        candidate = (
            warm
            and pred["label"] != "normal"
            and pred["confidence"] >= self.conf_thresh
            and motion["level"] >= self.motion_floor
        )

        if candidate and pred["label"] == self.streak_label:
            self.streak_count += 1
        elif candidate:
            self.streak_label = pred["label"]
            self.streak_count = 1
        else:
            self.streak_label = None
            self.streak_count = 0

        behavior_alert = candidate and self.streak_count >= self.alert_streak

        self.motion_streak = self.motion_streak + 1 if motion["level"] >= self.motion_capture else 0
        motion_event = self.motion_streak >= self.motion_streak_threshold

        capture_ready = (now - self.last_capture_at) >= self.capture_cooldown_sec

        publish_prediction = warm and pred["label"] != self.last_label
        if publish_prediction:
            self.last_label = pred["label"]

        action = "none"
        if (behavior_alert or motion_event) and capture_ready:
            self.last_capture_at = now
            self.streak_count = 0
            self.motion_streak = 0
            action = "alert" if behavior_alert else "motion_capture"

        return {
            "motion": motion,
            "prediction": pred,
            "warm": warm,
            "candidate": candidate,
            "behaviorAlert": behavior_alert,
            "motionEvent": motion_event,
            "publishPrediction": publish_prediction,
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
        self.conf_thresh = float(os.environ.get("AGENT_CONFIDENCE_THRESHOLD", "70"))
        self.motion_floor = float(os.environ.get("AGENT_MOTION_FLOOR", "1"))
        self.alert_streak = int(os.environ.get("AGENT_ALERT_STREAK", "3"))
        self.motion_capture = float(os.environ.get("AGENT_MOTION_CAPTURE", "2.5"))
        self.motion_streak = int(os.environ.get("AGENT_MOTION_STREAK", "3"))
        self.capture_cooldown_sec = float(os.environ.get("AGENT_CAPTURE_COOLDOWN_SEC", "30"))
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
        log("Loading ONNX models...")
        backbone = Backbone(str(MODELS_DIR / "backbone_int8.onnx"))
        head = TemporalHead(str(MODELS_DIR / "head.onnx"))
        self.decision_loop = AgentDecisionLoop(
            backbone, head,
            embed_window=self.embed_window, conf_thresh=self.conf_thresh,
            motion_floor=self.motion_floor, alert_streak=self.alert_streak,
            motion_capture=self.motion_capture, motion_streak=self.motion_streak,
            capture_cooldown_sec=self.capture_cooldown_sec,
        )
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

    async def _upload_segment(self) -> str | None:
        seg = await self._latest_finished_segment()
        if not seg:
            log("  (no finished segment yet — skipping upload)")
            return None
        data = await asyncio.to_thread((self.seg_dir / seg).read_bytes)
        resp = await self.client.post(
            "/api/recordings",
            files={"video": ("clip.webm", data, "video/webm")},
        )
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
                        "label": pred["label"], "confidence": pred["confidence"], "filename": filename,
                        "clipUrl": f"{self.client.client.base_url}/recordings/{filename}",
                        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                )
            log(f"  → n8n webhook notified ({pred['label']})")
        except Exception as err:
            log("  webhook failed:", err)

    async def _upload_and_alert(self, pred: dict) -> None:
        filename = await self._upload_segment()
        if not filename:
            return
        log(f"  ↑ Uploaded clip {filename} → suggested {pred['label']} (review in Label Studio)")

        try:
            await self.client.post(f"/api/labels/{filename}/suggest", json={"label": pred["label"]})
        except Exception:
            pass

        if self.alert_webhook_url:
            gate_resp = await self.client.post("/api/notify/gate", json={"label": pred["label"]})
            gate = gate_resp.json() if gate_resp.status_code == 200 else {"allow": False, "reason": "gate unreachable"}
            if gate.get("allow"):
                await self._notify_webhook(pred, filename)
            else:
                log(f"  → webhook skipped ({gate.get('reason')})")
        else:
            resp = await self.client.post(
                "/api/notify",
                json={"label": pred["label"], "confidence": pred["confidence"], "filename": filename},
            )
            result = resp.json()
            log(f"  ✉ Email: {'sent' if result.get('sent') else result.get('reason') or result.get('error')}")

    async def _upload_motion_clip(self, motion_level: float) -> None:
        filename = await self._upload_segment()
        if not filename:
            return
        log(f"  ↑ Motion clip {filename} saved unlabeled (motion={motion_level:.1f}) — review in Label Studio")

    async def _on_frame(self, frame_buf: bytes) -> None:
        pixel_frame = np.frombuffer(frame_buf, dtype=np.uint8).reshape(224, 224, 3)
        decision = self.decision_loop.handle_frame(pixel_frame)
        motion, pred = decision["motion"], decision["prediction"]

        warmup_tag = f"  (warmup {pred['windowFill']}/{self.embed_window})" if not decision["warm"] else ""
        log(
            f"frame  motion={motion['level']:.1f}  cells={motion['hotCells']}"
            f"{'  (scene-wide change, scored 0)' if motion['lighting'] else ''}"
            f"  →  {pred['label']} {pred['confidence']}%{warmup_tag}"
            f"{'  ⚠ ALERT' if decision['behaviorAlert'] else '  ● MOTION' if decision['motionEvent'] else ''}"
        )

        if decision["publishPrediction"]:
            try:
                await self.client.post("/api/predictions", json={"label": pred["label"], "confidence": pred["confidence"]})
            except Exception:
                pass

        if decision["action"] == "alert":
            try:
                await self._upload_and_alert(pred)
            except Exception as err:
                log("  alert error:", err)
        elif decision["action"] == "motion_capture":
            try:
                await self._upload_motion_clip(motion["level"])
            except Exception as err:
                log("  capture error:", err)

    async def _on_live_frame(self, jpeg_bytes: bytes) -> None:
        try:
            await self.client.post(
                "/api/stream/frame", content=jpeg_bytes, headers={"Content-Type": "image/jpeg"},
            )
        except Exception:
            pass

    async def start_capture(self) -> None:
        self.ffmpeg = FfmpegCapture(
            ffmpeg_path=os.environ.get("FFMPEG_PATH") or imageio_ffmpeg.get_ffmpeg_exe(),
            camera_format=self.camera_format, camera_input=self.camera_input,
            camera_size=self.camera_size, camera_fps=self.camera_fps,
            frame_interval=self.frame_interval, segment_secs=self.segment_secs,
            seg_dir=self.seg_dir, live_frame_path=self.live_frame_path,
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

    async def _poll_monitor_loop(self) -> None:
        while True:
            await asyncio.sleep(5)
            try:
                resp = await self.client.get("/api/monitor")
                enabled = resp.json().get("enabled", True)
            except Exception:
                continue  # server unreachable — keep current state
            if not enabled and not self.paused:
                self.paused = True
                log("⏸ Monitoring turned OFF from dashboard — releasing camera")
                if self.ffmpeg:
                    self.ffmpeg.stop()
                if self.live_poller:
                    self.live_poller.stop()
                self.decision_loop.motion = MotionDetector()  # don't blend pre-pause frames
                self.decision_loop.predictor.buffer.clear()
                self.decision_loop.last_label = None
                self.decision_loop.motion_streak = 0
            elif enabled and self.paused:
                self.paused = False
                log("▶ Monitoring turned ON from dashboard — starting capture")
                await self.start_capture()

    async def run(self) -> None:
        log("\U0001F407 BunnyCam Python Agent starting...")
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

        await asyncio.gather(self._poll_monitor_loop(), self._prune_segments_loop())


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
