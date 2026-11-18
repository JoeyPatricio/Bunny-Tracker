"""Golden data for the C++ capture agent in agent-cpp/.

Records what the Python agent does today so the port can be checked against
it, written to agent-cpp/tests/fixtures/:

- motion.json: MotionDetector over the 67 motion fixture frames as one
  stream. Levels are float64 reprs, not the JS ground truth's 0.1 tolerance,
  because decision thresholds compare with >= and a small error flips them.
- inference.json: normalize() samples, backbone_int8 embeddings, and head
  logits/probs for every full window over the first fixture frames. Calls
  Backbone/TemporalHead directly, so it needs no promoted model.
- decisions.jsonl: AgentDecisionLoop replays with motion and the predictor
  stubbed, one op per line (see write_decisions for the format).
- ffmpeg_args.json: FfmpegCapture._build_args() per camera config, with the
  live-view output rewritten from the polled JPEG file to fd 3.

Delete this script when the Python agent is removed. The fixtures stay.

Usage (from server/, runtime venv):
    .venv/bin/python -m scripts.dump_agent_golden
"""
import json
import random
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.capture import AgentDecisionLoop
from agent.ffmpeg_io import FfmpegCapture
from inference.backbone import Backbone
from inference.predictor import Predictor
from inference.temporal_head import TemporalHead
from shared.frames import normalize
from shared.labels import LABELS
from shared.motion import MotionDetector

SERVER_DIR = Path(__file__).resolve().parent.parent
FRAMES = SERVER_DIR / "tests" / "fixtures" / "motion_fixture_frames.npy"
MODELS_DIR = SERVER_DIR / "models"
OUT_DIR = SERVER_DIR.parent / "agent-cpp" / "tests" / "fixtures"

INFERENCE_FRAMES = 12
INFERENCE_WINDOW = 8
# Flat CHW indices spread over all three channels, compared bit-exact.
NORMALIZE_SAMPLES = [0, 1, 223, 224 * 112 + 112, 50175, 50176, 75000, 100351, 100352, 150527]


def motion_records(frames: np.ndarray) -> list[dict]:
    detector = MotionDetector()
    return [detector.motion_level(frame) for frame in frames]


def write_motion(frames: np.ndarray) -> None:
    (OUT_DIR / "motion.json").write_text(json.dumps(motion_records(frames)) + "\n")


def write_inference(frames: np.ndarray) -> None:
    backbone = Backbone(str(MODELS_DIR / "backbone_int8.onnx"))
    head = TemporalHead(str(MODELS_DIR / "head.onnx"))
    predictor = Predictor(backbone, head, INFERENCE_WINDOW)
    records = []
    for frame in frames[:INFERENCE_FRAMES]:
        pixels = normalize(frame)
        flat = pixels.reshape(-1)
        embedding = backbone.embed(pixels)
        predictor.buffer.append(embedding)
        record = {
            "normalize": {str(i): float(flat[i]) for i in NORMALIZE_SAMPLES},
            "embedding": [float(v) for v in embedding],
            "logits": None,
            "probs": None,
        }
        if len(predictor.buffer) == INFERENCE_WINDOW:
            logits = head.classify(np.stack(predictor.buffer))
            record["logits"] = [float(v) for v in logits]
            record["probs"] = [float(v) for v in predictor.predict()]
        records.append(record)
    out = {
        "backbone": "backbone_int8.onnx",
        "head": "head.onnx",
        "window": INFERENCE_WINDOW,
        "numClasses": head.num_classes,
        "frames": records,
    }
    (OUT_DIR / "inference.json").write_text(json.dumps(out) + "\n")


class ScriptedPredictor:
    """Stands in for Predictor: returns the next scripted probability vector
    once the window is full, like push_frame does, without running a model."""

    def __init__(self, window: int):
        self.buffer: deque = deque(maxlen=window)
        self.window = window
        self.next_probs: np.ndarray | None = None

    def push_frame(self, pixel_values):
        self.buffer.append(None)
        if len(self.buffer) < self.window:
            return None
        return self.next_probs


class Recorder:
    """Drives one AgentDecisionLoop and logs every op with its result.

    Line format, one JSON object per line:
      {"op": "new", "scenario": str, "params": {...}, "model": bool}
      {"op": "config", "args": {...}, "result": bool}
      {"op": "reset"}   mirrors Agent._reset_decision_state
      {"op": "frame", "now": float, "motion": {...}, "probs": [...] | null,
       "expect": <handle_frame return value>}
    "probs" is what the predictor hands back for that frame (null while the
    window fills, or with no model). Values are float32 widened to float64.
    """

    def __init__(self, lines: list, scenario: str, *, model: bool, **params):
        self.lines = lines
        self.loop = AgentDecisionLoop(None, None, **params)
        self.predictor = None
        if model:
            self.predictor = ScriptedPredictor(self.loop.embed_window)
            self.loop.predictor = self.predictor
        self.motion = None
        self.loop.motion = SimpleNamespace(motion_level=lambda frame: self.motion)
        lines.append({"op": "new", "scenario": scenario, "params": params, "model": model})

    def config(self, **args) -> None:
        result = self.loop.apply_clipping_config(**args)
        self.lines.append({"op": "config", "args": args, "result": result})

    def reset(self) -> None:
        loop = self.loop
        if loop.predictor is not None:
            loop.predictor.buffer.clear()
        loop.last_label = None
        loop.motion_streak = 0
        loop.streak_count = 0
        loop.streak_scores.clear()
        loop.last_capture_at = 0.0
        self.lines.append({"op": "reset"})

    def frame(self, now: float, motion: dict, probs=None) -> None:
        self.motion = dict(motion)
        returned = None
        if self.predictor is not None:
            vec = None if probs is None else np.asarray(probs, dtype=np.float32)
            self.predictor.next_probs = vec
            # Predict what push_frame will hand back so the line records it.
            if len(self.predictor.buffer) + 1 >= self.predictor.window and vec is not None:
                returned = [float(v) for v in vec]
        expect = self.loop.handle_frame(np.zeros((224, 224, 3), np.uint8), now=now)
        self.lines.append({
            "op": "frame", "now": now, "motion": motion, "probs": returned, "expect": expect,
        })


def motion(level: float, hot: int = 0, lighting: bool = False) -> dict:
    return {"level": level, "hotCells": hot, "lighting": lighting}


def onehot_probs(rng: random.Random, hot: int, mass: float) -> list[float]:
    """`mass` on class `hot`, the rest spread randomly over the other classes."""
    rest = [rng.random() for _ in LABELS]
    rest[hot] = 0.0
    total = sum(rest)
    return [mass if i == hot else (1 - mass) * r / total for i, r in enumerate(rest)]


def write_decisions(frames: np.ndarray) -> None:
    lines: list = []
    fixture_motion = motion_records(frames)
    idx = {label: i for i, label in enumerate(LABELS)}

    # Clipping Mode replays from test_clipping_mode.py: real motion, no model.
    for step in (0.5, 1.5):
        for threshold in (0.4, 2.5, 8.0):
            rec = Recorder(lines, f"clipping step={step} threshold={threshold}", model=False,
                           clip_threshold=threshold, clip_streak=1, clip_cooldown_sec=8)
            rec.config(clipping=True)
            for i, m in enumerate(fixture_motion):
                rec.frame(i * step, m)

    # Alert path with no model: motion captures only, 30s cooldown.
    rec = Recorder(lines, "motion-only alert path", model=False)
    for i, m in enumerate(fixture_motion):
        rec.frame(i * 1.5, m)

    # Cooldown exactly equal to the gap, and the last_capture_at=0 start.
    rec = Recorder(lines, "cooldown equals gap", model=False,
                   clip_threshold=1.0, clip_streak=1, clip_cooldown_sec=4)
    rec.config(clipping=True)
    for i in range(24):
        rec.frame(i * 0.5, motion(5.0, 3))

    # Model path: seeded random probabilities and motion, window 3 so alerts fire.
    rng = random.Random(447)
    for window, streak in ((3, 3), (8, 2)):
        rec = Recorder(lines, f"model random window={window} streak={streak}", model=True,
                       embed_window=window, alert_streak=streak, capture_cooldown_sec=6)
        for i in range(160):
            hot = rng.randrange(len(LABELS))
            mass = rng.choice([0.2, 0.45, 0.6, 0.7, 0.9, 0.99])
            level = rng.choice([0.0, 0.5, 1.0, 2.5, 4.0, 12.0])
            rec.frame(i * 1.5, motion(level, rng.randrange(10)), onehot_probs(rng, hot, mass))

    # Exact-threshold, half-rounding and tie-break cases.
    rec = Recorder(lines, "model ties and rounding", model=True,
                   embed_window=1, alert_streak=2, capture_cooldown_sec=0)
    t = 100.0
    crafted = [
        # interest exactly at the float32 rendering of 0.6.
        [0.6, 0.1, 0.1, 0.1, 0.0, 0.05, 0.05],
        # Two non-resting classes tied: first index wins alertLabel and label.
        [0.0, 0.375, 0.375, 0.0, 0.0, 0.25, 0.0],
        # Half-way confidences: 12.5 -> 12, 37.5 -> 38, 62.5 -> 62.
        [0.125, 0.0, 0.0, 0.0, 0.0, 0.25, 0.625],
        [0.0, 0.0, 0.0, 0.0, 0.625, 0.375, 0.0],
        # Resting argmax but a non-resting class above the threshold.
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    ]
    for probs in crafted:
        rec.frame(t, motion(1.0), probs)
        t += 1.0
    # Tied streak scores across two labels: insertion order names the alert.
    for _ in range(3):
        a = [0.0] * len(LABELS)
        a[idx["grooming"]] = 0.7
        a[idx["resting_lying"]] = 0.3
        b = [0.0] * len(LABELS)
        b[idx["rearing"]] = 0.7
        b[idx["resting_lying"]] = 0.3
        rec.frame(t, motion(3.0), a)
        rec.frame(t + 1, motion(3.0), b)
        t += 2
    # Streak average that rounds half to even: (75 + 50) / 2 = 62.5 -> 62.
    for p in (0.75, 0.5):
        v = [0.0] * len(LABELS)
        v[idx["feeding"]] = p
        v[idx["locomotion"]] = 0.6 if p == 0.5 else 0.0
        v[idx["resting_sitting"]] = 1 - sum(v)
        rec.frame(t, motion(2.0), v)
        t += 1

    # Live retunes, dry run, mode flips and resets mid-stream.
    rng = random.Random(1501)
    rec = Recorder(lines, "retune and mode flips", model=True,
                   embed_window=4, alert_streak=2, capture_cooldown_sec=10,
                   clip_threshold=2.5, clip_streak=1, clip_cooldown_sec=8)
    t = 0.0

    def burst(n: int, step: float) -> None:
        nonlocal t
        for _ in range(n):
            level = rng.choice([0.0, 0.3, 1.0, 2.5, 3.0, 9.0])
            rec.frame(t, motion(level, rng.randrange(8)),
                      onehot_probs(rng, rng.randrange(len(LABELS)), rng.choice([0.5, 0.8, 0.95])))
            t += step

    burst(20, 1.5)
    rec.config(clipping=True, threshold=3.0, cooldown=4.0, dry_run=True)
    burst(20, 0.5)
    rec.config(clipping=True, threshold=0.5, cooldown=2.0, streak=0)
    burst(20, 0.5)
    rec.config(clipping=True, streak=2)
    burst(10, 0.5)
    rec.reset()
    burst(10, 0.5)
    rec.config(clipping=False)
    burst(20, 1.5)
    rec.reset()
    burst(12, 1.5)

    # Clipping with RUN_MODEL set: the predictor runs but nothing alerts.
    rec = Recorder(lines, "clipping with run_model", model=True,
                   embed_window=2, clip_run_model=True)
    rec.config(clipping=True)
    for i in range(30):
        rec.frame(i * 0.5, motion([0.0, 3.0][i % 2]), onehot_probs(rng, 2, 0.9))

    with (OUT_DIR / "decisions.jsonl").open("w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


FFMPEG_CONFIGS = [
    {"name": "v4l2 alert cadence", "camera_format": "v4l2", "camera_input": "/dev/video0",
     "camera_size": "1280x720", "camera_fps": "30", "frame_interval": 1.5, "segment_secs": 12},
    {"name": "v4l2 clipping cadence", "camera_format": "v4l2", "camera_input": "/dev/video0",
     "camera_size": "1280x720", "camera_fps": "30", "frame_interval": 0.5, "segment_secs": 8},
    {"name": "lavfi alert cadence", "camera_format": "lavfi",
     "camera_input": "testsrc2=size=1280x720:rate=30", "camera_size": "1280x720",
     "camera_fps": "30", "frame_interval": 1.5, "segment_secs": 12},
    {"name": "lavfi odd interval", "camera_format": "lavfi",
     "camera_input": "testsrc2=size=640x480:rate=15", "camera_size": "640x480",
     "camera_fps": "15", "frame_interval": 0.7, "segment_secs": 5},
]
LIVE_FRAME_PATH = "/run/bunnycam/live_frame.jpg"


def write_ffmpeg_args() -> None:
    out = []
    for config in FFMPEG_CONFIGS:
        params = {k: v for k, v in config.items() if k != "name"}
        cap = FfmpegCapture(ffmpeg_path="/usr/bin/ffmpeg", seg_dir=Path("/run/bunnycam/segments"),
                            live_frame_path=Path(LIVE_FRAME_PATH), on_frame=None, **params)
        args = cap._build_args()
        # The live-view JPEG file becomes an MJPEG stream on fd 3.
        tail = ["-f", "image2", "-update", "1", LIVE_FRAME_PATH]
        assert args[-len(tail):] == tail, args
        args = args[:-len(tail)] + ["-f", "mjpeg", "pipe:3"]
        out.append({**config, "segment_dir": "/run/bunnycam/segments", "args": args})
    (OUT_DIR / "ffmpeg_args.json").write_text(json.dumps(out, indent=1) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = np.load(FRAMES)
    write_motion(frames)
    write_inference(frames)
    write_decisions(frames)
    write_ffmpeg_args()
    for path in sorted(OUT_DIR.iterdir()):
        print(f"wrote {path.relative_to(SERVER_DIR.parent)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
