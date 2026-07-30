"""ffmpeg-based frame decode to numpy, shared by the trainer and the agent.

No torch import here on purpose: the runtime venv (agent, no torch) and the
training venv both need this module.
"""
import re
import subprocess

import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

FFMPEG = get_ffmpeg_exe()
FRAME_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def get_duration(path: str) -> float:
    """Real container duration in seconds, via ffmpeg's own stderr banner.

    Mirrors the browser's `resolveDuration()` intent (a real, finite duration)
    but sidesteps the `duration === Infinity` MediaRecorder quirk entirely: an
    ffmpeg-read container always reports a real duration.
    """
    proc = subprocess.run(
        [FFMPEG, "-i", path],
        capture_output=True,
        text=True,
    )
    m = _DURATION_RE.search(proc.stderr)
    if not m:
        return 0.0
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def sample_timestamps(duration: float, n: int, frac: float = 0.95) -> list[float]:
    """n evenly-spaced timestamps across the first `frac` of the clip.

    Same scheme as the current browser trainer's `extractFrames()`, kept
    identical so accuracy comparisons against the JS baseline sample the same
    moments in each clip.
    """
    if duration <= 0:
        return [0.0] * n
    denom = (n - 1) or 1
    return [(i / denom) * duration * frac for i in range(n)]


def read_frame(path: str, timestamp: float, size: int = FRAME_SIZE) -> np.ndarray:
    """Decode a single frame at `timestamp`, resized to size x size RGB, uint8 HWC."""
    cmd = [
        FFMPEG,
        "-ss", f"{timestamp:.3f}",
        "-i", path,
        "-frames:v", "1",
        "-vf", f"scale={size}:{size}",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-loglevel", "error",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    raw = proc.stdout
    expected = size * size * 3
    if len(raw) != expected:
        raise ValueError(
            f"expected {expected} bytes decoding {path} @ {timestamp:.3f}s, got {len(raw)}"
        )
    return np.frombuffer(raw, dtype=np.uint8).reshape(size, size, 3)


def extract_frames(path: str, n: int, size: int = FRAME_SIZE, frac: float = 0.95) -> list[np.ndarray]:
    """n evenly-spaced frames from a clip, each uint8 HWC RGB at size x size."""
    duration = get_duration(path)
    timestamps = sample_timestamps(duration, n, frac)
    return [read_frame(path, t, size) for t in timestamps]


def normalize(frame: np.ndarray) -> np.ndarray:
    """uint8 HWC RGB [0,255] -> float32 CHW, ImageNet-normalized, for the timm backbone."""
    x = frame.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(x, (2, 0, 1))
