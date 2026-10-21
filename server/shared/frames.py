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


def read_frame_single(path: str, timestamp: float, size: int = FRAME_SIZE) -> np.ndarray:
    """Decode a single frame at `timestamp`, resized to size x size RGB, uint8 HWC.

    One ffmpeg process per frame. This was the only decode path until Phase 3;
    it stays as the reference that read_frames() is checked against
    (training/check_decode_parity.py, training/test_cache_v2.py).
    """
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


def _ms(t: float) -> str:
    # read_frame_single passes the seek target as "%.3f", so ffmpeg sees it
    # rounded to the millisecond. Selecting on the identical string is what
    # keeps the two paths picking the same frame.
    return f"{t:.3f}"


_SHOWINFO_RE = re.compile(r"\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:(-?[0-9.e+-]+)")


def read_frames(path: str, timestamps: list[float], size: int = FRAME_SIZE) -> list[np.ndarray]:
    """Decode the frames at every timestamp in ONE ffmpeg process.

    Byte-identical to [read_frame_single(path, t) for t in timestamps], about
    8x faster. read_frame_single's accurate input seek (-ss before -i) returns
    the first frame whose time is at or after the target. The select filter
    below keeps exactly those frames: for each target T, the frame with
    t >= T whose predecessor was < T (or which has no predecessor). Selection
    happens before scale, the scale filter and pix_fmt are unchanged, and
    swscale works per frame with no temporal state, so each selected frame
    goes through the identical conversion. Two targets that land on the same
    frame are both served by it, as the per-frame path would.

    -fps_mode passthrough stops ffmpeg duplicating or dropping frames to hit a
    constant output rate, and showinfo reports each selected frame's time so
    targets can be mapped back to frames.
    """
    targets = [_ms(t) for t in timestamps]
    terms = [f"gte(t\\,{t})*(isnan(prev_t)+lt(prev_t\\,{t}))" for t in sorted(set(targets), key=float)]
    cmd = [
        FFMPEG,
        "-i", path,
        "-vf", f"select='{'+'.join(terms)}',scale={size}:{size},showinfo",
        "-fps_mode", "passthrough",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-loglevel", "info",
        "-nostats",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    frame_bytes = size * size * 3
    times = [float(m.group(1)) for m in _SHOWINFO_RE.finditer(proc.stderr.decode(errors="replace"))]
    raw = proc.stdout
    if proc.returncode != 0 or len(raw) != len(times) * frame_bytes or not times:
        raise ValueError(
            f"decoding {path}: ffmpeg exit {proc.returncode}, {len(raw)} bytes for "
            f"{len(times)} selected frame(s)"
        )
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(len(times), size, size, 3)
    out = []
    for t in targets:
        # Tolerance only for showinfo printing pts_time with limited digits.
        idx = next((i for i, ft in enumerate(times) if ft >= float(t) - 1e-6), None)
        if idx is None:
            raise ValueError(f"no frame at or after {t}s in {path}")
        out.append(frames[idx])
    return out


def extract_frames_single(path: str, n: int, size: int = FRAME_SIZE, frac: float = 0.95) -> list[np.ndarray]:
    """The pre-Phase-3 extract_frames: one ffmpeg process per frame. Parity reference only."""
    duration = get_duration(path)
    timestamps = sample_timestamps(duration, n, frac)
    return [read_frame_single(path, t, size) for t in timestamps]


def extract_frames(path: str, n: int, size: int = FRAME_SIZE, frac: float = 0.95) -> list[np.ndarray]:
    """n evenly-spaced frames from a clip, each uint8 HWC RGB at size x size.

    Two ffmpeg processes per clip (duration probe, then read_frames), down from
    n + 1. Byte-identical to extract_frames_single.
    """
    duration = get_duration(path)
    timestamps = sample_timestamps(duration, n, frac)
    return read_frames(path, timestamps, size)


def normalize(frame: np.ndarray) -> np.ndarray:
    """uint8 HWC RGB [0,255] -> float32 CHW, ImageNet-normalized, for the timm backbone."""
    x = frame.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(x, (2, 0, 1))
