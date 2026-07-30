"""Build a motion-parity fixture: an ordered sequence of raw 224x224x3 frames
sampled at 1.5s spacing (matching capture.mjs's AGENT_FRAME_SECONDS default)
from real recordings, covering idle/resting, a real motion burst, and a
synthesized lighting-style scene change. See notes/python-port-plan.md
Phase 3 step 1 for why this reads from server/recordings/ instead of tapping
the live camera (read-only access to server/, never touches capture.mjs or
any pm2 process).

Usage (from server_py/, runtime venv):
    .venv/Scripts/python.exe -m scripts.dump_agent_frames
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.frames import get_duration, read_frame

ROOT = Path(__file__).resolve().parent.parent.parent  # BunnyTracker/
RECORDINGS_DIR = ROOT / "server" / "recordings"
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

FRAME_INTERVAL = 1.5  # matches AGENT_FRAME_SECONDS default in capture.mjs

# idle/resting, then a real motion burst. Two separate clips, each internally
# continuous — the join between them is a deliberate state-reset boundary,
# recorded as a segment break, not blended into a false first-frame diff.
IDLE_CLIP = "recording-rabbitat-normal-07.webm"
# binky-pogo-01 alone was too short (only 3 samples at 1.5s spacing) for a
# meaningful motion-burst fixture; these 10s zoomies clips give more frames.
MOTION_CLIPS = [
    "recording-binky-compilation-01.webm",
    "recording-binky-compilation-02.webm",
    "recording-binky-compilation-03.webm",
]


def sample_clip_at_interval(filename: str) -> list[np.ndarray]:
    path = RECORDINGS_DIR / filename
    duration = get_duration(str(path))
    n = max(2, int(duration // FRAME_INTERVAL))
    timestamps = [i * FRAME_INTERVAL for i in range(n) if i * FRAME_INTERVAL < duration]
    return [read_frame(str(path), t) for t in timestamps]


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    idle_frames = sample_clip_at_interval(IDLE_CLIP)
    segments = [{"name": "idle", "source": IDLE_CLIP, "count": len(idle_frames)}]
    all_frames = list(idle_frames)

    for clip in MOTION_CLIPS:
        frames = sample_clip_at_interval(clip)
        segments.append({"name": "motion_burst", "source": clip, "count": len(frames)})
        all_frames += frames

    # Synthesized scene-wide-change (lighting-step) fixture: no real auto-exposure
    # step was identifiable in the on-disk clips, so append the brightest and
    # darkest sampled frames back-to-back as a deliberate whole-scene jump. This
    # is a fair test of the `lighting` path (hot cells >= 70% of the grid) even
    # though it isn't from a real exposure step, per the plan's fallback note.
    brightness = [f.mean() for f in all_frames]
    dark_idx = int(np.argmin(brightness))
    bright_idx = int(np.argmax(brightness))
    lighting_pair = [all_frames[dark_idx], all_frames[bright_idx]]
    segments.append({"name": "synthesized_lighting_step", "source": "synthesized (dark+bright frame pair)", "count": len(lighting_pair)})
    all_frames = all_frames + lighting_pair

    stacked = np.stack(all_frames)  # [T, 224, 224, 3] uint8
    np.save(FIXTURES_DIR / "motion_fixture_frames.npy", stacked)
    # Raw concatenated bytes too, so the Node ground-truth generator (no numpy
    # available) can read the exact same frames without an .npy parser.
    stacked.tofile(FIXTURES_DIR / "motion_fixture_frames.bin")

    import json
    (FIXTURES_DIR / "motion_fixture_segments.json").write_text(json.dumps(segments, indent=2))

    print(f"Wrote {stacked.shape[0]} frames to {FIXTURES_DIR / 'motion_fixture_frames.npy'} (and .bin)")
    for seg in segments:
        print(f"  {seg['name']}: {seg['count']} frames from {seg['source']}")


if __name__ == "__main__":
    main()
