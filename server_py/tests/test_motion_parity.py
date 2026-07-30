"""Motion parity hard gate (plan section 5): for every fixture frame, `level`
must be within 0.1 percentage points, `hotCells` and `lighting` must match
exactly, against the JS ground truth in tests/fixtures/.

Usage (from server_py/, runtime venv):
    .venv/Scripts/python.exe -m tests.test_motion_parity
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.motion import MotionDetector

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
LEVEL_TOLERANCE = 0.1


def main():
    frames = np.load(FIXTURES_DIR / "motion_fixture_frames.npy")
    ground_truth = json.loads((FIXTURES_DIR / "motion_fixture_ground_truth.json").read_text())

    if len(frames) != len(ground_truth):
        print(f"FAIL: frame count {len(frames)} != ground truth count {len(ground_truth)}")
        sys.exit(1)

    detector = MotionDetector()
    failures = []
    for i, frame in enumerate(frames):
        result = detector.motion_level(frame)
        gt = ground_truth[i]

        level_ok = abs(result["level"] - gt["level"]) <= LEVEL_TOLERANCE
        cells_ok = result["hotCells"] == gt["hotCells"]
        lighting_ok = result["lighting"] == gt["lighting"]

        if not (level_ok and cells_ok and lighting_ok):
            failures.append({
                "frame": i,
                "python": result,
                "ground_truth": gt,
                "level_ok": level_ok, "cells_ok": cells_ok, "lighting_ok": lighting_ok,
            })

    print(f"Checked {len(frames)} frames.")
    if failures:
        print(f"FAIL: {len(failures)} frame(s) mismatched:")
        for f in failures[:20]:
            print(f"  frame {f['frame']}: python={f['python']} ground_truth={f['ground_truth']}"
                  f" (level_ok={f['level_ok']} cells_ok={f['cells_ok']} lighting_ok={f['lighting_ok']})")
        sys.exit(1)
    else:
        print("PASS: all frames within tolerance (level +/-0.1pp, hotCells exact, lighting exact)")


if __name__ == "__main__":
    main()
