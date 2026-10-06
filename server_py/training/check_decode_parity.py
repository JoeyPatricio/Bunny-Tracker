"""Byte parity of the single-process decode against the per-frame one, on real footage.

shared.frames.extract_frames moved from one ffmpeg process per frame to one
per clip (docs/upgrade-plan.md, Phase 3). The motion-parity fixtures and the
JS-parity story both assume the same timestamps decode to the same bytes, so
this checks every .webm in server/recordings/ at both sample counts the
pipeline uses (WINDOW for eval clips, TRAIN_FRAMES for training clips) and
fails on any byte that differs. Read-only.

Needs real recordings, so it is local_data in ci/test_scripts.py.

Usage (from server_py/, either venv):
    .venv-train/bin/python training/check_decode_parity.py [--limit N]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.frames import extract_frames, extract_frames_single
from tests.checks import check, finish

ROOT = Path(__file__).resolve().parent.parent.parent
RECORDINGS_DIR = ROOT / "server" / "recordings"

# Literals rather than an import from training.dataset, which pulls in torch;
# this script should also run in the runtime venv. Kept in step with
# dataset.WINDOW and dataset.TRAIN_FRAMES.
SAMPLE_COUNTS = (8, 16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="check only the first N clips")
    args = parser.parse_args()

    clips = sorted(RECORDINGS_DIR.glob("*.webm"))
    if args.limit:
        clips = clips[: args.limit]
    check("recordings present", bool(clips), f"no .webm files in {RECORDINGS_DIR}")

    mismatched: list[str] = []
    errors: list[str] = []
    t_single = t_multi = 0.0
    for path in clips:
        for n in SAMPLE_COUNTS:
            try:
                t0 = time.perf_counter()
                old = extract_frames_single(str(path), n)
                t1 = time.perf_counter()
                new = extract_frames(str(path), n)
                t2 = time.perf_counter()
            except ValueError as err:
                errors.append(f"{path.name} n={n}: {err}")
                continue
            t_single += t1 - t0
            t_multi += t2 - t1
            bad = [i for i, (a, b) in enumerate(zip(old, new, strict=True)) if not np.array_equal(a, b)]
            if bad:
                mismatched.append(f"{path.name} n={n}: frames {bad}")

    print(f"  {len(clips)} clips x {len(SAMPLE_COUNTS)} sample counts; "
          f"per-frame decode {t_single:.1f}s, single-process {t_multi:.1f}s")
    check("every clip decodes on both paths", not errors, "; ".join(errors[:5]))
    check("single-process decode is byte-identical to per-frame decode",
          not mismatched, "; ".join(mismatched[:5]))
    finish("decode parity")


if __name__ == "__main__":
    main()
