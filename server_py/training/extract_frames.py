"""Extract frames from server/recordings/*.webm, embed with the frozen
backbone, and cache per-clip embeddings to disk (section 3.3 / Phase 1 step 1).

ffmpeg reads real container duration, sidestepping the browser's
`duration === Infinity` MediaRecorder bug class entirely (no <video> element
involved at all).

Usage (from server_py/, using the training venv):
    .venv-train/Scripts/python.exe -m training.extract_frames
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.frames import extract_frames as ffmpeg_extract_frames, normalize
from shared.labels import LABELS
from training.dataset import (
    TRAIN_FRAMES,
    TRAIN_WINDOW_STARTS,
    WINDOW,
    make_variants,
    split_by_group,
)
from training.model import build_backbone

ROOT = Path(__file__).resolve().parent.parent.parent  # BunnyTracker/
RECORDINGS_DIR = ROOT / "server" / "recordings"
LABELS_JSON = ROOT / "server" / "labels.json"
CACHE_DIR = Path(__file__).resolve().parent.parent / "models" / "cache"
MANIFEST_PATH = CACHE_DIR / "manifest.json"


@torch.no_grad()
def embed_frames(backbone, frames: np.ndarray) -> np.ndarray:
    """frames: [T, H, W, 3] uint8 -> embeddings [T, 1280] float32."""
    batch = np.stack([normalize(f) for f in frames])  # [T, 3, 224, 224]
    x = torch.from_numpy(batch).float()
    out = backbone(x)  # [T, 1280]
    return out.numpy()


def load_labeled_clips() -> list[tuple[str, str]]:
    labels = json.loads(LABELS_JSON.read_text())
    return [(fn, label) for fn, label in labels.items() if label in LABELS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    clips = load_labeled_clips()
    print(f"{len(clips)} labeled clips")

    split = split_by_group(clips, val_fraction=args.val_fraction, seed=args.seed)
    print(
        f"Group-aware split: {split['groups']} source groups -> "
        f"{len(split['train'])} train clips / {len(split['val'])} val clips"
        + (" (fallback random split)" if split["fallback"] else "")
    )
    for k, label in enumerate(LABELS):
        print(f"  {label}: val {split['val_per_class'][k]}/{split['total_per_class'][k]}")

    backbone = build_backbone(pretrained=True)
    rng = random.Random(args.seed)

    manifest = {"train": [], "val": [], "window": WINDOW, "train_frames": TRAIN_FRAMES}

    for filename, label in split["val"]:
        path = RECORDINGS_DIR / filename
        frames = ffmpeg_extract_frames(str(path), n=WINDOW)
        emb = embed_frames(backbone, np.stack(frames))  # [WINDOW, 1280]
        out_path = CACHE_DIR / f"{filename}.val.npy"
        np.save(out_path, emb)
        manifest["val"].append({
            "filename": filename,
            "label": label,
            "embeddings": out_path.name,
            "windows": [[0, WINDOW]],
        })
        print(f"  val   {filename} ({label})")

    for filename, label in split["train"]:
        path = RECORDINGS_DIR / filename
        raw_frames = np.stack(ffmpeg_extract_frames(str(path), n=TRAIN_FRAMES))
        variants = make_variants(raw_frames, rng)
        for vi, variant in enumerate(variants):
            emb = embed_frames(backbone, variant)  # [TRAIN_FRAMES, 1280]
            out_path = CACHE_DIR / f"{filename}.train{vi}.npy"
            np.save(out_path, emb)
            windows = [[s, s + WINDOW] for s in TRAIN_WINDOW_STARTS]
            manifest["train"].append({
                "filename": filename,
                "label": label,
                "variant": vi,
                "embeddings": out_path.name,
                "windows": windows,
            })
        print(f"  train {filename} ({label}) x{len(variants)} variants")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written to {MANIFEST_PATH}")
    print(
        f"Total: {len(manifest['val'])} val clips (1 window each), "
        f"{len(manifest['train'])} train clip-variants "
        f"({len(manifest['train']) * len(TRAIN_WINDOW_STARTS)} windows)"
    )


if __name__ == "__main__":
    main()
