"""Decode, augment and embed every labeled clip into the schema-2 cache
(training/cache.py has the layout and why it is split-agnostic).

Two passes. The frames pass decodes each clip once with ffmpeg and keeps the
raw uint8 frames under frames/; a later run reuses them, so changing the
augmentation policy or the backbone re-runs only the embed pass, not every
decode. The embed pass always runs and writes emb/ plus the manifest.

ffmpeg reads real container duration, sidestepping the browser's
`duration === Infinity` MediaRecorder bug class entirely (no <video> element
involved at all).

Usage (from server_py/, using the training venv):
    .venv-train/bin/python -m training.extract_frames [--seed N] [--eval-windows 8 12]
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.frames import extract_frames as ffmpeg_extract_frames
from shared.frames import normalize
from shared.labels import ETHOGRAM_VERSION, LABELS
from training.cache import DEFAULT_CACHE_DIR, SCHEMA, manifest_path
from training.dataset import AUG_COPIES, TRAIN_FRAMES, WINDOW, make_variants
from training.model import BACKBONE_NAME, build_backbone
from training.splits import group_key

ROOT = Path(__file__).resolve().parent.parent.parent  # BunnyTracker/
RECORDINGS_DIR = ROOT / "server" / "recordings"
LABELS_JSON = ROOT / "server" / "labels.json"

# Recorded in the manifest so a result can be traced to the augmentation that
# produced it. Change this string whenever make_variants changes.
AUG_POLICY = f"flip+jitter:copies={AUG_COPIES}"


@torch.no_grad()
def embed_frames(backbone, frames: np.ndarray) -> np.ndarray:
    """frames: [T, H, W, 3] uint8 -> embeddings [T, 1280] float32."""
    batch = np.stack([normalize(f) for f in frames])  # [T, 3, 224, 224]
    x = torch.from_numpy(batch).float()
    out = backbone(x)  # [T, 1280]
    return out.numpy()


def load_labeled_clips(labels_json: Path = LABELS_JSON) -> list[tuple[str, str]]:
    """Model-class clips only. `out_of_view` and unlabeled clips are skipped."""
    labels = json.loads(labels_json.read_text())
    return [(fn, label) for fn, label in sorted(labels.items()) if label in LABELS]


def cached_frames(path: Path, recording: Path, n: int) -> np.ndarray:
    """The frames tier: decode once, reuse while the shape still matches."""
    if path.is_file():
        frames = np.load(path)
        if frames.shape[0] == n and frames.dtype == np.uint8:
            return frames
    frames = np.stack(ffmpeg_extract_frames(str(recording), n=n))
    np.save(path, frames)
    return frames


def build_cache(*, cache_dir: Path, labels_json: Path, recordings_dir: Path, backbone,
                backbone_name: str = BACKBONE_NAME, seed: int = 0,
                eval_windows: list[int] | None = None) -> dict:
    eval_windows = sorted(set(eval_windows or [WINDOW]))
    cache_dir = Path(cache_dir)
    mpath = manifest_path(cache_dir)
    if mpath.is_file() and json.loads(mpath.read_text()).get("schema", 1) != SCHEMA:
        raise SystemExit(
            f"{mpath} is a schema-1 cache. Archive it rather than writing over it "
            f"(docs/pipeline.md, 'After an ethogram change'), then re-run."
        )
    frames_dir, emb_dir = cache_dir / "frames", cache_dir / "emb"
    frames_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    labels_bytes = Path(labels_json).read_bytes()
    clips = load_labeled_clips(labels_json)
    print(f"{len(clips)} labeled model-class clips")

    manifest = {
        "schema": SCHEMA,
        "clips": [],
        "window": WINDOW,
        "eval_windows": eval_windows,
        "train_frames": TRAIN_FRAMES,
        "backbone": backbone_name,
        "labels": list(LABELS),
        "labels_sha256": hashlib.sha256(labels_bytes).hexdigest(),
        "ethogram_version": ETHOGRAM_VERSION,
        "aug_policy": AUG_POLICY,
        "seed": seed,
    }
    for filename, label in clips:
        recording = Path(recordings_dir) / filename
        raw = cached_frames(frames_dir / f"{filename}.npy", recording, TRAIN_FRAMES)
        # Seeded per clip, not from one shared stream, so a clip's variants do
        # not change when another clip is added or removed.
        rng = random.Random(f"{seed}:{filename}")
        variant_names = []
        for k, variant in enumerate(make_variants(raw, rng)):
            name = f"{filename}.v{k}.npy"
            np.save(emb_dir / name, embed_frames(backbone, variant))
            variant_names.append(name)
        eval_names = {}
        for w in eval_windows:
            ev = cached_frames(frames_dir / f"{filename}.eval{w}.npy", recording, w)
            name = f"{filename}.eval{w}.npy"
            np.save(emb_dir / name, embed_frames(backbone, ev))
            eval_names[str(w)] = name
        manifest["clips"].append({
            "filename": filename, "label": label, "group": group_key(filename),
            "variants": variant_names, "eval": eval_names,
        })
        print(f"  {filename} ({label}) x{len(variant_names)} variants")

    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written to {mpath}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="seed for the augmentation variants")
    parser.add_argument("--eval-windows", type=int, nargs="+", default=[WINDOW],
                        help="window lengths to cache an eval sample for (the window sweep)")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--labels", type=Path, default=LABELS_JSON)
    parser.add_argument("--recordings-dir", type=Path, default=RECORDINGS_DIR)
    args = parser.parse_args()

    build_cache(
        cache_dir=args.cache_dir, labels_json=args.labels, recordings_dir=args.recordings_dir,
        backbone=build_backbone(pretrained=True), seed=args.seed, eval_windows=args.eval_windows,
    )


if __name__ == "__main__":
    main()
