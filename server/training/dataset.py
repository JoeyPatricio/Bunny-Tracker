"""Clip augmentation, frame counts, and training class weights."""
import random

import numpy as np
import torch

from shared.labels import LABELS

WINDOW = 8          # frames per training/eval window (matches the current model's FRAMES_PER_CLIP)
TRAIN_FRAMES = 16    # frames extracted per training clip, to allow sliding windows
AUG_COPIES = 2       # matches the JS trainer: original + mirror + (AUG_COPIES - 1) jittered


# ── Frame augmentation (training clips only), applied consistently across a
# whole clip's frames so the temporal signal isn't scrambled — port of
# flipImageData/jitterImageData/makeVariants. ──────────────────────────────

def flip_frames(frames: np.ndarray) -> np.ndarray:
    """[N, H, W, 3] -> horizontally mirrored."""
    return frames[:, :, ::-1, :].copy()


def jitter_frames(frames: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    x = frames.astype(np.float32)
    x = (x - 128.0) * contrast + 128.0 + brightness
    return np.clip(x, 0, 255).astype(np.uint8)


def make_variants(frames: np.ndarray, rng: random.Random) -> list[np.ndarray]:
    """Original + mirror + (AUG_COPIES - 1) jittered (optionally also mirrored) variants."""
    variants = [frames, flip_frames(frames)]
    while len(variants) < AUG_COPIES + 1:
        brightness = rng.random() * 50 - 25    # +/-25 levels
        contrast = 0.85 + rng.random() * 0.35  # 0.85-1.2x
        also_flip = rng.random() < 0.5
        v = jitter_frames(frames, brightness, contrast)
        if also_flip:
            v = flip_frames(v)
        variants.append(v)
    return variants


def class_weights(entries_train: list[dict]) -> torch.Tensor:
    """Inverse-frequency class weights over training clips (not windows), matching
    the JS trainer's per-clip-count weighting rather than per-window."""
    counts = [0] * len(LABELS)
    for e in entries_train:
        counts[LABELS.index(e["label"])] += 1
    total = sum(counts)
    weights = [total / (len(LABELS) * max(1, c)) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)
