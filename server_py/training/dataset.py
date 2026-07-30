"""Group-aware split, augmentation, and windowed embedding dataset.

Ported from client/src/components/TrainingStudio.jsx's groupKey/splitByGroup
(section 3.3: "reproduce the good parts... fix its known problems").
"""
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from shared.labels import LABELS

WINDOW = 8          # frames per training/eval window (matches the current model's FRAMES_PER_CLIP)
TRAIN_FRAMES = 16    # frames extracted per training clip, to allow sliding windows
TRAIN_WINDOW_STARTS = [0, 4, 8]  # sliding-window starts over TRAIN_FRAMES=16, window=8
AUG_COPIES = 2       # matches the JS trainer: original + mirror + (AUG_COPIES - 1) jittered


def group_key(filename: str) -> str:
    """Source-group key for a clip — line-for-line port of groupKey() in
    TrainingStudio.jsx. Clips cut from the same source video/session share
    scene, lighting, and rabbit; letting them straddle train/val would measure
    "have I seen this video" instead of generalization.
    """
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2})", filename)
    if m:
        return m.group(1)
    m = re.match(r"^recording-([a-z]+)-src(\d+)", filename, re.IGNORECASE)
    if m:
        return f"{m.group(1)}-src{m.group(2)}"
    m = re.match(r"^recording-([a-z]+)-([a-z]+)", filename, re.IGNORECASE)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return filename


def split_by_group(clips: list[tuple[str, str]], val_fraction: float = 0.2, seed: int = 0):
    """Group-aware stratified split at the clip level. Port of splitByGroup()."""
    targets = [LABELS.index(label) for _, label in clips]
    class_total = [0] * len(LABELS)
    for t in targets:
        class_total[t] += 1
    quota = [max(1, round(c * val_fraction)) for c in class_total]

    group_map: dict[str, list[int]] = {}
    for i, (filename, _) in enumerate(clips):
        group_map.setdefault(group_key(filename), []).append(i)

    group_list = list(group_map.keys())
    random.Random(seed).shuffle(group_list)

    val_class = [0] * len(LABELS)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for g in group_list:
        idxs = group_map[g]
        counts = [0] * len(LABELS)
        for i in idxs:
            counts[targets[i]] += 1
        needed = any(c > 0 and val_class[k] < quota[k] for k, c in enumerate(counts))
        fits = all(c == 0 or val_class[k] + c <= quota[k] + 1 for k, c in enumerate(counts))
        if needed and fits:
            val_idx.extend(idxs)
            for k, c in enumerate(counts):
                val_class[k] += c
        else:
            train_idx.extend(idxs)

    fallback = False
    if not val_idx or not train_idx:
        fallback = True
        idx = list(range(len(clips)))
        random.Random(seed).shuffle(idx)
        at = int(len(clips) * (1 - val_fraction))
        train_idx, val_idx = idx[:at], idx[at:]
        val_class = [0] * len(LABELS)
        for i in val_idx:
            val_class[targets[i]] += 1

    return {
        "train": [clips[i] for i in train_idx],
        "val": [clips[i] for i in val_idx],
        "groups": len(group_map),
        "val_per_class": val_class,
        "total_per_class": class_total,
        "fallback": fallback,
    }


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


class WindowDataset(Dataset):
    """Windows of cached per-frame embeddings -> (window, target).

    Reads pre-computed embeddings (see extract_frames.py), not raw pixels —
    the backbone is frozen, so embeddings only need to be computed once.
    """

    def __init__(self, entries: list[dict]):
        # entries: [{"embeddings": np.ndarray [T,1280], "label": str, "windows": [(start,end), ...]}]
        self.samples: list[tuple[np.ndarray, int]] = []
        for e in entries:
            target = LABELS.index(e["label"])
            for start, end in e["windows"]:
                self.samples.append((e["embeddings"][start:end], target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, target = self.samples[idx]
        return torch.from_numpy(window).float(), target


def class_weights(entries_train: list[dict]) -> torch.Tensor:
    """Inverse-frequency class weights over training clips (not windows), matching
    the JS trainer's per-clip-count weighting rather than per-window."""
    counts = [0] * len(LABELS)
    for e in entries_train:
        counts[LABELS.index(e["label"])] += 1
    total = sum(counts)
    weights = [total / (len(LABELS) * max(1, c)) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)
