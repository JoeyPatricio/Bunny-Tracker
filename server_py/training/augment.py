"""Augmentation policies for the embedding cache.

A policy turns one clip's raw training frames into a list of variants, k=0
always the untouched clip. Every transform is sampled ONCE per variant and then
applied identically to every frame of that clip, so the temporal signal the
head learns from is motion of the rabbit, not frame-to-frame augmentation
noise. That is the constraint dataset.make_variants hand-implements for flips
and jitter, and what albumentations' ReplayCompose gives for everything else:
sample on frame 0, replay the recorded parameters on the rest.

Policies:
    legacy   dataset.make_variants unchanged: identity, mirror, and
             AUG_COPIES - 1 brightness/contrast jitters (optionally mirrored).
             3 variants. Byte-identical to the pre-Phase-4 cache.
    albu_v1  identity, mirror, then 6 ReplayCompose variants adding mild
             rotation/scale/shift, perspective, motion blur and CoarseDropout
             (occlusion) on top of the legacy flip and colour jitter.
             8 variants.

UNEVALUATED. albu_v1 exists to be measured, not adopted: run it as a paired
arm in crossval.py (--aug-policies legacy albu_v1) once the v2 recoding pass
has produced labels, and keep whichever wins on identical folds.

Why the albu_v1 magnitudes are conservative. Augmentation that changes what
the clip depicts is label noise, not regularisation:
  - Rotation: a rabbit rotated far enough stops looking like it is lying on
    the floor, so a rotated `resting_lying` clip can teach `resting_sitting`
    or `rearing`, which are distinguished by body-axis angle. Capped at +/-8
    degrees, about what a bumped camera does.
  - Motion blur: `locomotion` and `locomotion_rapid` are separated by speed,
    and blur is the visual signature of speed. Blurring a slow clip forges
    evidence for the fast class. Kept to the smallest kernel (3 px) and
    applied to a minority of variants.
  - Perspective and scale are small enough that the cage geometry stays
    recognisable; CoarseDropout holes are 8-15% of the frame, roughly a food
    bowl or a hand, not the rabbit.
Each policy is seeded per clip from the caller's rng, so a clip's variants do
not change when another clip is added or removed.
"""
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from training.dataset import AUG_COPIES, flip_frames, make_variants

ALBU_V1_EXTRA = 6  # ReplayCompose variants after identity and mirror


@dataclass(frozen=True)
class Policy:
    name: str
    detail: str
    n_variants: int
    make: Callable[[np.ndarray, random.Random], list[np.ndarray]]


def _albu_v1_transform():
    # Imported here, not at module top: albumentations is a training-venv
    # dependency and the legacy policy must not need it.
    import albumentations as A

    return A.ReplayCompose([
        A.HorizontalFlip(p=0.5),
        # Legacy jitter was +/-25 levels brightness and 0.85-1.2x contrast.
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.15, 0.2), p=0.8),
        A.Affine(rotate=(-8, 8), scale=(0.95, 1.05), translate_percent=(-0.04, 0.04),
                 border_mode=0, fill=0, p=0.7),
        A.Perspective(scale=(0.02, 0.05), p=0.3),
        A.MotionBlur(blur_limit=(3, 3), p=0.2),
        A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.08, 0.15),
                        hole_width_range=(0.08, 0.15), fill=128, p=0.3),
    ])


def replay_clip(transform, frames: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    """Sample `transform` once (seeded) on frame 0, replay it on every frame.
    Returns the augmented clip and the replay record."""
    import albumentations as A

    transform.set_random_seed(seed)
    first = transform(image=frames[0])
    replay = first["replay"]
    out = [first["image"]] + [A.ReplayCompose.replay(replay, image=f)["image"] for f in frames[1:]]
    return np.stack(out).astype(np.uint8), replay


def make_variants_albu_v1(frames: np.ndarray, rng: random.Random) -> list[np.ndarray]:
    transform = _albu_v1_transform()
    variants = [frames, flip_frames(frames)]
    for _ in range(ALBU_V1_EXTRA):
        variant, _ = replay_clip(transform, frames, rng.randrange(2**31))
        variants.append(variant)
    return variants


POLICIES: dict[str, Policy] = {
    "legacy": Policy("legacy", f"flip+jitter:copies={AUG_COPIES}", AUG_COPIES + 1, make_variants),
    "albu_v1": Policy(
        "albu_v1",
        "identity, mirror, 6x ReplayCompose(hflip 0.5, brightness/contrast, affine rot+-8 "
        "scale 0.95-1.05 shift 4%, perspective 0.02-0.05 p0.3, motion blur 3px p0.2, "
        "coarse dropout 1-2 holes 8-15% p0.3); UNEVALUATED",
        2 + ALBU_V1_EXTRA,
        make_variants_albu_v1,
    ),
}
DEFAULT_POLICY = "legacy"


def get_policy(name: str) -> Policy:
    if name not in POLICIES:
        raise SystemExit(f"Unknown augmentation policy {name!r}; choose from {sorted(POLICIES)}.")
    return POLICIES[name]
