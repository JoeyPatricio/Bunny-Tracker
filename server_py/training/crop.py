"""Static ROI crop for the `crop` cache view (docs/upgrade-plan.md, Phase 5).

One square box per clip, found from motion against a median background, then
applied to every frame of that clip's training and eval samples. The box is
re-decoded at higher resolution before cropping, so the animal gets more
pixels at the backbone's 224x224 input, not an upsampled 224 crop.

How a box is found:
  1. Background: per-pixel median of the pooled luma (shared.motion.pool_luma)
     over the frames-tier frames of every clip in the same source group
     (training.splits.group_key). Groups never straddle a fold, so no pixel
     from another fold's clips reaches this clip's crop. Labels are never read.
  2. Per frame: difference against the background, minus the frame's
     lower-middle median shift (illumination compensation, the same rule as
     shared.motion), thresholded at shared.motion.PIXEL_DIFF. A frame where
     the changed area covers LIGHTING_FRACTION of the grid is a lighting
     change or a bumped camera, not an animal, and counts as no blob.
  3. The largest 8-connected component of at least MIN_BLOB pooled pixels is
     that frame's blob. The box is the union of the blobs' bounding boxes,
     made square (in the 224x224 frame, which is the aspect the full-frame
     pipeline already feeds the backbone), widened by MARGIN on each side,
     at least MIN_SIDE, clamped inside the frame.

Fallback to full frame, recorded per clip in the manifest's "roi":
  - more than MAX_NO_BLOB of the clip's frames had no blob. A rabbit that sits
    still for a whole singleton clip is part of its own median background and
    is invisible to this method, so resting clips fall back more often than
    active ones. The fallback rate per class is printed at build time because
    it decides how much of `gru@crop` is actually cropped.
  - the box would cover at least FULL_SIDE of the frame, where cropping buys
    nothing.
A fallback clip reuses its full-view frames unchanged, so gru@crop and gru
see identical inputs for it.

Known limit: the illumination compensation removes a uniform offset between
a frame and the background. A lighting change part-way through a group's
clips moves the median background itself, and the box can then grow to
cover, or lock onto, where the animal used to be.

Deployment: the box here comes from the clip's own group. A runtime crop would
need a box from the camera's recent history and the same decode-crop-resize
chain in the agent. Neither exists, so a crop model cannot be exported yet;
crossval.py only runs crop as a paired `gru@crop` arm.

Training venv only (scipy, cv2 via albumentations).
"""
import cv2
import numpy as np
from scipy import ndimage

from shared.frames import FRAME_SIZE
from shared.frames import extract_frames as ffmpeg_extract_frames
from shared.motion import PIXEL_DIFF, POOLED_W, pool_luma

# Recorded in every crop manifest as "crop_params", so results stay
# attributable to the exact rule that made them.
HI_RES = 3 * FRAME_SIZE     # decode size before cropping
MIN_BLOB = 4                # pooled px (each 4x4 frame px) for a blob to count
LIGHTING_FRACTION = 0.5     # changed fraction of the pooled grid that means "not an animal"
MARGIN = 0.15               # of the box side, added on each side
MIN_SIDE = 0.3              # of the frame side
FULL_SIDE = 0.9             # box at least this big: use the full frame
MAX_NO_BLOB = 0.5           # fraction of frames without a blob before falling back

CROP_PARAMS = {"hi_res": HI_RES, "min_blob": MIN_BLOB, "lighting_fraction": LIGHTING_FRACTION,
               "margin": MARGIN, "min_side": MIN_SIDE, "full_side": FULL_SIDE,
               "max_no_blob": MAX_NO_BLOB, "pixel_diff": PIXEL_DIFF, "background": "group median"}


def background(frame_sets: list[np.ndarray]) -> np.ndarray:
    """Median pooled luma over every frame of every clip given. [56, 56]."""
    return np.median(np.stack([pool_luma(f) for frames in frame_sets for f in frames]), axis=0)


def frame_blob(frame: np.ndarray, bg: np.ndarray) -> tuple[int, int, int, int] | None:
    """(y0, x0, y1, x1) in pooled px, end-exclusive, of the largest moving
    component in one frame, or None."""
    diffs = pool_luma(frame) - bg
    shift = np.sort(diffs, axis=None)[diffs.size >> 1]  # lower-middle, as shared.motion
    mask = np.abs(diffs - shift) > PIXEL_DIFF
    if mask.mean() >= LIGHTING_FRACTION:
        return None
    labelled, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return None
    sizes = np.bincount(labelled.ravel())[1:]
    biggest = int(sizes.argmax())
    if sizes[biggest] < MIN_BLOB:
        return None
    ys, xs = ndimage.find_objects(labelled)[biggest]
    return ys.start, xs.start, ys.stop, xs.stop


def estimate_roi(frames: np.ndarray, bg: np.ndarray) -> dict:
    """Box for one clip as fractions of the frame, [x0, y0, x1, y1], or
    box None with the fallback reason."""
    blobs = [b for b in (frame_blob(f, bg) for f in frames) if b is not None]
    no_blob = 1 - len(blobs) / len(frames)
    out = {"no_blob_fraction": round(no_blob, 4), "box": None, "fallback": None}
    if no_blob > MAX_NO_BLOB:
        out["fallback"] = f"no motion blob in {no_blob:.0%} of frames"
        return out
    y0, x0 = min(b[0] for b in blobs), min(b[1] for b in blobs)
    y1, x1 = max(b[2] for b in blobs), max(b[3] for b in blobs)
    side = max(y1 - y0, x1 - x0) / POOLED_W
    side = max(MIN_SIDE, side * (1 + 2 * MARGIN))
    if side >= FULL_SIDE:
        out["fallback"] = f"box side {side:.2f} of the frame"
        return out
    cx, cy = (x0 + x1) / 2 / POOLED_W, (y0 + y1) / 2 / POOLED_W
    left = min(max(cx - side / 2, 0.0), 1.0 - side)
    top = min(max(cy - side / 2, 0.0), 1.0 - side)
    out["box"] = [round(left, 4), round(top, 4), round(left + side, 4), round(top + side, 4)]
    return out


def crop_frames(frames: np.ndarray, box: list[float]) -> np.ndarray:
    """Crop [T, S, S, 3] uint8 frames to `box` and resize to FRAME_SIZE."""
    size = frames.shape[1]
    x0, y0, x1, y1 = (int(round(v * size)) for v in box)
    return np.stack([cv2.resize(np.ascontiguousarray(f[y0:y1, x0:x1]), (FRAME_SIZE, FRAME_SIZE),
                                interpolation=cv2.INTER_AREA) for f in frames])


def cropped_sample(recording: str, n: int, roi: dict, full: np.ndarray) -> np.ndarray:
    """The crop-view sample of n frames: the same timestamps as the full-view
    sample `full`, decoded at HI_RES and cropped, or `full` itself on fallback."""
    if roi["box"] is None:
        return full
    return crop_frames(np.stack(ffmpeg_extract_frames(recording, n=n, size=HI_RES)), roi["box"])
