"""Port of poolLuma()/motionLevel() from server/agent/capture.mjs.

Percentage of pooled pixels whose frame-to-frame difference exceeds
PIXEL_DIFF, after subtracting the frame's median shift (illumination
compensation), scored over an 8x8 grid so a scene-wide change (lighting) is
distinguished from localized motion (an animal). See capture.mjs's own
comments for the full rationale; this is a line-for-line port, vectorized
with numpy but WITHOUT reordering the arithmetic: pool, diff vs previous,
subtract median, threshold, grid-score.
"""
import numpy as np

PIXEL_DIFF = 15               # on pooled luma
POOL = 4                      # 4x4 pixel blocks -> 56x56 luma grid
POOLED_W = 224 // POOL         # 56
GRID = 8                       # 8x8 cells over the pooled grid
CELL_W = POOLED_W // GRID      # 7 pooled px per cell side
CELL_HOT = 0.25                # fraction of a cell changed to call it hot
LIGHTING_CELLS = int(GRID * GRID * 0.7)  # hot cells => scene-wide change


def pool_luma(frame: np.ndarray) -> np.ndarray:
    """frame: [224, 224, 3] uint8 RGB -> [56, 56] float32 pooled luma."""
    # rough luma (r + 2g + b) / 4; exact weights don't matter for differencing
    luma = (frame[:, :, 0].astype(np.float32)
            + 2 * frame[:, :, 1].astype(np.float32)
            + frame[:, :, 2].astype(np.float32)) / 4
    pooled = luma.reshape(POOLED_W, POOL, POOLED_W, POOL).sum(axis=(1, 3))
    return pooled / (POOL * POOL)


class MotionDetector:
    """Stateful across calls (holds the previous pooled frame), matching the
    JS module-level `let prevPooled`. One instance per independent frame
    stream — do not share across unrelated clips/sessions."""

    def __init__(self):
        self.prev_pooled: np.ndarray | None = None

    def motion_level(self, frame: np.ndarray) -> dict:
        pooled = pool_luma(frame)
        if self.prev_pooled is None:
            self.prev_pooled = pooled
            return {"level": 0.0, "hotCells": 0, "lighting": False}

        diffs = pooled - self.prev_pooled
        self.prev_pooled = pooled

        # Illumination compensation: the global brightness shift. Deliberately
        # NOT np.median() — for an even-length array that averages the two
        # middle elements, but the JS original picks a single index
        # (`sorted[n >> 1]`), a "lower-middle" value. Match that exactly
        # rather than the statistically-conventional median.
        n = pooled.size
        median = np.sort(diffs, axis=None)[n >> 1]

        changed_mask = np.abs(diffs - median) > PIXEL_DIFF
        changed = int(changed_mask.sum())

        # Grid-score: reshape the POOLED_W x POOLED_W changed mask into
        # GRID x GRID cells of CELL_W x CELL_W each, sum per cell.
        cell_changed = changed_mask.reshape(GRID, CELL_W, GRID, CELL_W).sum(axis=(1, 3))
        hot_cells = int((cell_changed >= CELL_W * CELL_W * CELL_HOT).sum())

        lighting = hot_cells >= LIGHTING_CELLS
        level = 0.0 if lighting else (changed / n) * 100
        return {"level": level, "hotCells": hot_cells, "lighting": lighting}
