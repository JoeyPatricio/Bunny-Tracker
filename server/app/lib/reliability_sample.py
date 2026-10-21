"""The blind re-code sample for intra-observer reliability.

Drawn once, stored, and never redrawn. Agreement cannot be retrofitted: once
the coder has seen which clips are in the sample, or the sample shifts as clips
are added, the re-code stops being a fixed, pre-registered subset.

Deterministic by construction. The pool is sorted before sampling, so the same
pool and seed give the same clips in any process on any machine, independent of
directory enumeration order.
"""
import math
import random

SAMPLE_SEED = 0
SAMPLE_FRACTION = 0.25


def sample_size(pool_size: int, fraction: float = SAMPLE_FRACTION) -> int:
    """ceil, not round: Python's round() is banker's rounding, so 0.25 * 10
    would give 2 and 0.25 * 14 would give 4, flipping direction with parity.
    ceil always means "at least the stated fraction"."""
    if pool_size <= 0:
        return 0
    return min(pool_size, math.ceil(pool_size * fraction))


def draw_sample(pool: list[str], seed: int = SAMPLE_SEED, fraction: float = SAMPLE_FRACTION) -> list[str]:
    ordered = sorted(set(pool))
    k = sample_size(len(ordered), fraction)
    return sorted(random.Random(seed).sample(ordered, k))
