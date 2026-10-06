"""Prospective temporal holdout (docs/upgrade-plan.md Phase 3, docs/ethogram.md).

The 112 clips coded before the cut are the cross-validation pool. Every clip
captured on or after HOLDOUT_CUTOFF is a holdout clip: it can be coded, but
its label goes to labels.holdout.json, never labels.json. Every training
consumer reads labels.json only, so a holdout label cannot reach a fold, a
standardization buffer, a threshold or a selection decision. The file is
opened once, at paper freeze, to measure drift over time, which is the
deployment failure a fixed camera in a changing room actually has.

The boundary is inclusive: a clip captured exactly at the cut is holdout.

Capture time comes from the filename (recording names carry
toISOString()-style UTC stamps, `2026-09-23T13-53-40-795Z`), falling back to
the file's mtime in RECORDINGS_DIR. A clip with neither is treated as holdout
while a cut is set. That is the conservative direction: a mis-sorted holdout
clip costs one clip of training data, a mis-sorted training clip contaminates
the only test the paper has of time.
"""
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from app.config import HOLDOUT_CUTOFF_RAW

STAMP_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})-(\d{3})Z")


def parse_cutoff(raw: str | None) -> datetime | None:
    """`YYYY-MM-DD` (midnight UTC) or a full ISO-8601 timestamp. Unset or
    blank means no cut yet: today's behaviour, everything goes to labels.json.

    A bad value is not fatal, same rule as config._numberish, but it is loud,
    because silently running with no cut would put post-cut clips in training.
    """
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    try:
        if len(text) == 10:
            dt = datetime.strptime(text, "%Y-%m-%d")
        else:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        print(f"config: HOLDOUT_CUTOFF={raw!r} is not a date; holdout routing is OFF until it is fixed")
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def capture_time(filename: str, recordings_dir: Path) -> datetime | None:
    m = STAMP_RE.search(filename)
    if m:
        y, mo, d, h, mi, s, ms = (int(x) for x in m.groups())
        return datetime(y, mo, d, h, mi, s, ms * 1000, tzinfo=UTC)
    try:
        return datetime.fromtimestamp(os.stat(recordings_dir / filename).st_mtime, tz=UTC)
    except OSError:
        return None


def is_holdout(filename: str, cutoff: datetime | None, recordings_dir: Path) -> bool:
    if cutoff is None:
        return False
    captured = capture_time(filename, recordings_dir)
    return captured is None or captured >= cutoff


# Parsed once at import. Routers read it as `holdout.HOLDOUT_CUTOFF` (module
# attribute, not a from-import) so tests can repoint it.
HOLDOUT_CUTOFF = parse_cutoff(HOLDOUT_CUTOFF_RAW)


def describe_cutoff() -> str:
    if HOLDOUT_CUTOFF is None:
        return ("holdout: HOLDOUT_CUTOFF is unset (placeholder). Every label goes to labels.json; "
                "set it in server/.env to start the prospective temporal holdout.")
    return (f"holdout: clips captured on or after {HOLDOUT_CUTOFF.isoformat()} are coded into "
            f"labels.holdout.json, never labels.json.")
