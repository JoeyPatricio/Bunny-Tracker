"""Millisecond-precision UTC ISO timestamps matching JS's Date.toISOString()
exactly (YYYY-MM-DDTHH:MM:SS.mmmZ), used everywhere a Node route returns one.
"""
from datetime import datetime, timezone


def to_iso_millis(timestamp: float) -> str:
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def filename_timestamp() -> str:
    """Matches JS's `new Date().toISOString().replace(/[:.]/g, '-')`, e.g.
    2026-07-23T22-15-30-123Z — used for recording/import filenames."""
    return to_iso_millis(datetime.now(timezone.utc).timestamp()).replace(":", "-").replace(".", "-")
