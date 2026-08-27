"""Disk/backlog budget for Clipping Mode.

Clipping Mode harvests unlabeled clips unattended, so it needs a stop line.
Two rules, both fail-stop and neither destructive — nothing here ever deletes a
recording. The only deleter in the system stays the agent's segment pruner,
which touches scratch segments only.

1. Backlog cap: the number of recordings NOT present in labels.json. Labeled
   clips are excluded deliberately, so working through the labeling queue frees
   headroom automatically and the cap throttles the backlog rather than the
   archive.
2. Free-disk floor: an absolute floor under the recordings volume, so a full
   disk can never be the thing that stops capture.

Every filesystem call goes through asyncio.to_thread for the reason spelled out
in routers/recordings.py's module docstring: uvicorn runs one event loop, and a
synchronous directory walk on it stalls the MJPEG generator in routers/stream.py
along with every other in-flight request.
"""
import asyncio
import os
import shutil
from pathlib import Path

from app.lib.label_store import read_labels

GB = 1024 ** 3


def _scan(recordings_dir: Path) -> tuple[list[str], int]:
    """(recording filenames, total bytes). Deliberately not reusing
    routers/recordings.py::_scan_recordings — that builds ISO timestamps this
    never needs, and importing it here would make lib depend on a router."""
    names: list[str] = []
    total = 0
    try:
        with os.scandir(recordings_dir) as entries:
            for entry in entries:
                if not entry.name.endswith(".webm"):
                    continue
                names.append(entry.name)
                total += entry.stat().st_size
    except FileNotFoundError:
        return [], 0
    return names, total


def _free_bytes(recordings_dir: Path) -> int:
    try:
        return shutil.disk_usage(recordings_dir).free
    except OSError:
        # Can't read the volume — don't let that alone stop capture; the
        # backlog cap still applies.
        return -1


async def recordings_usage(*, recordings_dir: Path, max_unlabeled: int,
                           min_free_gb: float, labeled: set[str] | None = None) -> dict:
    """Current usage plus whether a clipping upload should be refused.

    max_unlabeled <= 0 disables the backlog cap; min_free_gb <= 0 disables the
    disk floor. `recordings_dir` is passed rather than imported so callers (and
    tests, which repoint routers/recordings.py's module globals) all agree on
    which directory is being measured. `labeled` defaults to the real
    labels.json.
    """
    names, total_bytes = await asyncio.to_thread(_scan, recordings_dir)
    free = await asyncio.to_thread(_free_bytes, recordings_dir)

    # A genuinely absent labels.json reads as {} (JsonStore.read returns the
    # default), which is correct: nothing is labeled yet. But a CORRUPT or
    # unreadable one raises, and treating that as "nothing is labeled" would
    # count the entire archive against the backlog cap, block the next upload,
    # and auto-stop the harvest with a false "cap reached" banner. A read error
    # says nothing about the backlog, so the backlog rule is skipped entirely;
    # the disk floor still applies, since that measurement is still valid.
    labels_readable = True
    if labeled is None:
        try:
            labeled = set(await read_labels())
        except Exception as err:
            print(f"clip_budget: cannot read labels ({err}) — backlog cap not enforced this check")
            labeled = set()
            labels_readable = False

    unlabeled = sum(1 for n in names if n not in labeled)
    min_free_bytes = int(min_free_gb * GB) if min_free_gb > 0 else 0

    blocked = False
    reason = None
    if labels_readable and max_unlabeled > 0 and unlabeled >= max_unlabeled:
        blocked = True
        reason = f"unlabeled clip cap reached ({unlabeled}/{max_unlabeled}) — label some clips to free room"
    elif min_free_bytes and 0 <= free < min_free_bytes:
        blocked = True
        reason = f"free disk below the {min_free_gb} GB floor ({free / GB:.1f} GB left)"

    return {
        "count": len(names),
        "unlabeled": unlabeled,
        "bytes": total_bytes,
        "freeBytes": free,
        "maxUnlabeled": max_unlabeled,
        "labelsReadable": labels_readable,
        "minFreeBytes": min_free_bytes,
        "blocked": blocked,
        "reason": reason,
    }
