"""Port of server/lib/backupLabels.js. Back up now, then once every 24h."""
import asyncio
import json
import re
from datetime import UTC, datetime

from app.config import LABELS_BACKUP_DIR, SERVER_DIR

KEEP = 7  # retain this many daily backups per file

# labels.json plus its Phase 2 sidecars. The reliability re-code is the one
# that matters most: intra-observer agreement cannot be retrofitted once the
# coder has seen their own answers, so a lost labels.reliability.json cannot
# be recoded, only abandoned. Each file gets its own daily series and pruning,
# so labels.json backups keep their existing labels-YYYY-MM-DD.json names.
BACKED_UP = (
    "labels.json",
    "labels.meta.json",
    "labels.reliability.json",
    "labels.reliability_sample.json",
)


def _day_re(stem: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(stem)}-\d{{4}}-\d{{2}}-\d{{2}}\.json$")


async def _make_backup() -> None:
    for name in BACKED_UP:
        await _backup_one(SERVER_DIR / name)


async def _backup_one(src) -> None:
    stem = src.name.removesuffix(".json")
    day_re = _day_re(stem)
    try:
        raw = await asyncio.to_thread(src.read_text, encoding="utf-8")
        text = raw.lstrip("﻿").strip() or "{}"
        obj = json.loads(text)
        if len(obj) == 0:
            return  # don't snapshot an empty/blank file over a good one

        await asyncio.to_thread(LABELS_BACKUP_DIR.mkdir, parents=True, exist_ok=True)
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        dest = LABELS_BACKUP_DIR / f"{stem}-{day}.json"
        await asyncio.to_thread(dest.write_text, raw, encoding="utf-8")  # one per day

        files = sorted(f.name for f in LABELS_BACKUP_DIR.iterdir() if day_re.match(f.name))
        for name in files[: max(0, len(files) - KEEP)]:
            try:
                await asyncio.to_thread((LABELS_BACKUP_DIR / name).unlink)
            except OSError:
                pass
    except (OSError, json.JSONDecodeError):
        pass  # missing/corrupt file — skip this cycle


async def _backup_loop() -> None:
    while True:
        await asyncio.sleep(24 * 60 * 60)
        await _make_backup()


def start_label_backups() -> asyncio.Task:
    """Back up immediately, then schedule the recurring loop. Returns the
    background task handle so the caller can keep a reference (FastAPI
    otherwise garbage-collects fire-and-forget tasks mid-flight)."""
    asyncio.ensure_future(_make_backup())
    return asyncio.ensure_future(_backup_loop())
