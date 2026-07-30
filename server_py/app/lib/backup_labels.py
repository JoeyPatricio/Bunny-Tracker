"""Port of server/lib/backupLabels.js. Back up now, then once every 24h."""
import asyncio
import json
import re
from datetime import datetime, timezone

from app.config import LABELS_BACKUP_DIR, SERVER_DIR

LABELS_FILE = SERVER_DIR / "labels.json"
KEEP = 7  # retain this many daily backups
DAY_RE = re.compile(r"^labels-\d{4}-\d{2}-\d{2}\.json$")


async def _make_backup() -> None:
    try:
        raw = await asyncio.to_thread(LABELS_FILE.read_text, encoding="utf-8")
        text = raw.lstrip("﻿").strip() or "{}"
        obj = json.loads(text)
        if len(obj) == 0:
            return  # don't snapshot an empty/blank file over a good one

        await asyncio.to_thread(LABELS_BACKUP_DIR.mkdir, parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        dest = LABELS_BACKUP_DIR / f"labels-{day}.json"
        await asyncio.to_thread(dest.write_text, raw, encoding="utf-8")  # one per day

        files = sorted(f.name for f in LABELS_BACKUP_DIR.iterdir() if DAY_RE.match(f.name))
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
