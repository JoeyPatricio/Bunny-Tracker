"""Port of server/lib/listHighlights.js.

List highlight candidates: every non-normal labeled clip on disk, newest
first, each annotated with its current demo-page visibility.
"""
import asyncio

from app.config import RECORDINGS_DIR
from app.lib.hidden_store import read_hidden
from app.lib.iso_time import to_iso_millis
from app.lib.label_store import read_labels


async def list_highlights(include_hidden: bool) -> list[dict]:
    try:
        labels = await read_labels()
    except Exception:
        labels = {}
    try:
        hidden = await read_hidden()
    except Exception:
        hidden = set()

    files = [f.name for f in await asyncio.to_thread(lambda: list(RECORDINGS_DIR.iterdir())) if f.name.endswith(".webm")]

    items = []
    for filename in files:
        label = labels.get(filename)
        if not label or label == "normal":
            continue
        if not include_hidden and filename in hidden:
            continue
        stat = await asyncio.to_thread((RECORDINGS_DIR / filename).stat)
        items.append({
            "filename": filename,
            "label": label,
            "createdAt": to_iso_millis(stat.st_mtime),
            "size": stat.st_size,
            "hidden": filename in hidden,
        })

    items.sort(key=lambda item: item["createdAt"], reverse=True)
    return items
