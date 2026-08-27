"""Port of server/routes/recordings.js.

Every filesystem call here goes through asyncio.to_thread. Uvicorn runs one
event loop, so a synchronous copy or a directory walk on it blocks the MJPEG
generator in routers/stream.py and every other in-flight request for its whole
duration (fixes.md 3.1).
"""
import asyncio
import os
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from app.auth import is_authed
from app.config import RECORDINGS_DIR, SEGMENTS_DIR
from app.lib.clip_budget import recordings_usage
from app.lib.iso_time import filename_timestamp, to_iso_millis
from app.lib.list_highlights import list_highlights
from app.lib.recording_name import is_recording_filename
from app.routers.monitor import read_monitor_state, stop_clipping

router = APIRouter()

# Preserved as-is per the plan: this hardcoded 60s cutoff differs from the
# agent's SEGMENT_SECS*2.5, a known pre-existing inconsistency, not something
# to fix here.
GRAB_STALE_MS = 60_000

# Upload sources. Only clips harvested by Clipping Mode are subject to the
# budget in lib/clip_budget.py — a manual Record or an alert clip is never
# refused, however full the labeling queue is.
CLIPPING_SOURCE = "clipping"


def _scan_recordings() -> list[dict]:
    """Whole listing in one thread hop. scandir carries the stat data from the
    directory enumeration itself, so this is one syscall per file rather than
    the two that calling f.stat() separately for mtime and size cost."""
    out = []
    # The data tree is git-ignored, so this can be missing on a fresh clone or
    # after a wipe. _scan_segments below already tolerates that; match it,
    # rather than 500ing every listing until the first upload creates the dir.
    if not RECORDINGS_DIR.exists():
        return out
    with os.scandir(RECORDINGS_DIR) as entries:
        for entry in entries:
            if not entry.name.endswith(".webm"):
                continue
            stat = entry.stat()
            out.append({
                "filename": entry.name,
                "createdAt": to_iso_millis(stat.st_mtime),
                "size": stat.st_size,
            })
    return out


def _scan_segments() -> list[dict]:
    if not SEGMENTS_DIR.exists():
        return []
    with os.scandir(SEGMENTS_DIR) as entries:
        stats = [
            {"name": e.name, "mtime_ms": e.stat().st_mtime * 1000}
            for e in entries if e.name.endswith(".webm")
        ]
    # Order by mtime, not filename — ffmpeg restarts segment numbering at 0,
    # so filename order can put a stale old segment "last" forever.
    stats.sort(key=lambda s: s["mtime_ms"])
    return stats


def _save_upload(src, dest: Path) -> os.stat_result:
    # The data tree is git-ignored, so this directory can be missing on a fresh
    # clone even though on_startup creates it — belt and braces, since the cost
    # of getting it wrong is a 500 on every capture.
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        shutil.copyfileobj(src, out)
    return dest.stat()


async def _current_usage() -> dict:
    state = await read_monitor_state()
    return await recordings_usage(
        recordings_dir=RECORDINGS_DIR,
        max_unlabeled=int(state.get("clippingMaxUnlabeled") or 0),
        min_free_gb=float(state.get("clippingMinFreeGb") or 0),
    )


@router.get("/highlights")
async def get_highlights():
    """PUBLIC: only non-normal labeled clips the owner has not hidden, for the
    demo page. Never exposes the full archive or unlabeled/resting footage."""
    try:
        items = await list_highlights(include_hidden=False)
        recordings = [
            {"filename": i["filename"], "label": i["label"], "createdAt": i["createdAt"], "size": i["size"]}
            for i in items
        ]
        return {"recordings": recordings}
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Failed to list highlights"})


@router.get("")
@router.get("/")
async def list_recordings(request: Request):
    """PRIVATE: full archive listing requires login."""
    if not is_authed(request):
        return JSONResponse(status_code=401, content={"error": "Login required"})
    try:
        recordings = await asyncio.to_thread(_scan_recordings)
        recordings.sort(key=lambda r: r["createdAt"], reverse=True)
        return {"recordings": recordings}
    except OSError as err:
        return JSONResponse(status_code=500, content={"error": "Failed to list recordings", "detail": str(err)})


@router.post("/grab")
async def grab_segment():
    """Save the agent's most recent finished segment as a plain (unlabeled)
    recording. Used by the live-feed Record button."""
    try:
        stats = await asyncio.to_thread(_scan_segments)
        if len(stats) < 2:
            return JSONResponse(status_code=409, content={"error": "No finished segment yet. Is the camera capturing?"})
        # The newest file is still being written by ffmpeg, so the
        # second-newest is the latest FINISHED segment.
        pick = stats[-2]
        if time.time() * 1000 - pick["mtime_ms"] > GRAB_STALE_MS:
            return JSONResponse(status_code=409, content={"error": "Newest segment is stale. Is the camera capturing?"})

        filename = f"recording-manual-{filename_timestamp()}.webm"
        await asyncio.to_thread(shutil.copyfile, SEGMENTS_DIR / pick["name"], RECORDINGS_DIR / filename)
        return {"filename": filename}
    except OSError as err:
        return JSONResponse(status_code=500, content={"error": "Grab failed", "detail": str(err)})


@router.get("/usage")
async def get_usage():
    """Clip count, labeling backlog and disk headroom — drives the Clipping
    Mode panel's counter and disk bar. Under a guarded prefix, so login-only."""
    try:
        return await _current_usage()
    except OSError as err:
        return JSONResponse(status_code=500, content={"error": "Failed to read usage", "detail": str(err)})


@router.post("")
@router.post("/")
async def upload_recording(video: UploadFile, source: str | None = Form(None)):
    if not video.content_type or not video.content_type.startswith("video/"):
        return JSONResponse(status_code=500, content={"error": "Only video files are allowed"})

    if source == CLIPPING_SOURCE:
        usage = await _current_usage()
        if usage["blocked"]:
            # Stop the harvest at the source rather than refusing clip after
            # clip: the agent picks the new state up on its next 5s poll.
            await stop_clipping(usage["reason"])
            return JSONResponse(
                status_code=507,
                content={"error": "Clipping budget reached", "reason": usage["reason"], "usage": usage},
            )

    prefix = "recording-clip-" if source == CLIPPING_SOURCE else "recording-"
    filename = f"{prefix}{filename_timestamp()}.webm"
    dest = RECORDINGS_DIR / filename
    stat = await asyncio.to_thread(_save_upload, video.file, dest)
    return {"filename": filename, "createdAt": to_iso_millis(stat.st_mtime), "size": stat.st_size}


@router.delete("/{filename:path}")
async def delete_recording(filename: str):
    if not is_recording_filename(filename):
        return JSONResponse(status_code=400, content={"error": "Invalid filename"})
    filepath = RECORDINGS_DIR / filename
    try:
        filepath.unlink()
        return {"deleted": filename}
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "File not found"})
    except OSError as err:
        return JSONResponse(status_code=500, content={"error": "Failed to delete", "detail": str(err)})
