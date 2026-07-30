"""Port of server/routes/logs.js — GET route only (Phase 2 scope).
POST /, DELETE / are Phase 4.
"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.auth import is_authed
from app.config import SERVER_DIR
from app.lib.json_store import create_store

_store = create_store(SERVER_DIR, "activity-log.json", [])

router = APIRouter()


@router.get("")
@router.get("/")
async def get_logs(request: Request):
    """PRIVATE: the activity log is a timeline of when the room is active.
    adminGuard already covers this prefix, so this self-check is redundant
    defense-in-depth, matching the Node route's own behavior."""
    if not is_authed(request):
        return JSONResponse(status_code=401, content={"error": "Login required"})
    try:
        return {"entries": await _store.read()}
    except Exception:
        return {"entries": []}
