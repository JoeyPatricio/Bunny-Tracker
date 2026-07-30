"""Port of server/routes/monitor.js."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import SERVER_DIR
from app.lib.json_store import create_store

_store = create_store(SERVER_DIR, "monitor.json", {})

DEFAULTS = {"enabled": True, "emailAlerts": True}

router = APIRouter()


async def read_monitor_state() -> dict:
    try:
        return {**DEFAULTS, **(await _store.read())}
    except Exception:
        return dict(DEFAULTS)


@router.get("")
@router.get("/")
async def get_monitor():
    """Public — the demo page shows online/offline anyway."""
    return await read_monitor_state()


class MonitorBody(BaseModel):
    enabled: bool | None = None
    emailAlerts: bool | None = None


@router.post("")
@router.post("/")
async def set_monitor(body: MonitorBody):
    """Admin only (guarded by AdminGuardMiddleware, /api/monitor is a guarded
    prefix and POST is never in PUBLIC_GET_PATHS)."""
    try:
        def mutate(current):
            merged = {**DEFAULTS, **current}
            if body.enabled is not None:
                merged["enabled"] = bool(body.enabled)
            if body.emailAlerts is not None:
                merged["emailAlerts"] = bool(body.emailAlerts)
            return merged

        next_state = await _store.update(mutate)
        return {**DEFAULTS, **next_state}
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Failed to save monitor state"})
