"""Live motion telemetry for tuning Clipping Mode.

Same shape as routers/predictions.py: an in-memory ring buffer, empty on boot,
single-process only, nothing outlives a restart.

Why this exists: picking a motion threshold with no labeled data is guesswork.
The agent already computes a level for every frame (shared/motion.py) but only
logs it, so the only way to tune was to read pm2 output and edit .env. Pushing
the same numbers here lets the dashboard draw the trace with the threshold
overlaid — watch a rabbit move, see where the spikes land, put the line under
them.

PRIVACY: this prefix is registered in auth.GUARDED_PREFIXES and is deliberately
NOT in PUBLIC_GET_PATHS. A live motion trace of the inside of a house is a
presence signal; the public demo feed gets behavior text, not this.
"""
import time

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.lib.iso_time import to_iso_millis
from app.rate_limit import public_limiter

router = APIRouter()

# ~5 minutes of trace at the clipping frame cadence (0.5s/frame).
MAX_EVENTS = 600
events: list[dict] = []


class MotionBody(BaseModel):
    level: float = 0.0
    hotCells: int = 0
    lighting: bool = False
    threshold: float | None = None
    wouldCapture: bool = False
    captured: bool = False


@router.post("")
@router.post("/")
async def post_motion(body: MotionBody):
    """The agent reports one frame's motion reading — AGENT_TOKEN-guarded."""
    events.append({
        "level": round(body.level, 2),
        "hotCells": body.hotCells,
        "lighting": body.lighting,
        "threshold": body.threshold,
        "wouldCapture": body.wouldCapture,
        "captured": body.captured,
        "time": to_iso_millis(time.time()),
    })
    while len(events) > MAX_EVENTS:
        events.pop(0)
    return {"ok": True}


@router.get("")
@router.get("/")
@public_limiter.limit("240/minute")
async def get_motion(request: Request):
    """PRIVATE (see module docstring). Rate-limited at twice the predictions
    feed's budget, since the tuning panel polls faster than the demo page."""
    return {"events": events}
