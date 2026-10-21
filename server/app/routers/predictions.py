"""Port of server/routes/predictions.js."""
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.lib.iso_time import to_iso_millis
from app.rate_limit import public_limiter

router = APIRouter()

# In-memory ring buffer of recent predictions for the public demo feed. Empty
# on boot, same as the Node version — nothing here outlives a restart.
MAX_EVENTS = 50
events: list[dict] = []


class PredictionBody(BaseModel):
    label: str | None = None
    confidence: float | None = None


@router.post("")
@router.post("/")
async def post_prediction(body: PredictionBody):
    """Admin client (the agent) reports a prediction change — AGENT_TOKEN-guarded."""
    if not body.label:
        return JSONResponse(status_code=400, content={"error": "label required"})
    events.append({"label": body.label, "confidence": body.confidence, "time": to_iso_millis(time.time())})
    while len(events) > MAX_EVENTS:
        events.pop(0)
    return {"ok": True}


@router.get("")
@router.get("/")
@public_limiter.limit("120/minute")
async def get_predictions(request: Request):
    """Public: text-only live feed (no video involved). Rate-limited (120/min,
    matches express-rate-limit's windowMs 60s max 120) so bots can't DoS it."""
    return {"events": events}
