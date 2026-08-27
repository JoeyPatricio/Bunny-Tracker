"""Port of server/routes/monitor.js, plus the Clipping Mode control surface.

monitor.json is the one place dashboard-toggled agent state lives; the agent
polls GET /api/monitor every 5s (agent/capture.py::_poll_monitor_loop), so
everything here is live-tunable without a pm2 restart.
"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.auth import is_authed
from app.config import CLIPPING_MAX_UNLABELED, CLIPPING_MIN_FREE_GB, SERVER_DIR
from app.lib.json_store import create_store

_store = create_store(SERVER_DIR, "monitor.json", {})

# Raw motion levels that each 1-10 sensitivity preset maps to. motion["level"]
# is the percentage of pooled cells that changed (shared/motion.py), nominally
# 0-100, but the useful band is roughly 0-10 — the alerting defaults sit at 1.0
# and 2.5. A raw 0-100 slider would spend 90% of its travel doing nothing, so
# the dashboard exposes this scale instead. Higher sensitivity = lower
# threshold = more clips.
SENSITIVITY_THRESHOLDS = (8.0, 6.0, 4.5, 3.5, 2.5, 2.0, 1.5, 1.0, 0.7, 0.4)


def threshold_for_sensitivity(sensitivity: int) -> float:
    index = min(max(int(sensitivity), 1), len(SENSITIVITY_THRESHOLDS)) - 1
    return SENSITIVITY_THRESHOLDS[index]


DEFAULTS = {
    "enabled": True,
    "emailAlerts": True,
    # ── Clipping Mode ────────────────────────────────────────────────────
    "clipping": False,
    "clippingSensitivity": 5,                       # 1-10 preset
    "clippingMotionThreshold": threshold_for_sensitivity(5),
    "clippingCooldownSec": 8,                       # == segment length: one clip per segment
    "clippingDryRun": False,
    "clippingMaxUnlabeled": CLIPPING_MAX_UNLABELED,
    "clippingMinFreeGb": CLIPPING_MIN_FREE_GB,
    "clippingStoppedReason": None,                  # set by the server on an auto-stop
}

# What an unauthenticated caller sees. GET /api/monitor is in
# auth.PUBLIC_GET_PATHS so the demo page can show online/offline — but the
# harvesting configuration is not the public's business, and clippingStoppedReason
# in particular would leak disk state. The agent is unaffected: it sends
# X-Agent-Token, so is_authed() is true for it.
PUBLIC_FIELDS = ("enabled", "emailAlerts")

router = APIRouter()


async def read_monitor_state() -> dict:
    try:
        return {**DEFAULTS, **(await _store.read())}
    except Exception:
        return dict(DEFAULTS)


@router.get("")
@router.get("/")
async def get_monitor(request: Request):
    """Public — the demo page shows online/offline anyway — but only the two
    public fields unless the caller is the admin or the agent."""
    state = await read_monitor_state()
    if not is_authed(request):
        return {k: state[k] for k in PUBLIC_FIELDS}
    return state


class MonitorBody(BaseModel):
    enabled: bool | None = None
    emailAlerts: bool | None = None
    clipping: bool | None = None
    clippingSensitivity: int | None = Field(None, ge=1, le=10)
    clippingMotionThreshold: float | None = Field(None, ge=0, le=100)
    clippingCooldownSec: float | None = Field(None, ge=0, le=3600)
    clippingDryRun: bool | None = None
    clippingMaxUnlabeled: int | None = Field(None, ge=0, le=1_000_000)
    clippingMinFreeGb: float | None = Field(None, ge=0, le=100_000)
    clippingStoppedReason: str | None = None


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

            if body.clippingSensitivity is not None:
                merged["clippingSensitivity"] = int(body.clippingSensitivity)
                # An explicit threshold in the same request wins; otherwise the
                # preset drives it, so the two never drift apart.
                if body.clippingMotionThreshold is None:
                    merged["clippingMotionThreshold"] = threshold_for_sensitivity(
                        body.clippingSensitivity
                    )
            if body.clippingMotionThreshold is not None:
                merged["clippingMotionThreshold"] = float(body.clippingMotionThreshold)
            if body.clippingCooldownSec is not None:
                merged["clippingCooldownSec"] = float(body.clippingCooldownSec)
            if body.clippingDryRun is not None:
                merged["clippingDryRun"] = bool(body.clippingDryRun)
            if body.clippingMaxUnlabeled is not None:
                merged["clippingMaxUnlabeled"] = int(body.clippingMaxUnlabeled)
            if body.clippingMinFreeGb is not None:
                merged["clippingMinFreeGb"] = float(body.clippingMinFreeGb)
            if body.clippingStoppedReason is not None:
                merged["clippingStoppedReason"] = body.clippingStoppedReason or None

            if body.clipping is not None:
                merged["clipping"] = bool(body.clipping)
                # Re-arming clears whatever stopped it last time, so a stale
                # banner can't outlive the condition that caused it.
                if merged["clipping"] and body.clippingStoppedReason is None:
                    merged["clippingStoppedReason"] = None

            return merged

        next_state = await _store.update(mutate)
        return {**DEFAULTS, **next_state}
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Failed to save monitor state"})


async def stop_clipping(reason: str) -> None:
    """Server-side auto-stop. The server owns this state, not the agent — the
    agent just sees clipping go false on its next 5s poll."""
    def mutate(current):
        return {**DEFAULTS, **current, "clipping": False, "clippingStoppedReason": reason}

    try:
        await _store.update(mutate)
    except Exception:
        pass
