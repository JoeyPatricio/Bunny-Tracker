"""Port of server/routes/notify.js."""
import asyncio
import math
import smtplib
import time
from email.message import EmailMessage

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import EMAIL_PASS, EMAIL_USER, NOTIFY_COOLDOWN_MINUTES, NOTIFY_TO, RECORDINGS_DIR
from app.lib.recording_name import is_recording_filename
from app.lib.valid_labels import VALID_LABELS
from app.routers.monitor import read_monitor_state

router = APIRouter()

# Every behavior except the resting baseline is worth an alert.
ALERT_LABELS = {l for l in VALID_LABELS if l != "normal"}

# notify.js's OWN label phrase map (for email subject/body text) — distinct
# from shared/labels.py's LABEL_PHRASE, which is the client UI's wording.
LABEL_PHRASE = {
    "zoomies": "doing zoomies \U0001F407\U0001F4A8",
    "yawn": "yawning \U0001F62A",
    "grooming": "grooming \U0001F43E",
    "standing": "standing up \U0001F998",
}

# Per-label cooldown map, plus a global floor (half the cooldown) between any
# two alerts.
_last_sent_at: dict[str, float] = {}
_last_sent_any: float = 0.0


def _is_configured() -> bool:
    return bool(EMAIL_USER and EMAIL_PASS and NOTIFY_TO)


async def alert_gate(label: str | None) -> dict:
    global _last_sent_any

    if label not in ALERT_LABELS:
        return {"allow": False, "reason": "label not an alert trigger"}

    monitor_state = await read_monitor_state()
    if not monitor_state["emailAlerts"]:
        return {"allow": False, "reason": "alerts disabled from dashboard"}

    cooldown_sec = NOTIFY_COOLDOWN_MINUTES * 60
    global_floor = cooldown_sec / 2
    now = time.time()

    if _last_sent_any and now - _last_sent_any < global_floor:
        wait_sec = math.ceil(global_floor - (now - _last_sent_any))
        return {"allow": False, "reason": f"global floor, {wait_sec}s remaining"}
    if label in _last_sent_at and now - _last_sent_at[label] < cooldown_sec:
        wait_min = math.ceil((cooldown_sec - (now - _last_sent_at[label])) / 60)
        return {"allow": False, "reason": f"cooldown ({label}), {wait_min}m remaining"}

    _last_sent_any = now
    _last_sent_at[label] = now
    return {"allow": True}


def _send_email_sync(label: str, confidence: float | None, clip_path) -> None:
    phrase = LABEL_PHRASE.get(label, label)
    conf_str = f" ({confidence}% confidence)" if confidence else ""
    subject = f"\U0001F407 BunnyCam Alert: bunny is {phrase}"
    text = f"Your bunny is {phrase}{conf_str}!\n\nBunnyCam detected this at {time.strftime('%I:%M:%S %p')}."

    msg = EmailMessage()
    msg["From"] = f'"BunnyCam \U0001F407" <{EMAIL_USER}>'
    msg["To"] = NOTIFY_TO
    msg["Subject"] = subject
    msg.set_content(text)

    if clip_path is not None:
        with open(clip_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="video", subtype="webm", filename=clip_path.name)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_USER, EMAIL_PASS)
        smtp.send_message(msg)


@router.get("/status")
async def notify_status():
    """Not in PUBLIC_GET_PATHS, so adminGuard already requires login for this
    (despite the low-sensitivity payload) — matches the real Node behavior."""
    return {"configured": _is_configured(), "cooldownMinutes": NOTIFY_COOLDOWN_MINUTES}


class GateBody(BaseModel):
    label: str | None = None


@router.post("/gate")
async def notify_gate(body: GateBody):
    """Apply the alert policy (toggle + cooldown) and reserve it. The agent
    calls this before firing the n8n webhook so the webhook obeys the same
    policy as email."""
    return await alert_gate(body.label)


class NotifyBody(BaseModel):
    label: str | None = None
    confidence: float | None = None
    filename: str | None = None


@router.post("")
@router.post("/")
async def send_notify(body: NotifyBody):
    if not _is_configured():
        return {"sent": False, "reason": "email not configured, add .env"}

    clip_path = None
    if body.filename:
        if not is_recording_filename(body.filename):
            return JSONResponse(status_code=400, content={"sent": False, "error": "Invalid filename"})
        clip_path = RECORDINGS_DIR / body.filename

    gate = await alert_gate(body.label)
    if not gate["allow"]:
        return {"sent": False, "reason": gate["reason"]}

    try:
        await asyncio.to_thread(_send_email_sync, body.label, body.confidence, clip_path)
        return {"sent": True, "label": body.label, "hasAttachment": bool(body.filename)}
    except Exception as err:
        return JSONResponse(status_code=500, content={"sent": False, "error": str(err)})
