"""Env var reads, 1:1 with each process.env site in server/index.js and
server/routes/*.js / server/lib/*.js. Loads server/.env directly (the SAME
env file the live Node server uses) so both servers share one source of
config/secrets — no data migration, no second .env to keep in sync.
"""
import math
import os
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
SERVER_PY_DIR = APP_DIR.parent
ROOT_DIR = SERVER_PY_DIR.parent  # BunnyTracker/
SERVER_DIR = ROOT_DIR / "server"          # ../server — same data files as Node
CLIENT_DIST_DIR = ROOT_DIR / "client" / "dist"

RECORDINGS_DIR = SERVER_DIR / "recordings"
SEGMENTS_DIR = SERVER_DIR / "agent" / "segments"
MODEL_DIR = SERVER_DIR / "model"          # the abandoned tfjs artifact, still served
MODEL_BACKUP_DIR = MODEL_DIR / "backups"
# The ONNX artifacts the agent actually loads and runs. Distinct from MODEL_DIR
# above, which holds the original tfjs model that no Python code path can run.
MODELS_ONNX_DIR = SERVER_PY_DIR / "models"
LABELS_BACKUP_DIR = SERVER_DIR / "backups"

load_dotenv(SERVER_DIR / ".env")

PORT = int(os.environ.get("PY_PORT", "3002"))

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001"
).split(",")

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")
AGENT_TOKEN = os.environ.get("AGENT_TOKEN")

EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")
NOTIFY_TO = os.environ.get("NOTIFY_TO")
NOTIFY_COOLDOWN_DEFAULT_MINUTES = 10


def _numberish(raw: str, *, name: str, default):
    """Matches JS's Number(x) -> JSON.stringify: whole values serialize
    without a trailing .0.

    A bad value must not be fatal. float() raises, and this runs at import
    time, so one typo in .env took the entire server down and left pm2
    crash-looping it to max_restarts and then staying dead. The JS this ports,
    Number(process.env.X ?? 10), produced NaN and kept running: a config typo
    should degrade one feature, not the service.

    Non-finite values are rejected too, not just unparseable ones. NaN and inf
    survive float() happily, and NOTIFY_COOLDOWN_MINUTES feeds comparisons in
    routers/notify.py's alert gate where NaN makes every comparison false and
    silently changes which alerts get sent.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        print(f"config: {name}={raw!r} is not a number, falling back to {default}")
        return default
    if not math.isfinite(value):
        print(f"config: {name}={raw!r} is not finite, falling back to {default}")
        return default
    return int(value) if value.is_integer() else value


NOTIFY_COOLDOWN_MINUTES = _numberish(
    os.environ.get("NOTIFY_COOLDOWN_MINUTES", str(NOTIFY_COOLDOWN_DEFAULT_MINUTES)),
    name="NOTIFY_COOLDOWN_MINUTES",
    default=NOTIFY_COOLDOWN_DEFAULT_MINUTES,
)

# Clipping-mode budget. These are first-run defaults only — once monitor.json
# holds clippingMaxUnlabeled / clippingMinFreeGb, that store is authoritative.
CLIPPING_MAX_UNLABELED = _numberish(
    os.environ.get("CLIPPING_MAX_UNLABELED", "500"),
    name="CLIPPING_MAX_UNLABELED",
    default=500,
)
CLIPPING_MIN_FREE_GB = _numberish(
    os.environ.get("CLIPPING_MIN_FREE_GB", "10"),
    name="CLIPPING_MIN_FREE_GB",
    default=10,
)
