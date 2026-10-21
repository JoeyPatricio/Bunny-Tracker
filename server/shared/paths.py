"""Transient camera paths shared by the server and agent after dotenv loads."""
import os
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SERVER_DIR.parent / "data"


def segment_dir() -> Path:
    return Path(os.environ.get("AGENT_SEGMENT_DIR", DATA_DIR / "agent" / "segments"))


def live_frame_path() -> Path:
    return Path(os.environ.get("AGENT_LIVE_FRAME_PATH", SERVER_DIR / "agent" / "live_frame.jpg"))
