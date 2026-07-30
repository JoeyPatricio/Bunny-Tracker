"""Port of capture.mjs's login()/loginWithRetry()/api() helper, replaced by
AGENT_TOKEN (plan Phase 5): no login, no session, no re-login-on-401 wrapper
— a 401 now means a config error (wrong/missing token), nothing to retry.

Reads server/.env directly (independent of app/config.py) since the agent is
meant to run as its own OS process, separate from the FastAPI server.
"""
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

SERVER_PY_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SERVER_PY_DIR.parent
SERVER_DIR = ROOT_DIR / "server"

load_dotenv(SERVER_DIR / ".env")

SERVER_URL = os.environ.get("AGENT_SERVER_URL", "http://localhost:3001")
AGENT_TOKEN = os.environ.get("AGENT_TOKEN")


class AgentClient:
    def __init__(self, base_url: str = SERVER_URL, token: str | None = AGENT_TOKEN):
        if not token:
            raise RuntimeError("AGENT_TOKEN not set in server/.env — the agent cannot authenticate")
        self.token = token
        self.client = httpx.AsyncClient(base_url=base_url, headers={"X-Agent-Token": token}, timeout=10.0)

    async def verify(self) -> None:
        """One authenticated GET at boot to confirm the token actually works,
        so a config error surfaces immediately instead of failing silently on
        the first real request."""
        resp = await self.client.get("/api/auth/check")
        resp.raise_for_status()
        data = resp.json()
        if not data.get("authed"):
            raise RuntimeError("AGENT_TOKEN rejected by server — check server/.env on both sides")

    async def get(self, path: str, **kwargs) -> httpx.Response:
        return await self.client.get(path, **kwargs)

    async def post(self, path: str, **kwargs) -> httpx.Response:
        return await self.client.post(path, **kwargs)

    async def close(self) -> None:
        await self.client.aclose()
