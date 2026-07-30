"""Port of server/routes/auth.js: session cookie issue/check/clear, adminGuard,
PUBLIC_GET_PATHS, and the /login /logout /check endpoints.

Set-Cookie headers are built as raw strings (not via Starlette's cookie
helpers) so the exact attribute string matches the Node server byte for byte
— this is a security-sensitive contract other clients (the browser, the
future agent) depend on.
"""
import hmac
import json
import secrets
import time

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import ADMIN_PASSWORD, AGENT_TOKEN
from app.rate_limit import is_login_rate_limited, record_login_failure
from app.security import get_client_ip, is_secure_request

COOKIE_NAME = "bunnyauth"
SESSION_TTL_SECONDS = 30 * 24 * 3600  # matches the cookie Max-Age

# In-memory session tokens -> expiry timestamp (epoch seconds). Cleared on
# server restart, fine for personal use.
_sessions: dict[str, float] = {}

# Read endpoints that stay public without a login: the demo page's feeds and
# the agent's unauthenticated monitor poll. Everything else under the guarded
# prefixes requires auth. This is an allowlist so a newly added GET route is
# private by default instead of leaking until someone remembers to guard it.
PUBLIC_GET_PATHS = {
    "/api/predictions",
    "/api/stream/status",
    "/api/recordings/highlights",
    "/api/monitor",
}

# Prefixes adminGuard covers (mirrors the app.use([...], adminGuard) list in
# server/index.js). GET requests to /api/import and /api/model both have no
# public paths, so ALL their GETs need auth except where listed above.
GUARDED_PREFIXES = (
    "/api/recordings", "/api/labels", "/api/logs", "/api/import", "/api/model",
    "/api/notify", "/api/predictions", "/api/stream", "/api/monitor", "/api/highlights",
)


def _secret_equals(supplied: str, expected: str) -> bool:
    """Timing-safe compare that tolerates non-ASCII input.

    hmac.compare_digest raises TypeError on str operands holding codepoints
    above 127. Both call sites take attacker-controlled input (a JSON password,
    a latin-1-decoded X-Agent-Token header), so that TypeError was reachable as
    an uncaught 500 — and on the login path it fired before
    record_login_failure, letting a non-ASCII guess skip the rate limiter
    entirely. Comparing bytes has no such restriction and stays timing-safe.
    """
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def is_agent_authed(request: Request) -> bool:
    """Service-token path for the camera agent (plan Phase 5, the actual fix
    for fixes.md item 4.3): entirely separate from the human cookie/session
    store and the login rate limiter, so a bad token or a reconnect burst can
    never lock out the human admin. Only active when AGENT_TOKEN is set."""
    if not AGENT_TOKEN:
        return False
    header = request.headers.get("x-agent-token")
    if not header:
        return False
    return _secret_equals(header, AGENT_TOKEN)


def is_authed(request: Request) -> bool:
    if is_agent_authed(request):
        return True
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return False
    exp = _sessions.get(token)
    if exp is None:
        return False
    if exp <= time.time():
        _sessions.pop(token, None)
        return False
    return True


class AdminGuardMiddleware(BaseHTTPMiddleware):
    """Require admin auth for any request under a guarded prefix, except the
    allowlisted public GET paths. Mirrors adminGuard's ordering: this runs
    before the request reaches any router."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path.rstrip("/") or "/"
        if any(path == p or path.startswith(p + "/") for p in GUARDED_PREFIXES):
            if not is_authed(request):
                if request.method == "GET" and path in PUBLIC_GET_PATHS:
                    return await call_next(request)
                return JSONResponse(status_code=401, content={"error": "Login required"})
        return await call_next(request)


router = APIRouter()


@router.post("/login")
async def login(request: Request, response: Response):
    ip = get_client_ip(request)
    if is_login_rate_limited(ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Too many login attempts. Try again in 15 minutes."},
        )
    if not ADMIN_PASSWORD:
        return JSONResponse(status_code=503, content={"error": "ADMIN_PASSWORD not set in server/.env"})

    raw = await request.body()
    if not raw:
        # An empty body is valid per body-parser's behavior (defaults to {}),
        # not a parse error — only non-empty malformed JSON is a 400.
        body = {}
    else:
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})
    password = str(body.get("password") or "")

    # Timing-safe comparison so response time doesn't leak password length/content.
    ok = _secret_equals(password, ADMIN_PASSWORD)
    if not ok:
        record_login_failure(ip)
        return JSONResponse(status_code=401, content={"error": "Wrong password"})

    token = secrets.token_hex(32)
    _sessions[token] = time.time() + SESSION_TTL_SECONDS

    # Only mark the cookie Secure on a genuine HTTPS connection. Over plain
    # http://localhost the browser would refuse to send a Secure cookie back,
    # which would silently break every authenticated request.
    secure = "; Secure" if is_secure_request(request) else ""
    response.headers.append(
        "set-cookie",
        f"{COOKIE_NAME}={token}; HttpOnly; SameSite=Lax; Path=/; Max-Age={SESSION_TTL_SECONDS}{secure}",
    )
    return {"ok": True}


@router.post("/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get(COOKIE_NAME)
    if token:
        _sessions.pop(token, None)
    response.headers.append("set-cookie", f"{COOKIE_NAME}=; HttpOnly; Path=/; Max-Age=0")
    return {"ok": True}


@router.get("/check")
async def check(request: Request):
    return {"authed": is_authed(request)}
