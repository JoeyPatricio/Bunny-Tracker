"""Regression gate for the security fixes in docs/provenance.md, fixes.md, installment 1.

Each check below reproduces a bug that was live on 2026-07-24 and asserts the
fixed behavior. Same shape as test_motion_parity.py: no pytest, plain main(),
exit 1 on failure.

Usage (from server_py/, runtime venv):
    .venv/Scripts/python.exe -m tests.test_security_regressions

Uses TestClient, which unquotes the URL path into the ASGI scope exactly as
uvicorn does, so the %2e%2e traversal cases reproduce here rather than needing
a live server. Read-only apart from POSTs that touch in-memory state only:
nothing under server/ is written. The one check that logs in successfully
repoints auth.SESSIONS_FILE at a temp directory for its own duration.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# These checks must exercise authentication without an owner's .env file.
os.environ["ADMIN_PASSWORD"] = "security-test-password"
os.environ["AGENT_TOKEN"] = "security-test-agent-token"

from fastapi.testclient import TestClient
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from app import auth, rate_limit
from app.config import ADMIN_PASSWORD, AGENT_TOKEN, MODEL_BACKUP_DIR, SERVER_DIR
from app.main import app
from app.security import CSP, CSP_HTTPS
from tests.checks import check, finish

LAN_PEER = ("192.168.1.50", 50000)
LOOPBACK_PEER = ("127.0.0.1", 50000)


def proxied_client(peer: tuple[str, int]) -> TestClient:
    """The app as uvicorn serves it: ProxyHeadersMiddleware in front, trusting
    only loopback (its default --forwarded-allow-ips), with `peer` as the TCP
    peer address."""
    return TestClient(ProxyHeadersMiddleware(app, trusted_hosts="127.0.0.1"), client=peer)


def test_spa_traversal(client: TestClient) -> None:
    """docs/provenance.md, fixes.md 1.1: the SPA fallback served any file on disk."""
    # A known non-secret file two levels up from client/dist, used so the test
    # asserts on content it can name without touching server/.env.
    canary = SERVER_DIR.parent / "server_py" / "requirements.txt"
    canary_text = canary.read_text(encoding="utf-8").strip()

    for path in (
        "/%2e%2e/%2e%2e/server_py/requirements.txt",
        "/../../server_py/requirements.txt",
        "/assets/%2e%2e/%2e%2e/%2e%2e/server_py/requirements.txt",
    ):
        res = client.get(path)
        leaked = canary_text and canary_text in res.text
        check(f"1.1 traversal blocked: {path}", not leaked, f"served {canary} verbatim")

    # The real target: server/.env holds ADMIN_PASSWORD, AGENT_TOKEN, EMAIL_PASS.
    # Assert on marker names only, never on values.
    res = client.get("/%2e%2e/%2e%2e/server/.env")
    exposed = [k for k in ("ADMIN_PASSWORD", "AGENT_TOKEN", "EMAIL_PASS") if k in res.text]
    check("1.1 server/.env not served", not exposed, f"leaked keys {exposed}")

    # The fallback must still work for real assets and unknown SPA routes.
    check("1.1 index still served", client.get("/").status_code == 200)
    check("1.1 unknown route falls back to index", client.get("/highlights").status_code == 200)


def test_non_ascii_secrets(client: TestClient) -> None:
    """docs/provenance.md, fixes.md 1.2: compare_digest raised TypeError on non-ASCII str."""
    res = client.post("/api/auth/login", json={"password": "pässwörd"})
    # 429 would mean an earlier run tripped the limiter; still not a 500.
    check("1.2 non-ASCII password is 401 not 500", res.status_code in (401, 429), f"got {res.status_code}")

    # Sent as raw bytes because httpx refuses to ascii-encode a str header
    # value; this is what is actually on the wire. Starlette decodes headers as
    # latin-1, so the route sees a str with codepoints above 127.
    res = client.get("/api/labels", headers={"X-Agent-Token": "tökén".encode("latin-1")})
    check("1.2 non-ASCII agent token is 401 not 500", res.status_code == 401, f"got {res.status_code}")


def test_model_backups(client: TestClient) -> None:
    """docs/provenance.md, fixes.md 1.3: startswith('backups') was bypassable via normalization."""
    backup = next((f.name for f in MODEL_BACKUP_DIR.iterdir()), None) if MODEL_BACKUP_DIR.exists() else None
    if not backup:
        check("1.3 model backups", False, "no backup file on disk to test against")
        return
    for path in (
        f"/model/backups/{backup}",
        f"/model/%2e%2fbackups/{backup}",
        f"/model/x%2f%2e%2e%2fbackups/{backup}",
        f"/model/./backups/{backup}",
    ):
        res = client.get(path)
        check(f"1.3 backup blocked: {path}", res.status_code == 403, f"got {res.status_code}")

    # The current model must stay public: the agent fetches it without a cookie.
    check("1.3 current model still public", client.get("/model/model.json").status_code == 200)


def test_security_headers(client: TestClient) -> None:
    """docs/provenance.md, fixes.md 1.4: AdminGuard's 401 short-circuited past the header middleware."""
    res = client.get("/api/labels")
    check("1.4 guarded route is 401", res.status_code == 401, f"got {res.status_code}")
    check("1.4 CSP present on 401", res.headers.get("content-security-policy") == CSP)
    check("1.4 X-Frame-Options present on 401", res.headers.get("x-frame-options") == "DENY")

    res = client.get("/api/health")
    check("1.4 CSP still present on 200", res.headers.get("content-security-policy") == CSP)

    # CORS must not have regressed when the middleware order changed.
    res = client.options(
        "/api/monitor",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    check("1.4 CORS preflight still answered", res.status_code == 200, f"got {res.status_code}")
    check(
        "1.4 CORS preflight allows origin",
        res.headers.get("access-control-allow-origin") == "http://localhost:3000",
    )


def test_stream_frame_limits(client: TestClient) -> None:
    """docs/provenance.md, fixes.md 1.5: bare request.body() with no cap and no type check."""
    if not AGENT_TOKEN:
        check("1.5 stream frame limits", False, "AGENT_TOKEN not set, cannot exercise the route")
        return
    hdr = {"X-Agent-Token": AGENT_TOKEN}

    res = client.post("/api/stream/frame", content=b"not a jpeg", headers={**hdr, "Content-Type": "text/plain"})
    check("1.5 wrong content-type is 415", res.status_code == 415, f"got {res.status_code}")

    oversized = b"\xff\xd8" + b"\x00" * (2 * 1024 * 1024 + 1)
    res = client.post("/api/stream/frame", content=oversized, headers={**hdr, "Content-Type": "image/jpeg"})
    check("1.5 oversized body is 413", res.status_code == 413, f"got {res.status_code}")

    # The agent's real call shape must still work (capture.py:305).
    res = client.post("/api/stream/frame", content=b"\xff\xd8\xff\xd9", headers={**hdr, "Content-Type": "image/jpeg"})
    check("1.5 valid jpeg accepted", res.status_code == 200, f"got {res.status_code}")


def test_forwarded_for_trust() -> None:
    """Pi review finding 8: any LAN caller could rotate X-Forwarded-For past the login limiter."""
    wrong = {"password": "wrong-password"}
    saved_failures = dict(rate_limit._login_failures)
    rate_limit._login_failures.clear()
    try:
        lan = proxied_client(LAN_PEER)
        codes = [
            lan.post("/api/auth/login", json=wrong, headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code
            for i in range(12)
        ]
        check("8 untrusted peer's rotating XFF is ignored", codes[:10] == [401] * 10 and codes[10:] == [429, 429],
              str(codes))
        check("8 lockout keyed on the real LAN peer", LAN_PEER[0] in rate_limit._login_failures,
              str(list(rate_limit._login_failures)))

        # The tunnel path: cloudflared on loopback is trusted, so each visitor
        # is limited separately rather than all sharing 127.0.0.1.
        rate_limit._login_failures.clear()
        tunnel = proxied_client(LOOPBACK_PEER)
        codes = [
            tunnel.post("/api/auth/login", json=wrong, headers={"X-Forwarded-For": "203.0.113.7"}).status_code
            for _ in range(11)
        ]
        check("8 trusted proxy's client is locked out", codes[-1] == 429, str(codes))
        res = tunnel.post("/api/auth/login", json=wrong, headers={"X-Forwarded-For": "198.51.100.9"})
        check("8 another tunnel client is not locked out", res.status_code == 401, f"got {res.status_code}")
        check("8 loopback itself is never the limiter key", LOOPBACK_PEER[0] not in rate_limit._login_failures,
              str(list(rate_limit._login_failures)))
    finally:
        rate_limit._login_failures.clear()
        rate_limit._login_failures.update(saved_failures)


def test_https_only_cookie_and_csp() -> None:
    """Pi review finding 8 and CSP note: forwarded scheme trust, and a plain-HTTP LAN dashboard."""
    good = {"password": ADMIN_PASSWORD}
    https = {"X-Forwarded-Proto": "https"}
    saved_sessions = dict(auth._sessions)
    saved_failures = dict(rate_limit._login_failures)
    saved_sessions_file = auth.SESSIONS_FILE
    tmp = Path(tempfile.mkdtemp(prefix="bunnysessions-"))
    # These logins succeed, so they persist: only this test's sessions may
    # reach the file, and the file must be a temp one.
    auth.SESSIONS_FILE = tmp / "sessions.json"
    auth._sessions.clear()
    rate_limit._login_failures.clear()
    try:
        tunnel = proxied_client(LOOPBACK_PEER)
        res = tunnel.post("/api/auth/login", json=good, headers=https)
        check("8 tunnel login succeeds", res.status_code == 200, f"got {res.status_code}")
        check("8 tunnel HTTPS cookie is Secure", "Secure" in res.headers.get("set-cookie", ""),
              res.headers.get("set-cookie", ""))
        check("CSP upgrades subresources over the tunnel",
              res.headers.get("content-security-policy") == CSP_HTTPS)

        lan = proxied_client(LAN_PEER)
        res = lan.post("/api/auth/login", json=good, headers=https)
        check("8 LAN login succeeds", res.status_code == 200, f"got {res.status_code}")
        check("8 spoofed X-Forwarded-Proto from LAN does not mark the cookie Secure",
              "Secure" not in res.headers.get("set-cookie", ""), res.headers.get("set-cookie", ""))

        res = lan.get("/api/health")
        check("plain-HTTP LAN CSP has no upgrade-insecure-requests",
              res.headers.get("content-security-policy") == CSP
              and "upgrade-insecure-requests" not in CSP, res.headers.get("content-security-policy", ""))
        check("other security headers kept on HTTP",
              res.headers.get("x-frame-options") == "DENY"
              and res.headers.get("x-content-type-options") == "nosniff")

        direct = TestClient(app, base_url="https://testserver")
        res = direct.get("/api/health")
        check("CSP upgrades subresources on direct HTTPS",
              res.headers.get("content-security-policy") == CSP_HTTPS)
    finally:
        auth._sessions.clear()
        auth._sessions.update(saved_sessions)
        rate_limit._login_failures.clear()
        rate_limit._login_failures.update(saved_failures)
        auth.SESSIONS_FILE = saved_sessions_file
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    # No `with` block: entering the lifespan would start the label-backup task,
    # which writes to server/backups/. These checks need no startup state.
    client = TestClient(app)
    for fn in (
        test_spa_traversal,
        test_non_ascii_secrets,
        test_model_backups,
        test_security_headers,
        test_stream_frame_limits,
    ):
        print(f"\n{fn.__doc__.splitlines()[0]}")
        fn(client)

    for fn in (test_forwarded_for_trust, test_https_only_cookie_and_csp):
        print(f"\n{fn.__doc__.splitlines()[0]}")
        fn()

    finish("security regression")


if __name__ == "__main__":
    main()
