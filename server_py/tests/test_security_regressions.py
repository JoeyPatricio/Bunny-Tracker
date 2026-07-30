"""Regression gate for the security fixes in notes/fixes.md installment 1.

Each check below reproduces a bug that was live on 2026-07-24 and asserts the
fixed behavior. Same shape as test_motion_parity.py: no pytest, plain main(),
exit 1 on failure.

Usage (from server_py/, runtime venv):
    .venv/Scripts/python.exe -m tests.test_security_regressions

Uses TestClient, which unquotes the URL path into the ASGI scope exactly as
uvicorn does, so the %2e%2e traversal cases reproduce here rather than needing
a live server. Read-only apart from POSTs that touch in-memory state only:
nothing under server/ is written.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.config import AGENT_TOKEN, MODEL_BACKUP_DIR, SERVER_DIR
from app.main import app
from app.security import CSP

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")
        failures.append(name)


def test_spa_traversal(client: TestClient) -> None:
    """fixes.md 1.1: the SPA fallback served any file on disk."""
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
    """fixes.md 1.2: compare_digest raised TypeError on non-ASCII str."""
    res = client.post("/api/auth/login", json={"password": "pässwörd"})
    # 429 would mean an earlier run tripped the limiter; still not a 500.
    check("1.2 non-ASCII password is 401 not 500", res.status_code in (401, 429), f"got {res.status_code}")

    # Sent as raw bytes because httpx refuses to ascii-encode a str header
    # value; this is what is actually on the wire. Starlette decodes headers as
    # latin-1, so the route sees a str with codepoints above 127.
    res = client.get("/api/labels", headers={"X-Agent-Token": "tökén".encode("latin-1")})
    check("1.2 non-ASCII agent token is 401 not 500", res.status_code == 401, f"got {res.status_code}")


def test_model_backups(client: TestClient) -> None:
    """fixes.md 1.3: startswith('backups') was bypassable via normalization."""
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
    """fixes.md 1.4: AdminGuard's 401 short-circuited past the header middleware."""
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
    """fixes.md 1.5: bare request.body() with no cap and no type check."""
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

    if failures:
        print(f"\nFAIL: {len(failures)} check(s) failed")
        sys.exit(1)
    print("\nPASS: all security regression checks passed")


if __name__ == "__main__":
    main()
