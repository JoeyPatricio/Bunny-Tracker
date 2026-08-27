"""Port of server/index.js: FastAPI() instance, middleware stack, static
mounts, /recordings/:filename, startup hook.
"""
import time as _time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from slowapi.errors import RateLimitExceeded

from app import auth
from app.config import (
    ALLOWED_ORIGINS,
    CLIENT_DIST_DIR,
    MODEL_BACKUP_DIR,
    MODEL_DIR,
    PORT,
    RECORDINGS_DIR,
    SEGMENTS_DIR,
)
from app.lib.backup_labels import start_label_backups
from app.lib.hidden_store import read_hidden
from app.lib.label_store import read_labels
from app.lib.recording_name import is_recording_filename
from app.rate_limit import public_limiter
from app.routers import (
    highlights,
    import_clips,
    labels,
    logs,
    model,
    monitor,
    motion,
    notify,
    predictions,
    recordings,
    stream,
)
from app.security import SecurityHeadersMiddleware

_start_time = _time.time()

app = FastAPI()
app.state.limiter = public_limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"error": "Too many requests, slow down"})


# Order matters: Starlette makes the LAST-added middleware the outermost, and
# BaseHTTPMiddleware sets its headers on the way back out. SecurityHeaders must
# therefore be added last, or AdminGuard's 401 short-circuit never passes
# through it and those responses ship with no CSP (helmet ran before adminGuard
# in the Node original, so this was a port regression).
app.add_middleware(auth.AdminGuardMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)


# Guarded recording file access. Logged-in admin gets the full archive;
# logged-out demo visitors may only fetch curated non-normal highlight clips
# that the owner has not hidden. is_recording_filename (no slashes) blocks
# path traversal regardless of how a %2F sequence is decoded by the router.
@app.get("/recordings/{filename:path}")
async def get_recording(filename: str, request: Request):
    if not is_recording_filename(filename):
        return JSONResponse(status_code=400, content={"error": "Invalid filename"})
    if not auth.is_authed(request):
        try:
            labels_map = await read_labels()
        except Exception:
            labels_map = {}
        try:
            hidden = await read_hidden()
        except Exception:
            hidden = set()
        label = labels_map.get(filename)
        # A hidden clip must not be reachable by guessing its URL, or "hide
        # from the demo" would only hide it from the listing.
        if not label or label == "normal" or filename in hidden:
            return JSONResponse(status_code=403, content={"error": "Private. Log in to view this clip."})
    target = RECORDINGS_DIR / filename
    if not target.is_file():
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return FileResponse(target)


app.include_router(auth.router, prefix="/api/auth")
app.include_router(recordings.router, prefix="/api/recordings")
app.include_router(logs.router, prefix="/api/logs")
app.include_router(labels.router, prefix="/api/labels")
app.include_router(model.router, prefix="/api/model")
app.include_router(notify.router, prefix="/api/notify")
app.include_router(predictions.router, prefix="/api/predictions")
app.include_router(stream.router, prefix="/api/stream")
app.include_router(monitor.router, prefix="/api/monitor")
app.include_router(motion.router, prefix="/api/motion")
app.include_router(highlights.router, prefix="/api/highlights")
app.include_router(import_clips.router, prefix="/api/import")


# Serve trained model weights. The agent loads these without a session cookie,
# so the current model and the MobileNet backbone stay public, but the dated
# backups under model/backups/ are not something anyone needs over HTTP.
@app.get("/model/{full_path:path}")
async def get_model_file(full_path: str):
    root = MODEL_DIR.resolve()
    target = (MODEL_DIR / full_path).resolve()
    if root not in target.parents and target != root:
        return JSONResponse(status_code=403, content={"error": "Forbidden"})
    # Test the RESOLVED path, not the raw string. `full_path.startswith("backups")`
    # was bypassable with any detour that re-enters the directory after
    # normalization: %2e%2fbackups/... and x%2f%2e%2e%2fbackups/... both served
    # the backup while passing the containment check above.
    backups = MODEL_BACKUP_DIR.resolve()
    if target == backups or backups in target.parents:
        return JSONResponse(status_code=403, content={"error": "Forbidden"})
    if not target.is_file():
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return FileResponse(target)


@app.get("/api/health")
async def health():
    return {"status": "ok", "uptime": _time.time() - _start_time}


# Serve the built web app (client/dist) — SPA fallback for non-API routes.
@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    if full_path.startswith("api/") or full_path.startswith("recordings/") or full_path.startswith("model/"):
        return JSONResponse(status_code=404, content={"error": "Not found"})
    # Uvicorn percent-decodes the path before routing, so `..` segments reach
    # full_path intact and `CLIENT_DIST_DIR / full_path` walks straight out of
    # client/dist. Resolve and require containment, same as get_model_file
    # above. Without this, GET /%2e%2e/%2e%2e/server/.env served the secrets file.
    if full_path:
        root = CLIENT_DIST_DIR.resolve()
        candidate = (CLIENT_DIST_DIR / full_path).resolve()
        if (candidate == root or root in candidate.parents) and candidate.is_file():
            return FileResponse(candidate)
    return FileResponse(CLIENT_DIST_DIR / "index.html")


_backup_task = None


@app.on_event("startup")
async def on_startup():
    global _backup_task
    # The data tree is git-ignored, so a fresh clone -- or a wipe -- leaves
    # these missing. upload_recording() opens its destination directly and
    # _scan_recordings() scandirs, so both raise until something creates them;
    # only import_clips.py used to. Create them once at boot instead.
    for directory in (RECORDINGS_DIR, SEGMENTS_DIR):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            print(f"startup: could not create {directory}: {err}")
    print(f"BunnyCam Python server running at http://localhost:{PORT}")
    _backup_task = start_label_backups()
