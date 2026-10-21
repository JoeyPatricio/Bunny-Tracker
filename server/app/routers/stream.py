"""Port of server/routes/stream.js."""
import asyncio
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.auth import is_authed
from app.rate_limit import public_limiter

router = APIRouter()

# Latest frame pushed by the camera agent (module-level, matches the Node
# route's in-process globals — reset on restart, single-process only).
latest_frame: bytes | None = None
latest_at: float = 0

# Restores the limits the Node route had as express.raw({ type: 'image/jpeg',
# limit: '2mb' }). Whatever lands here is held in process memory until the next
# frame replaces it and is served back to the dashboard labeled image/jpeg, so
# an unbounded body from a wedged agent is worth rejecting even though this
# route already requires the agent token.
MAX_FRAME_BYTES = 2 * 1024 * 1024


@router.post("/frame")
async def push_frame(request: Request):
    """Agent pushes a JPEG (AGENT_TOKEN-guarded, same as index.js's adminGuard
    on this prefix — is_authed() accepts the agent token)."""
    global latest_frame, latest_at
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if content_type != "image/jpeg":
        return JSONResponse(status_code=415, content={"error": "Expected image/jpeg"})
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_FRAME_BYTES:
        return JSONResponse(status_code=413, content={"error": "Frame too large"})
    body = await request.body()
    # Re-check after reading: content-length is absent on a chunked upload.
    if len(body) > MAX_FRAME_BYTES:
        return JSONResponse(status_code=413, content={"error": "Frame too large"})
    latest_frame = body
    latest_at = time.time()
    return {"ok": True}


@router.get("/status")
@public_limiter.limit("120/minute")
async def stream_status(request: Request):
    """Public: is the agent streaming right now? Rate-limited (120/min)."""
    return {"live": bool(latest_frame) and (time.time() - latest_at) < 10}


@router.get("/live")
async def stream_live(request: Request):
    """PRIVATE: requires admin login. MJPEG multipart stream, ~12fps to match
    the agent's stream output cadence."""
    if not is_authed(request):
        return JSONResponse(status_code=401, content={"error": "Login required. The live stream is private."})

    async def frame_generator():
        last_sent = 0.0
        try:
            while True:
                await asyncio.sleep(0.08)  # ~12 fps
                if await request.is_disconnected():
                    break
                if not latest_frame or latest_at == last_sent:
                    continue
                last_sent = latest_at
                yield (
                    b"--bunnyframe\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(latest_frame)).encode() + b"\r\n\r\n"
                    + latest_frame + b"\r\n"
                )
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=bunnyframe",
        headers={"Cache-Control": "no-cache, no-store", "Pragma": "no-cache"},
    )
