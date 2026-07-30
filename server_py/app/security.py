"""Port of the helmet CSP block and 'trust proxy' client-IP handling in
server/index.js.
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Matches server/index.js's helmet() output byte-for-byte (verified against
# the live server's actual response header, not just the source): helmet
# merges the explicit directives configured in index.js with its own default
# directives for everything not overridden (base-uri, font-src, form-action,
# script-src-attr, upgrade-insecure-requests) — those aren't visible in
# index.js's source, only in helmet's own defaults.
CSP = (
    "default-src 'self';"
    "script-src 'self';"
    "style-src 'self' 'unsafe-inline';"
    "img-src 'self' data: blob:;"
    "media-src 'self' blob:;"
    "connect-src 'self';"
    "frame-ancestors 'none';"
    "object-src 'none';"
    "base-uri 'self';"
    "font-src 'self' https: data:;"
    "form-action 'self';"
    "script-src-attr 'none';"
    "upgrade-insecure-requests"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """CSP is relaxed for the SPA (inline styles, blob media for video,
    same-origin connect) but blocks framing and cross-origin script.
    crossOriginEmbedderPolicy is left off (helmet's default is also off
    unless enabled) — MJPEG/video streaming compatibility."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-DNS-Prefetch-Control"] = "off"
        response.headers["X-Download-Options"] = "noopen"
        return response


def get_client_ip(request: Request) -> str:
    """Cloudflare Tunnel is the only hop today, so trusting the first
    X-Forwarded-For entry is an accepted simplification, not a faithful
    'trust proxy' port (plan section 6, risk 9) — revisit if that topology
    changes."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def is_secure_request(request: Request) -> bool:
    """Mirrors Express's req.secure with 'trust proxy' set: true behind the
    Cloudflare tunnel (x-forwarded-proto), true on direct HTTPS."""
    proto = request.headers.get("x-forwarded-proto")
    if proto:
        return proto.split(",")[0].strip() == "https"
    return request.url.scheme == "https"
