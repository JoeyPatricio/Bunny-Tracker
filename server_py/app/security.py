"""Port of the helmet CSP block and 'trust proxy' client-IP handling in
server/index.js.

Forwarded headers are never read here. Uvicorn's ProxyHeadersMiddleware
(on by default) rewrites the ASGI client and scheme from X-Forwarded-For and
X-Forwarded-Proto only when the TCP peer is in --forwarded-allow-ips, which
defaults to 127.0.0.1 where cloudflared runs. Any other caller, including a
LAN browser hitting the 0.0.0.0 bind directly, is seen as its real address.
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Matches server/index.js's helmet() output (verified against the live
# server's actual response header, not just the source): helmet merges the
# explicit directives configured in index.js with its own default directives
# for everything not overridden (base-uri, font-src, form-action,
# script-src-attr, upgrade-insecure-requests) — those aren't visible in
# index.js's source, only in helmet's own defaults. upgrade-insecure-requests
# is split out and sent on HTTPS only: on the plain-HTTP LAN dashboard it
# rewrites every subresource to https://, which has no fallback, so the page
# loads blank.
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
    "script-src-attr 'none'"
)
CSP_HTTPS = CSP + ";upgrade-insecure-requests"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """CSP is relaxed for the SPA (inline styles, blob media for video,
    same-origin connect) but blocks framing and cross-origin script.
    crossOriginEmbedderPolicy is left off (helmet's default is also off
    unless enabled) — MJPEG/video streaming compatibility."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            CSP_HTTPS if is_secure_request(request) else CSP
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-DNS-Prefetch-Control"] = "off"
        response.headers["X-Download-Options"] = "noopen"
        return response


def get_client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def is_secure_request(request: Request) -> bool:
    """Express's req.secure: true on direct HTTPS and behind the Cloudflare
    tunnel, whose https scheme arrives via the proxy middleware above."""
    return request.url.scheme == "https"
