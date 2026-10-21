"""Port of the two rate limiters in server/index.js and server/routes/auth.js.

- Public GET limiter (slowapi): /api/predictions and /api/stream/status,
  120 req/min (matches express-rate-limit's windowMs 60s, max 120).
- Login limiter (hand-rolled): 10 failed attempts per 15 min per IP.
  slowapi has no failures-only mode, so this one is written directly —
  small enough not to need a library.
"""
import time
from collections import defaultdict

from slowapi import Limiter

from app.security import get_client_ip

public_limiter = Limiter(key_func=get_client_ip)

LOGIN_WINDOW_SECONDS = 15 * 60
LOGIN_MAX_FAILURES = 10

_login_failures: dict[str, list[float]] = defaultdict(list)


def _prune(ip: str) -> None:
    now = time.time()
    _login_failures[ip] = [t for t in _login_failures[ip] if now - t < LOGIN_WINDOW_SECONDS]


def is_login_rate_limited(ip: str) -> bool:
    _prune(ip)
    return len(_login_failures[ip]) >= LOGIN_MAX_FAILURES


def record_login_failure(ip: str) -> None:
    _prune(ip)
    _login_failures[ip].append(time.time())
