import express from 'express'
import crypto from 'crypto'
import rateLimit from 'express-rate-limit'

const router = express.Router()

// Throttle login attempts: 10 tries per 15 min per IP, then locked out
const loginLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many login attempts. Try again in 15 minutes.' },
  skipSuccessfulRequests: true, // only count failed attempts
})

// In-memory session tokens → expiry timestamp (cleared on server restart, fine
// for personal use). A Map with expiry, not a Set, so tokens don't accumulate
// forever: every browser login and every agent re-login adds one.
const sessions = new Map()
const SESSION_TTL_MS = 30 * 24 * 3600 * 1000 // matches the cookie Max-Age

// Drop expired tokens hourly so the Map stays bounded on a long-running server.
setInterval(() => {
  const now = Date.now()
  for (const [token, exp] of sessions) if (exp <= now) sessions.delete(token)
}, 3600 * 1000).unref()

const COOKIE_NAME = 'bunnyauth'

// Read endpoints that stay public without a login: the demo page's feeds and the
// agent's unauthenticated monitor poll. Everything else under the guarded
// prefixes requires auth. This is an allowlist so a newly added GET route is
// private by default instead of leaking until someone remembers to guard it.
const PUBLIC_GET_PATHS = new Set([
  '/api/predictions',
  '/api/stream/status',
  '/api/recordings/highlights',
  '/api/monitor',
])

function parseCookies(req) {
  const header = req.headers.cookie
  if (!header) return {}
  return Object.fromEntries(
    header.split(';').map(c => {
      const [k, ...v] = c.trim().split('=')
      return [k, v.join('=')]
    })
  )
}

export function isAuthed(req) {
  const token = parseCookies(req)[COOKIE_NAME]
  if (!token) return false
  const exp = sessions.get(token)
  if (!exp) return false
  if (exp <= Date.now()) { sessions.delete(token); return false }
  return true
}

/**
 * Middleware: require admin auth. Reads are private by default; only the paths
 * in PUBLIC_GET_PATHS are reachable logged out. Any mutating request needs auth.
 */
export function adminGuard(req, res, next) {
  if (isAuthed(req)) return next()
  if (req.method === 'GET') {
    const path = req.originalUrl.split('?')[0].replace(/\/+$/, '') || '/'
    if (PUBLIC_GET_PATHS.has(path)) return next()
  }
  res.status(401).json({ error: 'Login required' })
}

// POST /api/auth/login  { password }
router.post('/login', loginLimiter, (req, res) => {
  const { password } = req.body
  const expected = process.env.ADMIN_PASSWORD

  if (!expected) {
    return res.status(503).json({ error: 'ADMIN_PASSWORD not set in server/.env' })
  }

  // Timing-safe comparison so response time doesn't leak password length/content
  const a = Buffer.from(String(password ?? ''))
  const b = Buffer.from(expected)
  const ok = a.length === b.length && crypto.timingSafeEqual(a, b)
  if (!ok) {
    return res.status(401).json({ error: 'Wrong password' })
  }

  const token = crypto.randomBytes(32).toString('hex')
  sessions.set(token, Date.now() + SESSION_TTL_MS)

  // Only mark the cookie Secure on a genuine HTTPS connection. Over plain
  // http://localhost the browser would refuse to send a Secure cookie back,
  // which would silently break every authenticated request. With `trust proxy`
  // set, req.secure is true behind the Cloudflare tunnel (x-forwarded-proto).
  const secure = req.secure ? '; Secure' : ''
  res.setHeader('Set-Cookie',
    `${COOKIE_NAME}=${token}; HttpOnly; SameSite=Lax; Path=/; Max-Age=${30 * 24 * 3600}${secure}`)
  res.json({ ok: true })
})

// POST /api/auth/logout
router.post('/logout', (req, res) => {
  const token = parseCookies(req)[COOKIE_NAME]
  if (token) sessions.delete(token)
  res.setHeader('Set-Cookie', `${COOKIE_NAME}=; HttpOnly; Path=/; Max-Age=0`)
  res.json({ ok: true })
})

// GET /api/auth/check
router.get('/check', (req, res) => {
  res.json({ authed: isAuthed(req) })
})

export default router
