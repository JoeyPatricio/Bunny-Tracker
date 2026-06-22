import 'dotenv/config'
import express from 'express'
import cors from 'cors'
import helmet from 'helmet'
import rateLimit from 'express-rate-limit'
import path from 'path'
import { fileURLToPath } from 'url'
import recordingRoutes from './routes/recordings.js'
import logRoutes from './routes/logs.js'
import labelRoutes from './routes/labels.js'
import importRoutes from './routes/import.js'
import modelRoutes from './routes/model.js'
import smsRoutes from './routes/sms.js'
import authRoutes, { adminGuard, isAuthed } from './routes/auth.js'
import predictionRoutes from './routes/predictions.js'
import streamRoutes from './routes/stream.js'
import monitorRoutes from './routes/monitor.js'
import highlightRoutes from './routes/highlights.js'
import { startLabelBackups } from './lib/backupLabels.js'
import { readLabels } from './lib/labelStore.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const PORT = process.env.PORT || 3001

const app = express()

app.set('trust proxy', 1) // behind Cloudflare Tunnel — needed for correct client IPs (rate limiting)

// Security headers. CSP is relaxed for the SPA (inline styles, blob media for
// video, same-origin connect) but blocks framing and cross-origin script.
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc:  ["'self'"],
      styleSrc:   ["'self'", "'unsafe-inline'"],
      imgSrc:     ["'self'", 'data:', 'blob:'],
      mediaSrc:   ["'self'", 'blob:'],
      connectSrc: ["'self'"],
      frameAncestors: ["'none'"],
      objectSrc:  ["'none'"],
    },
  },
  crossOriginEmbedderPolicy: false, // MJPEG/video streaming compatibility
}))

// Same-origin in production (app is served by this server). Allow configured
// origins for local dev only.
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS ??
  'http://localhost:3000,http://localhost:3001').split(',')
app.use(cors({
  origin: (origin, cb) => {
    if (!origin || ALLOWED_ORIGINS.includes(origin)) return cb(null, true)
    cb(null, false)
  },
  credentials: true,
}))
app.use(express.json({ limit: '50mb' }))

// Guarded recording file access. Logged-in admin gets the full archive;
// logged-out demo visitors may only fetch curated non-normal highlight clips.
// The strict filename regex (no slashes) also blocks path traversal.
const RECORDINGS_DIR = path.join(__dirname, 'recordings')
app.get('/recordings/:filename', async (req, res) => {
  const { filename } = req.params
  if (!/^recording-[A-Za-z0-9._-]+\.webm$/.test(filename)) {
    return res.status(400).json({ error: 'Invalid filename' })
  }
  if (!isAuthed(req)) {
    const labels = await readLabels().catch(() => ({}))
    const label  = labels[filename]
    if (!label || label === 'normal') {
      return res.status(403).json({ error: 'Private — log in to view this clip' })
    }
  }
  res.sendFile(path.join(RECORDINGS_DIR, filename))
})

// Rate limit public read-only endpoints so bots can't DoS the server
const publicLimiter = rateLimit({
  windowMs: 60 * 1000,   // 1 minute
  max: 120,              // 2 req/sec average — plenty for polling at 5s intervals
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests — slow down' },
})
app.use('/api/predictions', publicLimiter)
app.use('/api/stream/status', publicLimiter)

// Auth (public login/logout/check)
app.use('/api/auth', authRoutes)

// Admin guard: GET requests stay public (demo mode reads clips, labels,
// predictions); anything mutating requires login
app.use(['/api/recordings', '/api/labels', '/api/logs', '/api/import', '/api/model',
         '/api/sms', '/api/predictions', '/api/stream', '/api/monitor',
         '/api/highlights'], adminGuard)

// API routes
app.use('/api/recordings', recordingRoutes)
app.use('/api/logs', logRoutes)
app.use('/api/labels', labelRoutes)
app.use('/api/import', importRoutes)
app.use('/api/model', modelRoutes)
app.use('/api/sms', smsRoutes)
app.use('/api/predictions', predictionRoutes)
app.use('/api/stream', streamRoutes)
app.use('/api/monitor', monitorRoutes)
app.use('/api/highlights', highlightRoutes)

// Serve trained model weights
app.use('/model', express.static(path.join(__dirname, 'model')))

// Health check
app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', uptime: process.uptime() })
})

// Serve the built web app (client/dist) — SPA fallback for non-API routes
const DIST = path.join(__dirname, '..', 'client', 'dist')
app.use(express.static(DIST))
app.get('*', (req, res, next) => {
  if (req.path.startsWith('/api/') || req.path.startsWith('/recordings/') || req.path.startsWith('/model/')) {
    return next()
  }
  res.sendFile(path.join(DIST, 'index.html'), err => { if (err) next() })
})

app.listen(PORT, () => {
  console.log(`🐇 BunnyCam server running at http://localhost:${PORT}`)
  startLabelBackups()
})
