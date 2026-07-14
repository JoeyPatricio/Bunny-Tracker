import express from 'express'
import { isAuthed } from './auth.js'

const router = express.Router()

// Latest frame pushed by the camera agent
let latestFrame = null
let latestAt    = 0

// POST /api/stream/frame — agent pushes a JPEG (admin-guarded in index.js)
router.post('/frame', express.raw({ type: 'image/jpeg', limit: '2mb' }), (req, res) => {
  latestFrame = req.body
  latestAt    = Date.now()
  res.json({ ok: true })
})

// GET /api/stream/status — public: is the agent streaming right now?
router.get('/status', (_req, res) => {
  res.json({ live: !!latestFrame && Date.now() - latestAt < 10_000 })
})

// GET /api/stream/live — MJPEG stream. PRIVATE: requires admin login.
router.get('/live', (req, res) => {
  if (!isAuthed(req)) {
    return res.status(401).json({ error: 'Login required. The live stream is private.' })
  }

  res.writeHead(200, {
    'Content-Type': 'multipart/x-mixed-replace; boundary=bunnyframe',
    'Cache-Control': 'no-cache, no-store',
    Connection: 'keep-alive',
    Pragma: 'no-cache',
  })

  let lastSent = 0
  const timer = setInterval(() => {
    if (!latestFrame || latestAt === lastSent) return
    lastSent = latestAt
    res.write(`--bunnyframe\r\nContent-Type: image/jpeg\r\nContent-Length: ${latestFrame.length}\r\n\r\n`)
    res.write(latestFrame)
    res.write('\r\n')
  }, 80) // ~12 fps to match the agent's stream output

  req.on('close', () => clearInterval(timer))
})

export default router
