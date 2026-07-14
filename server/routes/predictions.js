import express from 'express'

const router = express.Router()

// In-memory ring buffer of recent predictions for the public demo feed
const MAX_EVENTS = 50
let events = []

// GET /api/predictions — public: text-only live feed (no video involved).
// The buffer is in-memory and empty on boot, so nothing here outlives a restart.
router.get('/', (_req, res) => {
  res.json({ events })
})

// POST /api/predictions — admin client reports a prediction change
// (protected by adminGuard in index.js)
router.post('/', (req, res) => {
  const { label, confidence } = req.body
  if (!label) return res.status(400).json({ error: 'label required' })

  events.push({ label, confidence: confidence ?? null, time: new Date().toISOString() })
  if (events.length > MAX_EVENTS) events = events.slice(-MAX_EVENTS)
  res.json({ ok: true })
})

export default router
