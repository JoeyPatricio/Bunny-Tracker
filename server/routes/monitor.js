import express from 'express'
import { createStore } from '../lib/jsonStore.js'

const store = createStore('monitor.json', { defaultValue: {} })

const router = express.Router()

const DEFAULTS = { enabled: true, emailAlerts: true }

export async function readMonitorState() {
  try {
    return { ...DEFAULTS, ...(await store.read()) }
  } catch {
    return { ...DEFAULTS }
  }
}

// GET /api/monitor — public (demo page shows online/offline anyway)
router.get('/', async (_req, res) => {
  res.json(await readMonitorState())
})

// POST /api/monitor { enabled?, emailAlerts? } — admin only (guarded in index.js)
router.post('/', async (req, res) => {
  try {
    const next = await store.update(current => {
      const merged = { ...DEFAULTS, ...current }
      if ('enabled' in req.body)     merged.enabled     = !!req.body.enabled
      if ('emailAlerts' in req.body) merged.emailAlerts = !!req.body.emailAlerts
      return merged
    })
    res.json({ ...DEFAULTS, ...next })
  } catch (err) {
    console.error('Save monitor state failed:', err.message)
    res.status(500).json({ error: 'Failed to save monitor state' })
  }
})

export default router
