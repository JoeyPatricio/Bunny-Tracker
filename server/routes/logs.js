import express from 'express'
import { createStore } from '../lib/jsonStore.js'
import { isAuthed } from './auth.js'

const store = createStore('activity-log.json', { defaultValue: [] })

const router = express.Router()

// GET /api/logs — PRIVATE: the activity log is a timeline of when the room is
// active. adminGuard only covers non-GET requests, so this must guard itself.
router.get('/', async (req, res) => {
  if (!isAuthed(req)) return res.status(401).json({ error: 'Login required' })
  try {
    res.json({ entries: await store.read() })
  } catch {
    res.json({ entries: [] })
  }
})

// POST /api/logs — append one entry (kept to the last 1000)
router.post('/', async (req, res) => {
  const { entry } = req.body
  if (!entry) return res.status(400).json({ error: 'No entry provided' })

  try {
    const newEntry = { ...entry, id: Date.now(), savedAt: new Date().toISOString() }
    await store.update(entries => [...entries, newEntry].slice(-1000))
    res.json({ entry: newEntry })
  } catch (err) {
    console.error('Save log entry failed:', err.message)
    res.status(500).json({ error: 'Failed to save log entry' })
  }
})

// DELETE /api/logs — clear the log
router.delete('/', async (_req, res) => {
  try {
    await store.update(() => [])
    res.json({ cleared: true })
  } catch (err) {
    res.status(500).json({ error: 'Failed to clear log' })
  }
})

export default router
