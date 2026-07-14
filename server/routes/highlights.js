import express from 'express'
import { isAuthed } from './auth.js'
import { setHidden } from '../lib/hiddenStore.js'
import { listHighlights } from '../lib/listHighlights.js'
import { isRecordingFilename } from '../lib/recordingName.js'

const router = express.Router()

// GET /api/highlights — PRIVATE owner view: every auto-highlight candidate
// (non-normal labeled clip) with its current homepage visibility.
router.get('/', async (req, res) => {
  if (!isAuthed(req)) return res.status(401).json({ error: 'Login required' })
  try {
    res.json({ highlights: await listHighlights({ includeHidden: true }) })
  } catch (err) {
    console.error('List highlight candidates failed:', err.message)
    res.status(500).json({ error: 'Failed to load highlights' })
  }
})

// POST /api/highlights/:filename  { hidden: boolean } — show/hide on the demo
router.post('/:filename', async (req, res) => {
  const { filename } = req.params
  if (!isRecordingFilename(filename)) {
    return res.status(400).json({ error: 'Invalid filename' })
  }
  try {
    const hide = !!req.body.hidden
    await setHidden(filename, hide)
    res.json({ filename, hidden: hide })
  } catch (err) {
    console.error('Update highlight visibility failed:', err.message)
    res.status(500).json({ error: 'Failed to update highlight' })
  }
})

export default router
