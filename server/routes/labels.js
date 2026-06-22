import express from 'express'
import { readLabels, updateLabels } from '../lib/labelStore.js'
import { VALID_LABELS } from '../lib/validLabels.js'
import { isAuthed } from './auth.js'

const router = express.Router()

const VALID = VALID_LABELS

// GET /api/labels/valid — return the canonical label list
router.get('/valid', (_req, res) => res.json({ labels: VALID_LABELS }))

// GET /api/labels — return all labels (PRIVATE: full map requires login)
router.get('/', async (req, res) => {
  if (!isAuthed(req)) return res.status(401).json({ error: 'Login required' })
  try {
    const labels = await readLabels()
    res.json({ labels })
  } catch (err) {
    res.status(500).json({ error: 'Failed to read labels', detail: err.message })
  }
})

// POST /api/labels/:filename — set label for a clip
router.post('/:filename', async (req, res) => {
  const { filename } = req.params
  const { label } = req.body

  if (!filename.startsWith('recording-') || !filename.endsWith('.webm')) {
    return res.status(400).json({ error: 'Invalid filename' })
  }
  if (!VALID.includes(label)) {
    return res.status(400).json({ error: `Invalid label: ${label}` })
  }

  try {
    await updateLabels(labels => { labels[filename] = label; return labels })
    res.json({ filename, label })
  } catch (err) {
    res.status(500).json({ error: 'Failed to save label', detail: err.message })
  }
})

// DELETE /api/labels/:filename — remove label from a clip
router.delete('/:filename', async (req, res) => {
  const { filename } = req.params
  try {
    await updateLabels(labels => { delete labels[filename]; return labels })
    res.json({ deleted: filename })
  } catch (err) {
    res.status(500).json({ error: 'Failed to delete label', detail: err.message })
  }
})

export default router
