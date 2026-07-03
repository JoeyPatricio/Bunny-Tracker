import express from 'express'
import { readLabels, updateLabels, readAutoLabels, updateAutoLabels } from '../lib/labelStore.js'
import { VALID_LABELS } from '../lib/validLabels.js'
import { isAuthed } from './auth.js'

const router = express.Router()

const VALID = VALID_LABELS

// GET /api/labels/valid — return the canonical label list
router.get('/valid', (_req, res) => res.json({ labels: VALID_LABELS }))

// GET /api/labels — return all labels (PRIVATE: full map requires login).
// `labels` holds human-confirmed labels only (the training set); `suggestions`
// holds agent-predicted labels awaiting review in Label Studio.
router.get('/', async (req, res) => {
  if (!isAuthed(req)) return res.status(401).json({ error: 'Login required' })
  try {
    const labels      = await readLabels()
    const suggestions = await readAutoLabels()
    res.json({ labels, suggestions })
  } catch (err) {
    res.status(500).json({ error: 'Failed to read labels', detail: err.message })
  }
})

// POST /api/labels/:filename/suggest — record the agent's predicted label as a
// review suggestion. Deliberately NOT written to labels.json: the model's own
// predictions must never feed back into its training data unreviewed.
router.post('/:filename/suggest', async (req, res) => {
  const { filename } = req.params
  const { label } = req.body

  if (!filename.startsWith('recording-') || !filename.endsWith('.webm')) {
    return res.status(400).json({ error: 'Invalid filename' })
  }
  if (!VALID.includes(label)) {
    return res.status(400).json({ error: `Invalid label: ${label}` })
  }

  try {
    await updateAutoLabels(auto => { auto[filename] = label; return auto })
    res.json({ filename, suggested: label })
  } catch (err) {
    res.status(500).json({ error: 'Failed to save suggestion', detail: err.message })
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
    // The human has reviewed this clip — the agent's suggestion is resolved.
    await updateAutoLabels(auto => { delete auto[filename]; return auto }).catch(() => {})
    res.json({ filename, label })
  } catch (err) {
    res.status(500).json({ error: 'Failed to save label', detail: err.message })
  }
})

// DELETE /api/labels/:filename — remove label (and any suggestion) from a clip
router.delete('/:filename', async (req, res) => {
  const { filename } = req.params
  try {
    await updateLabels(labels => { delete labels[filename]; return labels })
    await updateAutoLabels(auto => { delete auto[filename]; return auto }).catch(() => {})
    res.json({ deleted: filename })
  } catch (err) {
    res.status(500).json({ error: 'Failed to delete label', detail: err.message })
  }
})

export default router
