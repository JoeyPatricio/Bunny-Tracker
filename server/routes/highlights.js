import express from 'express'
import path from 'path'
import fs from 'fs/promises'
import { fileURLToPath } from 'url'
import { isAuthed } from './auth.js'
import { readLabels } from '../lib/labelStore.js'
import { readHidden, setHidden } from '../lib/hiddenStore.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const RECORDINGS_DIR = path.join(__dirname, '..', 'recordings')

const router = express.Router()

// GET /api/highlights — PRIVATE owner view: every auto-highlight candidate
// (non-normal labeled clip) with its current homepage visibility.
router.get('/', async (req, res) => {
  if (!isAuthed(req)) return res.status(401).json({ error: 'Login required' })
  try {
    const labels = await readLabels().catch(() => ({}))
    const hidden = await readHidden().catch(() => new Set())
    const files  = (await fs.readdir(RECORDINGS_DIR)).filter(f => f.endsWith('.webm'))
    const items  = await Promise.all(
      files
        .filter(f => labels[f] && labels[f] !== 'normal')
        .map(async (filename) => {
          const stat = await fs.stat(path.join(RECORDINGS_DIR, filename))
          return {
            filename,
            label: labels[filename],
            createdAt: stat.mtime.toISOString(),
            size: stat.size,
            hidden: hidden.has(filename),
          }
        })
    )
    items.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
    res.json({ highlights: items })
  } catch (err) {
    console.error('List highlight candidates failed:', err.message)
    res.status(500).json({ error: 'Failed to load highlights' })
  }
})

// POST /api/highlights/:filename  { hidden: boolean } — show/hide on the demo
router.post('/:filename', async (req, res) => {
  const { filename } = req.params
  if (!filename.startsWith('recording-') || !filename.endsWith('.webm')) {
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
