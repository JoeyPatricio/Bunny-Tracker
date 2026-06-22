import express from 'express'
import multer from 'multer'
import path from 'path'
import fs from 'fs/promises'
import { fileURLToPath } from 'url'
import { isAuthed } from './auth.js'
import { readLabels } from '../lib/labelStore.js'
import { readHidden } from '../lib/hiddenStore.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const RECORDINGS_DIR = path.join(__dirname, '..', 'recordings')
const SEG_DIR        = path.join(__dirname, '..', 'agent', 'segments')

const router = express.Router()

await fs.mkdir(RECORDINGS_DIR, { recursive: true })

// POST /api/recordings/grab — save the agent's most recent finished segment
// as a plain (unlabeled) recording. Used by the live-feed Record button.
router.post('/grab', async (_req, res) => {
  try {
    const files = (await fs.readdir(SEG_DIR).catch(() => []))
      .filter(f => f.endsWith('.webm')).sort()
    const seg = files.length >= 2 ? files[files.length - 2] : files[0]
    if (!seg) return res.status(409).json({ error: 'No segment available — is the agent running?' })

    const stamp    = new Date().toISOString().replace(/[:.]/g, '-')
    const filename = `recording-manual-${stamp}.webm`
    await fs.copyFile(path.join(SEG_DIR, seg), path.join(RECORDINGS_DIR, filename))
    res.json({ filename })
  } catch (err) {
    res.status(500).json({ error: 'Grab failed', detail: err.message })
  }
})

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, RECORDINGS_DIR),
  filename: (_req, _file, cb) => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
    cb(null, `recording-${timestamp}.webm`)
  }
})

const upload = multer({
  storage,
  fileFilter: (_req, file, cb) => {
    if (file.mimetype.startsWith('video/')) cb(null, true)
    else cb(new Error('Only video files are allowed'))
  }
})

// GET /api/recordings/highlights — PUBLIC: only non-normal labeled clips, for
// the demo page. Never exposes the full archive or unlabeled/resting footage.
router.get('/highlights', async (_req, res) => {
  try {
    const labels = await readLabels().catch(() => ({}))
    const hidden = await readHidden().catch(() => new Set())
    const files  = (await fs.readdir(RECORDINGS_DIR)).filter(f => f.endsWith('.webm'))
    const highlights = await Promise.all(
      files
        .filter(f => labels[f] && labels[f] !== 'normal' && !hidden.has(f))
        .map(async (filename) => {
          const stat = await fs.stat(path.join(RECORDINGS_DIR, filename))
          return { filename, label: labels[filename], createdAt: stat.mtime.toISOString(), size: stat.size }
        })
    )
    highlights.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
    res.json({ recordings: highlights })
  } catch {
    res.status(500).json({ error: 'Failed to list highlights' })
  }
})

// GET /api/recordings — PRIVATE: full archive listing requires login.
router.get('/', async (req, res) => {
  if (!isAuthed(req)) return res.status(401).json({ error: 'Login required' })
  try {
    const files = await fs.readdir(RECORDINGS_DIR)
    const recordingFiles = files.filter(f => f.endsWith('.webm'))

    const recordings = await Promise.all(
      recordingFiles.map(async (filename) => {
        const stat = await fs.stat(path.join(RECORDINGS_DIR, filename))
        return {
          filename,
          createdAt: stat.mtime.toISOString(),
          size: stat.size,
        }
      })
    )

    recordings.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
    res.json({ recordings })
  } catch (err) {
    console.error('List recordings failed:', err.message)
    res.status(500).json({ error: 'Failed to list recordings' })
  }
})

router.post('/', upload.single('video'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No video uploaded' })
  }
  res.json({
    filename: req.file.filename,
    createdAt: new Date().toISOString(),
    size: req.file.size,
  })
})

router.delete('/:filename', async (req, res) => {
  const { filename } = req.params
  if (!filename.startsWith('recording-') || !filename.endsWith('.webm')) {
    return res.status(400).json({ error: 'Invalid filename' })
  }

  const filepath = path.join(RECORDINGS_DIR, filename)
  try {
    await fs.unlink(filepath)
    res.json({ deleted: filename })
  } catch (err) {
    if (err.code === 'ENOENT') res.status(404).json({ error: 'File not found' })
    else { console.error('Delete recording failed:', err.message); res.status(500).json({ error: 'Failed to delete' }) }
  }
})

export default router
