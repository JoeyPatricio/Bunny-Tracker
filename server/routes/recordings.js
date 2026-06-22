import express from 'express'
import multer from 'multer'
import path from 'path'
import fs from 'fs/promises'
import { fileURLToPath } from 'url'

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

router.get('/', async (_req, res) => {
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
    res.status(500).json({ error: 'Failed to list recordings', detail: err.message })
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
    else res.status(500).json({ error: 'Failed to delete', detail: err.message })
  }
})

export default router
