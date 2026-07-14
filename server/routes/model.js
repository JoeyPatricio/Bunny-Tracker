import express from 'express'
import path from 'path'
import fs from 'fs/promises'
import { fileURLToPath } from 'url'
import { isAuthed } from './auth.js'

const __dirname   = path.dirname(fileURLToPath(import.meta.url))
const MODEL_DIR   = path.join(__dirname, '..', 'model')
const BACKUP_DIR  = path.join(MODEL_DIR, 'backups')

const router = express.Router()

await fs.mkdir(MODEL_DIR,  { recursive: true })
await fs.mkdir(BACKUP_DIR, { recursive: true })

// POST /api/model — save tfjs model (model.json + weights)
// Body: { modelJson: {...}, weightData: [...] }  (weights as base64)
router.post('/', async (req, res) => {
  try {
    const { modelTopology, weightSpecs, weightData, labels } = req.body
    if (!modelTopology || !weightData) {
      return res.status(400).json({ error: 'Missing modelTopology or weightData' })
    }

    const modelJson = {
      modelTopology,
      weightsManifest: [{
        paths: ['weights.bin'],
        weights: weightSpecs,
      }],
    }

    // Back up the current model before overwriting so it can be restored
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
    await fs.copyFile(
      path.join(MODEL_DIR, 'model.json'),
      path.join(BACKUP_DIR, `model-${stamp}.json`)
    ).catch(() => {}) // ok if no previous model exists yet
    await fs.copyFile(
      path.join(MODEL_DIR, 'weights.bin'),
      path.join(BACKUP_DIR, `weights-${stamp}.bin`)
    ).catch(() => {})

    await fs.writeFile(
      path.join(MODEL_DIR, 'model.json'),
      JSON.stringify(modelJson),
      'utf-8'
    )

    // weightData is base64-encoded binary
    const buf = Buffer.from(weightData, 'base64')
    await fs.writeFile(path.join(MODEL_DIR, 'weights.bin'), buf)

    // Save label map so inference knows which index = which class
    if (labels) {
      await fs.writeFile(
        path.join(MODEL_DIR, 'labels.json'),
        JSON.stringify(labels, null, 2),
        'utf-8'
      )
    }

    res.json({ saved: true, path: MODEL_DIR })
  } catch (err) {
    res.status(500).json({ error: 'Failed to save model', detail: err.message })
  }
})

// GET /api/model — PRIVATE: exposes the label map and the list of dated model
// backups. adminGuard only covers non-GET requests, so this must guard itself.
router.get('/', async (req, res) => {
  if (!isAuthed(req)) return res.status(401).json({ error: 'Login required' })
  try {
    const modelPath  = path.join(MODEL_DIR, 'model.json')
    const labelsPath = path.join(MODEL_DIR, 'labels.json')
    await fs.access(modelPath)
    const labels  = JSON.parse(await fs.readFile(labelsPath, 'utf-8').catch(() => 'null'))
    const stat    = await fs.stat(modelPath)
    const backups = (await fs.readdir(BACKUP_DIR).catch(() => []))
      .filter(f => f.startsWith('model-'))
      .sort()
      .reverse()
    res.json({ exists: true, savedAt: stat.mtime.toISOString(), labels, backups })
  } catch {
    res.json({ exists: false })
  }
})

export default router
