import express from 'express'
import nodemailer from 'nodemailer'
import path from 'path'
import { fileURLToPath } from 'url'
import { readMonitorState } from './monitor.js'
import { isRecordingFilename } from '../lib/recordingName.js'
import { VALID_LABELS } from '../lib/validLabels.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const router    = express.Router()

// Every behavior except the resting baseline is worth an alert. Derived from the
// canonical list so a new behavior can't be silently un-alertable.
const ALERT_LABELS = new Set(VALID_LABELS.filter(l => l !== 'normal'))

const LABEL_PHRASE = {
  zoomies:  'doing zoomies 🐇💨',
  yawn:     'yawning 😪',
  grooming: 'grooming 🐾',
  standing: 'standing up 🦘',
}

// Per-label cooldown map — each behavior has its own timer so a yawn 3 minutes
// after zoomies still fires, but the same label can't spam every few seconds.
// A global floor (half the cooldown) still prevents rapid cross-label bursts.
const lastSentAt = {}   // { label: timestamp }
let   lastSentAny = 0   // global floor

function isConfigured() {
  return !!(process.env.EMAIL_USER && process.env.EMAIL_PASS && process.env.NOTIFY_TO)
}

function createTransport() {
  return nodemailer.createTransport({
    service: 'gmail',
    auth: {
      user: process.env.EMAIL_USER,
      pass: process.env.EMAIL_PASS,
    },
  })
}

// GET /api/notify/status
router.get('/status', (_req, res) => {
  res.json({
    configured:      isConfigured(),
    cooldownMinutes: Number(process.env.NOTIFY_COOLDOWN_MINUTES ?? 10),
  })
})

// Shared alert policy: the dashboard alerts toggle, a per-label cooldown, and a
// global floor between any two alerts. Both the email path and the agent's n8n
// webhook path go through this, so turning alerts off or being inside a cooldown
// suppresses BOTH. Reserves the cooldown when it allows, so only call it when you
// actually intend to send.
async function alertGate(label) {
  if (!ALERT_LABELS.has(label)) return { allow: false, reason: 'label not an alert trigger' }

  const { emailAlerts } = await readMonitorState()
  if (!emailAlerts) return { allow: false, reason: 'alerts disabled from dashboard' }

  const cooldownMs  = Number(process.env.NOTIFY_COOLDOWN_MINUTES ?? 10) * 60 * 1000
  const globalFloor = cooldownMs / 2 // minimum gap between ANY two alerts
  const now = Date.now()

  if (lastSentAny && now - lastSentAny < globalFloor) {
    const waitSec = Math.ceil((globalFloor - (now - lastSentAny)) / 1000)
    return { allow: false, reason: `global floor, ${waitSec}s remaining` }
  }
  if (lastSentAt[label] && now - lastSentAt[label] < cooldownMs) {
    const waitMin = Math.ceil((cooldownMs - (now - lastSentAt[label])) / 60000)
    return { allow: false, reason: `cooldown (${label}), ${waitMin}m remaining` }
  }

  lastSentAny = now
  lastSentAt[label] = now
  return { allow: true }
}

// POST /api/notify/gate — apply the alert policy (toggle + cooldown) and reserve
// it. The agent calls this before firing the n8n webhook so the webhook obeys
// the same policy as email. Returns { allow, reason }.
router.post('/gate', async (req, res) => {
  res.json(await alertGate(req.body?.label))
})

// POST /api/notify
// Body: { label: string, confidence?: number, filename?: string }
router.post('/', async (req, res) => {
  const { label, confidence, filename } = req.body

  if (!isConfigured()) {
    return res.json({ sent: false, reason: 'email not configured, add .env' })
  }

  // Validate the attachment name before reserving the cooldown.
  let clipPath = null
  if (filename) {
    if (!isRecordingFilename(filename)) {
      return res.status(400).json({ sent: false, error: 'Invalid filename' })
    }
    clipPath = path.join(__dirname, '..', 'recordings', filename)
  }

  const gate = await alertGate(label)
  if (!gate.allow) return res.json({ sent: false, reason: gate.reason })

  try {
    const phrase    = LABEL_PHRASE[label] ?? label
    const confStr   = confidence ? ` (${confidence}% confidence)` : ''
    const subject   = `🐇 BunnyCam Alert: bunny is ${phrase}`
    const text      = `Your bunny is ${phrase}${confStr}!\n\nBunnyCam detected this at ${new Date().toLocaleTimeString()}.`

    const mailOptions = {
      from:    `"BunnyCam 🐇" <${process.env.EMAIL_USER}>`,
      to:      process.env.NOTIFY_TO,
      subject,
      text,
      ...(clipPath ? { attachments: [{ filename, path: clipPath }] } : {}),
    }

    const transporter = createTransport()
    await transporter.sendMail(mailOptions)

    res.json({ sent: true, label, hasAttachment: !!filename })
  } catch (err) {
    console.error('Email send failed:', err.message)
    res.status(500).json({ sent: false, error: err.message })
  }
})

export default router
