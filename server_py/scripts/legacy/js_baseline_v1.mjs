/**
 * LEGACY. Reproduces the ethogram-v1 tfjs baseline number (74.4%) and nothing
 * else. It is NOT part of the current pipeline and nothing in training/ reads
 * its output any more.
 *
 * Why it was retired: the tfjs model in server/model/ predicts the five v1
 * classes and physically cannot predict ethogram v2's seven, so comparing a v2
 * retrain against it is meaningless. training/evaluate.py's accuracy gate used
 * to do exactly that. See docs/upgrade-plan.md for what replaced it (a
 * `static_logreg` arm on the same frozen embeddings, retrained per fold).
 *
 * Reads server/model/ and server/recordings/ read-only, plus a v1-schema
 * manifest. Writes only models/js_baseline.v1.json.
 *
 * Usage (from server_py/scripts/legacy/):
 *   node js_baseline_v1.mjs
 *   node js_baseline_v1.mjs --manifest &lt;path&gt; --labels a,b,c --out &lt;path&gt;
 *
 * ffmpeg is resolved from $FFMPEG, then @ffmpeg-installer/ffmpeg, then PATH.
 */
import { execFileSync } from 'child_process'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import express from 'express'

await import('@tensorflow/tfjs-node') // registers the native 'tensorflow' backend
const tf = await import('@tensorflow/tfjs')
const mobilenetModule = await import('@tensorflow-models/mobilenet')

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(__dirname, '..', '..', '..')    // BunnyTracker/ (now two levels deeper)

// Minimal --flag value parser; no dependency needed for four options.
const argv = process.argv.slice(2)
const arg = (name, fallback) => {
  const i = argv.indexOf(`--${name}`)
  return i !== -1 && argv[i + 1] ? argv[i + 1] : fallback
}
const SERVER_MODEL_DIR = path.join(ROOT, 'server', 'model')
const RECORDINGS_DIR = path.join(ROOT, 'server', 'recordings')
// Defaults point at the archived v1 cache, since the live models/cache/ is
// regenerated under v2 and its manifest has neither the v1 labels nor the
// schema-1 `val` key this script reads.
const MANIFEST_PATH = arg('manifest',
  path.join(ROOT, 'server_py', 'models', 'archive', 'v1-ethogram', 'cache', 'manifest.json'))
// Renamed from js_baseline.json so no v2 code can pick it up by accident.
const OUT_PATH = arg('out', path.join(ROOT, 'server_py', 'models', 'js_baseline.v1.json'))

// Was hardcoded to @ffmpeg-installer/win32-x64/ffmpeg.exe, which does not
// exist off Windows. Resolve in order and fail with an actionable message.
function resolveFfmpeg() {
  if (process.env.FFMPEG) return process.env.FFMPEG
  const installed = path.join(ROOT, 'server', 'node_modules', '@ffmpeg-installer', 'ffmpeg', 'index.js')
  if (fs.existsSync(installed)) {
    try { return JSON.parse(execFileSync('node', ['-p', `JSON.stringify(require('${installed}').path)`], { encoding: 'utf8' })) }
    catch { /* fall through */ }
  }
  return 'ffmpeg' // rely on PATH
}
const FFMPEG = resolveFfmpeg()

// The v1 vocabulary, in v1 order. Overridable so the script stays honest about
// which ethogram it is scoring rather than hardcoding a list that silently
// disagrees with shared/labels.py.
const LABELS = arg('labels', 'grooming,normal,standing,yawn,zoomies').split(',')
const FRAMES_PER_CLIP = 8
const FRAME_SIZE = 224

function getDuration(filePath) {
  let stderr = ''
  try {
    execFileSync(FFMPEG, ['-i', filePath], { encoding: 'utf8' })
  } catch (err) {
    // ffmpeg exits non-zero with no output file given; stderr still has the banner
    stderr = err.stderr?.toString() ?? ''
  }
  const m = stderr.match(/Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)/)
  if (!m) return 0
  return Number(m[1]) * 3600 + Number(m[2]) * 60 + Number(m[3])
}

function readFrame(filePath, timestamp) {
  const args = [
    '-ss', timestamp.toFixed(3),
    '-i', filePath,
    '-frames:v', '1',
    '-vf', `scale=${FRAME_SIZE}:${FRAME_SIZE}`,
    '-f', 'rawvideo',
    '-pix_fmt', 'rgb24',
    '-loglevel', 'error',
    'pipe:1',
  ]
  const buf = execFileSync(FFMPEG, args, { maxBuffer: 1024 * 1024 * 32 })
  const expected = FRAME_SIZE * FRAME_SIZE * 3
  if (buf.length !== expected) {
    throw new Error(`expected ${expected} bytes, got ${buf.length} for ${filePath} @ ${timestamp}`)
  }
  return buf
}

function extractFrames(filePath, n) {
  const duration = getDuration(filePath)
  const denom = (n - 1) || 1
  const frames = []
  for (let i = 0; i < n; i++) {
    const t = duration > 0 ? (i / denom) * duration * 0.95 : 0
    frames.push(readFrame(filePath, t))
  }
  return frames
}

function features(embeddings, withStd) {
  const dim = embeddings[0].length
  const feat = new Float32Array(withStd ? dim * 2 : dim)
  for (const emb of embeddings) for (let i = 0; i < dim; i++) feat[i] += emb[i]
  for (let i = 0; i < dim; i++) feat[i] /= embeddings.length
  if (withStd) {
    for (const emb of embeddings) {
      for (let i = 0; i < dim; i++) {
        const d = emb[i] - feat[i]
        feat[dim + i] += d * d
      }
    }
    for (let i = 0; i < dim; i++) feat[dim + i] = Math.sqrt(feat[dim + i] / embeddings.length)
  }
  return feat
}

async function main() {
  const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf8'))
  const valClips = manifest.val // [{filename, label, ...}]
  console.log(`${valClips.length} held-out clips (from manifest.json)`)

  // Serve server/model/ ourselves so mobilenet.load's modelUrl + inputRange works
  // without a running server (same trick as server/scripts/bench-inference.mjs).
  const app = express()
  app.use('/model', express.static(SERVER_MODEL_DIR))
  const srv = await new Promise(res => { const s = app.listen(0, '127.0.0.1', () => res(s)) })
  const base = `http://127.0.0.1:${srv.address().port}`

  await tf.ready()
  console.log(`backend=${tf.getBackend()}`)

  const mobilenet = await mobilenetModule.load({
    version: 2, alpha: 1.0,
    modelUrl: `${base}/model/mobilenet/model.json`,
    inputRange: [0, 1],
  })
  const classifier = await tf.loadLayersModel(`${base}/model/model.json`)
  const inputDim = classifier.inputs[0].shape[1]
  const withStd = inputDim === 2560
  console.log(`classifier input dim ${inputDim} (withStd=${withStd})`)

  const embedFrame = buf => tf.tidy(() => {
    const img = tf.tensor3d(buf, [FRAME_SIZE, FRAME_SIZE, 3], 'int32')
    return Array.from(mobilenet.infer(img, true).dataSync())
  })

  const trueIds = []
  const predIds = []
  const cm = LABELS.map(() => Array(LABELS.length).fill(0))

  for (const { filename, label } of valClips) {
    const filePath = path.join(RECORDINGS_DIR, filename)
    const frames = extractFrames(filePath, FRAMES_PER_CLIP)
    const embeddings = frames.map(embedFrame)
    const feat = features(embeddings, withStd)
    const pred = tf.tidy(() => classifier.predict(tf.tensor2d([Array.from(feat)])).argMax(1).dataSync()[0])

    const trueIdx = LABELS.indexOf(label)
    trueIds.push(trueIdx)
    predIds.push(pred)
    cm[trueIdx][pred]++
    console.log(`  ${filename}  true=${label}  pred=${LABELS[pred]}${label === LABELS[pred] ? '' : '  <-- MISS'}`)
  }

  const acc = trueIds.filter((t, i) => t === predIds[i]).length / trueIds.length
  const perClass = {}
  LABELS.forEach((label, k) => {
    const tp = cm[k][k]
    const support = cm[k].reduce((a, b) => a + b, 0)
    const predTotal = cm.reduce((a, row) => a + row[k], 0)
    perClass[label] = {
      precision: predTotal ? tp / predTotal : 0,
      recall: support ? tp / support : 0,
      support,
    }
  })

  console.log(`\nJS baseline accuracy: ${(acc * 100).toFixed(1)}%`)
  for (const [label, r] of Object.entries(perClass)) {
    console.log(`  ${label.padEnd(10)} precision=${(r.precision * 100).toFixed(1)}%  recall=${(r.recall * 100).toFixed(1)}%  support=${r.support}`)
  }

  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true })
  fs.writeFileSync(OUT_PATH, JSON.stringify({
    ethogram_version: '1.0',
    note: 'Legacy v1 tfjs baseline. Single split, single seed; a selection score, not a held-out estimate.',
    accuracy: acc, confusion_matrix: cm, per_class: perClass, labels: LABELS,
  }, null, 2))
  console.log(`\nWrote ${OUT_PATH}`)

  srv.close()
}

main().catch(err => { console.error(err); process.exit(1) })
