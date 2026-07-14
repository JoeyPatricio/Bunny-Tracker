/**
 * Per-frame inference benchmark: pure-JS tfjs backend vs tfjs-node native.
 *
 * Mirrors the agent's hot path in capture.mjs: MobileNetV2 embedding
 * (infer + dataSync), rolling 8-embedding mean+std features, classifier head.
 * The frame budget is AGENT_FRAME_SECONDS (default 1.5s); if the total here
 * approaches that, the embedding window stops spanning ~12s of wall clock and
 * predictions silently drift off-distribution (see Notes/architecture.md).
 *
 * Usage:
 *   node scripts/bench-inference.mjs js       # pure-JS CPU backend
 *   node scripts/bench-inference.mjs native   # tfjs-node native backend
 *   node scripts/bench-inference.mjs native 50
 *
 * Serves ../model over its own local port, so it runs with the stack down.
 * Run it on the Pi 5 before committing to the migration.
 */
import path from 'path'
import { fileURLToPath } from 'url'
import express from 'express'

const mode = process.argv[2] ?? 'native'
const ITER = Number(process.argv[3] ?? 20)

if (!['js', 'native'].includes(mode)) {
  console.error(`Unknown mode "${mode}" — use js or native`)
  process.exit(1)
}

if (mode === 'native') {
  // Registers the higher-priority 'tensorflow' backend on the shared tfjs core.
  await import('@tensorflow/tfjs-node')
}
const tf = await import('@tensorflow/tfjs')
const mobilenetModule = await import('@tensorflow-models/mobilenet')

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// Serve the model dir ourselves so the benchmark needs no running server.
const app = express()
app.use('/model', express.static(path.join(__dirname, '..', 'model')))
const srv = await new Promise(res => { const s = app.listen(0, '127.0.0.1', () => res(s)) })
const base = `http://127.0.0.1:${srv.address().port}`

await tf.ready()
console.log(`backend=${tf.getBackend()}  node=${process.version}  ${process.platform}/${process.arch}`)

// inputRange [0,1] is required with a custom modelUrl — same as capture.mjs.
const mobilenet = await mobilenetModule.load({
  version: 2, alpha: 1.0,
  modelUrl: `${base}/model/mobilenet/model.json`,
  inputRange: [0, 1],
})
const classifier = await tf.loadLayersModel(`${base}/model/model.json`)
const inputDim = classifier.inputs[0].shape[1]
const withStd = inputDim === 2560
console.log(`models loaded (classifier input ${inputDim})`)

// Deterministic pseudo-random frame so runs are comparable across machines.
const frame = Buffer.alloc(224 * 224 * 3)
let seed = 42
for (let i = 0; i < frame.length; i++) {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff
  frame[i] = seed & 0xff
}

const embedFrame = buf => tf.tidy(() => {
  const img = tf.tensor3d(buf, [224, 224, 3], 'int32')
  return mobilenet.infer(img, true).dataSync()
})

// Timing stand-in for the agent's windowFeatures; the real parity contract
// lives in capture.mjs.
function features(window) {
  const dim = window[0].length
  const feat = new Float32Array(withStd ? dim * 2 : dim)
  for (const emb of window) for (let i = 0; i < dim; i++) feat[i] += emb[i]
  for (let i = 0; i < dim; i++) feat[i] /= window.length
  if (withStd) {
    for (const emb of window) {
      for (let i = 0; i < dim; i++) {
        const d = emb[i] - feat[i]
        feat[dim + i] += d * d
      }
    }
    for (let i = 0; i < dim; i++) feat[dim + i] = Math.sqrt(feat[dim + i] / window.length)
  }
  return feat
}

const window_ = []
const embedMs = [], headMs = []

for (let i = 0; i < ITER + 3; i++) {
  const t0 = performance.now()
  const emb = embedFrame(frame)
  const t1 = performance.now()

  window_.push(emb)
  if (window_.length > 8) window_.shift()
  const feat = features(window_)
  tf.tidy(() => classifier.predict(tf.tensor2d(feat, [1, feat.length])).squeeze().dataSync())
  const t2 = performance.now()

  if (i >= 3) { // first 3 iterations are warmup (kernel compilation, JIT)
    embedMs.push(t1 - t0)
    headMs.push(t2 - t1)
  }
}

const stats = a => {
  const s = [...a].sort((x, y) => x - y)
  const mean = a.reduce((x, y) => x + y) / a.length
  return `mean ${mean.toFixed(1)}ms  median ${s[a.length >> 1].toFixed(1)}ms  max ${s[a.length - 1].toFixed(1)}ms`
}

console.log(`embed (MobileNetV2):  ${stats(embedMs)}`)
console.log(`head  (features+fc):  ${stats(headMs)}`)
console.log(`total per frame:      ${stats(embedMs.map((v, i) => v + headMs[i]))}`)

srv.close()
process.exit(0)
