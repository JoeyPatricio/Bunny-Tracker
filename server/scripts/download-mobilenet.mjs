/**
 * download-mobilenet.mjs
 * ──────────────────────
 * Downloads the MobileNet v2 (alpha 1.0) TFJS graph model — the exact model
 * @tensorflow-models/mobilenet@2.x uses for {version: 2, alpha: 1.0} — into
 * server/model/mobilenet/ so the app can serve it itself.
 *
 * Why: the dashboard's CSP (connect-src 'self') correctly blocks the browser
 * from fetching models off tfhub.dev, and a self-hosted copy also works
 * offline and loads faster. All three load sites (TrainingStudio, useInference,
 * agent) load from /model/mobilenet/model.json with a CDN fallback.
 *
 * NOTE for loaders: this TFHub model expects inputRange [0, 1]. The mobilenet
 * package only applies that automatically for its built-in URLs, so custom
 * modelUrl loads MUST pass { inputRange: [0, 1] } or embeddings shift and the
 * trained classifier silently breaks.
 *
 * Usage:  node scripts/download-mobilenet.mjs
 */

import path from 'path'
import fs from 'fs/promises'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const OUT_DIR = path.join(__dirname, '..', 'model', 'mobilenet')

// Same URL as MODEL_INFO['2.00']['1.00'] in @tensorflow-models/mobilenet,
// fetched the way loadGraphModel(..., {fromTFHub: true}) does.
const BASE = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2'

async function download(url) {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`)
  return Buffer.from(await res.arrayBuffer())
}

await fs.mkdir(OUT_DIR, { recursive: true })

console.log('Downloading model.json…')
const modelJsonBuf = await download(`${BASE}/model.json?tfjs-format=file`)
const modelJson = JSON.parse(modelJsonBuf.toString('utf8'))

const shardPaths = (modelJson.weightsManifest ?? []).flatMap(m => m.paths)
if (shardPaths.length === 0) throw new Error('No weight shards listed in model.json')

for (const shard of shardPaths) {
  console.log(`Downloading ${shard}…`)
  const buf = await download(`${BASE}/${shard}?tfjs-format=file`)
  await fs.writeFile(path.join(OUT_DIR, shard), buf)
}
await fs.writeFile(path.join(OUT_DIR, 'model.json'), modelJsonBuf)

const files = await fs.readdir(OUT_DIR)
let total = 0
for (const f of files) total += (await fs.stat(path.join(OUT_DIR, f))).size
console.log(`\n✅ Saved ${files.length} files (${(total / 1e6).toFixed(1)} MB) to server/model/mobilenet/`)
console.log('Served at /model/mobilenet/model.json — restart the server if it was running.')
