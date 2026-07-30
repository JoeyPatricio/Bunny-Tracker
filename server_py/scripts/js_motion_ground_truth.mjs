/**
 * Ground truth for motion-parity testing (Phase 3): poolLuma()/motionLevel()
 * copied VERBATIM from server/agent/capture.mjs (read-only reference, not
 * imported — this script never touches the live file or process), run over
 * the exact same frame sequence dump_agent_frames.py produced, to a JSON
 * sidecar of {level, hotCells, lighting} per frame.
 *
 * Usage: node scripts/js_motion_ground_truth.mjs   (run from server_py/scripts/)
 */
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const FIXTURES_DIR = path.join(__dirname, '..', 'tests', 'fixtures')
const FRAMES_BIN = path.join(FIXTURES_DIR, 'motion_fixture_frames.bin')
const OUT_PATH = path.join(FIXTURES_DIR, 'motion_fixture_ground_truth.json')

const FRAME_SIZE = 224 * 224 * 3

// ── Verbatim copy from server/agent/capture.mjs (constants + both functions) ──
const PIXEL_DIFF = 15
const POOL       = 4
const POOLED_W   = 224 / POOL
const GRID       = 8
const CELL_W     = POOLED_W / GRID
const CELL_HOT   = 0.25
const LIGHTING_CELLS = Math.floor(GRID * GRID * 0.7)
let prevPooled = null

function poolLuma(frameBuf) {
  const out = new Float32Array(POOLED_W * POOLED_W)
  for (let y = 0; y < 224; y++) {
    const row = ((y / POOL) | 0) * POOLED_W
    for (let x = 0; x < 224; x++) {
      const i = (y * 224 + x) * 3
      out[row + ((x / POOL) | 0)] += (frameBuf[i] + 2 * frameBuf[i + 1] + frameBuf[i + 2]) / 4
    }
  }
  for (let i = 0; i < out.length; i++) out[i] /= POOL * POOL
  return out
}

function motionLevel(frameBuf) {
  const pooled = poolLuma(frameBuf)
  if (!prevPooled) { prevPooled = pooled; return { level: 0, hotCells: 0, lighting: false } }

  const n = pooled.length
  const diffs = new Float32Array(n)
  for (let i = 0; i < n; i++) diffs[i] = pooled[i] - prevPooled[i]
  prevPooled = pooled

  const median = Float32Array.from(diffs).sort()[n >> 1]

  let changed = 0
  const cellChanged = new Uint16Array(GRID * GRID)
  for (let i = 0; i < n; i++) {
    if (Math.abs(diffs[i] - median) > PIXEL_DIFF) {
      changed++
      const cy = (((i / POOLED_W) | 0) / CELL_W) | 0
      const cx = ((i % POOLED_W) / CELL_W) | 0
      cellChanged[cy * GRID + cx]++
    }
  }

  let hotCells = 0
  for (let c = 0; c < cellChanged.length; c++) {
    if (cellChanged[c] >= CELL_W * CELL_W * CELL_HOT) hotCells++
  }

  const lighting = hotCells >= LIGHTING_CELLS
  return { level: lighting ? 0 : (changed / n) * 100, hotCells, lighting }
}
// ── End verbatim copy ──────────────────────────────────────────────────────

const buf = fs.readFileSync(FRAMES_BIN)
const nFrames = buf.length / FRAME_SIZE
if (!Number.isInteger(nFrames)) {
  throw new Error(`frame buffer size ${buf.length} is not a multiple of FRAME_SIZE ${FRAME_SIZE}`)
}

const results = []
for (let i = 0; i < nFrames; i++) {
  const frame = buf.subarray(i * FRAME_SIZE, (i + 1) * FRAME_SIZE)
  results.push(motionLevel(frame))
}

fs.writeFileSync(OUT_PATH, JSON.stringify(results, null, 2))
console.log(`Wrote ground truth for ${nFrames} frames to ${OUT_PATH}`)
console.log(results.slice(0, 5))
