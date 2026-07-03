import path from 'path'
import fs from 'fs/promises'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

/**
 * Create an atomic, serialized JSON map store (filename → label).
 * All read-modify-write operations are queued so two requests can never
 * interleave and clobber each other.
 */
function createStore(filename) {
  const FILE     = path.join(__dirname, '..', filename)
  const TMP_FILE = FILE + '.tmp'
  const BAK_FILE = FILE + '.bak'

  let queue = Promise.resolve()

  /**
   * Read the map. Returns {} ONLY when the file legitimately doesn't exist.
   * On a parse error (e.g. a mid-write partial read or corruption) this THROWS,
   * so callers abort instead of overwriting good data with an empty object.
   */
  async function read() {
    let raw
    try {
      raw = await fs.readFile(FILE, 'utf-8')
    } catch (err) {
      if (err.code === 'ENOENT') return {}
      throw err
    }
    const text = raw.replace(/^﻿/, '').trim() // strip BOM/whitespace
    if (text === '') return {}
    try {
      return JSON.parse(text)
    } catch (err) {
      throw new Error(`${filename} is corrupt — refusing to overwrite: ${err.message}`)
    }
  }

  /** Atomic write: write a temp file, then rename over the real one. */
  async function writeAtomic(obj) {
    const json = JSON.stringify(obj, null, 2)
    await fs.writeFile(TMP_FILE, json, 'utf-8')
    await fs.rename(TMP_FILE, FILE)
  }

  /**
   * Run a read-modify-write transaction safely and serially.
   * `mutate(map)` receives a copy, returns the new map object.
   * Keeps a .bak of the previous good state before each write.
   */
  function update(mutate) {
    queue = queue.then(async () => {
      const current = await read()
      const next    = mutate({ ...current })
      // Safety net: never silently wipe a populated file down to empty.
      if (Object.keys(current).length > 5 && Object.keys(next).length === 0) {
        throw new Error(`Refusing to clear all of ${filename} in one operation`)
      }
      await fs.copyFile(FILE, BAK_FILE).catch(() => {}) // best-effort backup
      await writeAtomic(next)
      return next
    })
    // Make sure a failed transaction doesn't poison the queue for the next caller
    const result = queue
    queue = queue.catch(() => {})
    return result
  }

  return { read, update }
}

// Human-confirmed labels — the ONLY source of training data.
const labelStore = createStore('labels.json')
export const readLabels   = labelStore.read
export const updateLabels = labelStore.update

// Agent-predicted labels awaiting human review in Label Studio. Kept separate
// from labels.json so the model's own (possibly wrong) predictions can never
// feed back into training — see uploadAndAlert in agent/capture.mjs.
const autoLabelStore = createStore('autolabels.json')
export const readAutoLabels   = autoLabelStore.read
export const updateAutoLabels = autoLabelStore.update
