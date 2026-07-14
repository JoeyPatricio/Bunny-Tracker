import path from 'path'
import fs from 'fs/promises'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

const clone   = v => (Array.isArray(v) ? [...v] : { ...v })
const size    = v => (Array.isArray(v) ? v.length : Object.keys(v).length)
const isEmpty = v => size(v) === 0

/**
 * Create an atomic, serialized JSON store for one file. Works for either an
 * object map or an array (pass the matching `defaultValue`).
 *
 * All read-modify-write operations are queued, so two requests can never
 * interleave and clobber each other, and every write goes through a temp file
 * plus rename so a crash mid-write can't leave a half-written file.
 *
 * Options:
 *   defaultValue - returned by read() when the file is missing or blank ({} or [])
 *   guardWipe    - if true, refuse to shrink a populated store (>5 entries) to
 *                  empty in one operation (a safety net for the label files)
 */
export function createStore(filename, { defaultValue = {}, guardWipe = false } = {}) {
  const FILE     = path.join(__dirname, '..', filename)
  const TMP_FILE = FILE + '.tmp'
  const BAK_FILE = FILE + '.bak'

  let queue = Promise.resolve()

  /**
   * Read the stored value. Returns a copy of `defaultValue` ONLY when the file
   * legitimately doesn't exist or is blank. On a parse error (a mid-write
   * partial read or corruption) this THROWS, so callers abort instead of
   * overwriting good data with an empty value.
   */
  async function read() {
    let raw
    try {
      raw = await fs.readFile(FILE, 'utf-8')
    } catch (err) {
      if (err.code === 'ENOENT') return clone(defaultValue)
      throw err
    }
    const text = raw.replace(/^﻿/, '').trim() // strip BOM/whitespace
    if (text === '') return clone(defaultValue)
    try {
      return JSON.parse(text)
    } catch (err) {
      throw new Error(`${filename} is corrupt, refusing to overwrite: ${err.message}`)
    }
  }

  /** Atomic write: write a temp file, then rename over the real one. */
  async function writeAtomic(value) {
    const json = JSON.stringify(value, null, 2)
    await fs.writeFile(TMP_FILE, json, 'utf-8')
    await fs.rename(TMP_FILE, FILE)
  }

  /**
   * Run a read-modify-write transaction safely and serially.
   * `mutate(copy)` receives a copy of the current value and returns the new one.
   * Keeps a .bak of the previous good state before each write.
   */
  function update(mutate) {
    queue = queue.then(async () => {
      const current = await read()
      const next    = mutate(clone(current))
      if (guardWipe && size(current) > 5 && isEmpty(next)) {
        throw new Error(`Refusing to clear all of ${filename} in one operation`)
      }
      await fs.copyFile(FILE, BAK_FILE).catch(() => {}) // best-effort backup
      await writeAtomic(next)
      return next
    })
    // A failed transaction must not poison the queue for the next caller
    const result = queue
    queue = queue.catch(() => {})
    return result
  }

  return { read, update }
}
