import path from 'path'
import fs from 'fs/promises'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const FILE = path.join(__dirname, '..', 'hidden.json')

// Set of recording filenames the owner has hidden from the public demo page.
export async function readHidden() {
  try {
    return new Set(JSON.parse(await fs.readFile(FILE, 'utf-8')))
  } catch {
    return new Set()
  }
}

export async function setHidden(filename, hidden) {
  const set = await readHidden()
  if (hidden) set.add(filename)
  else set.delete(filename)
  await fs.writeFile(FILE, JSON.stringify([...set]))
  return set
}
