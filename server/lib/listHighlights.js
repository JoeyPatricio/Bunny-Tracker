import path from 'path'
import fs from 'fs/promises'
import { fileURLToPath } from 'url'
import { readLabels } from './labelStore.js'
import { readHidden } from './hiddenStore.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const RECORDINGS_DIR = path.join(__dirname, '..', 'recordings')

/**
 * List highlight candidates: every non-normal labeled clip on disk, newest
 * first, each annotated with its current demo-page visibility.
 *
 * With { includeHidden: false } (the public demo), clips the owner has hidden
 * are omitted entirely. With { includeHidden: true } (the owner's manager
 * view), they are returned with `hidden: true` so the UI can show and toggle
 * them. Both routes share this so the definition of a highlight lives in one
 * place.
 */
export async function listHighlights({ includeHidden }) {
  const [labels, hidden] = await Promise.all([
    readLabels().catch(() => ({})),
    readHidden().catch(() => new Set()),
  ])
  const files = (await fs.readdir(RECORDINGS_DIR)).filter(f => f.endsWith('.webm'))

  const items = await Promise.all(
    files
      .filter(f => labels[f] && labels[f] !== 'normal')
      .filter(f => includeHidden || !hidden.has(f))
      .map(async (filename) => {
        const stat = await fs.stat(path.join(RECORDINGS_DIR, filename))
        return {
          filename,
          label: labels[filename],
          createdAt: stat.mtime.toISOString(),
          size: stat.size,
          hidden: hidden.has(filename),
        }
      })
  )
  items.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
  return items
}
