/**
 * dedupe-recordings.mjs
 * ─────────────────────
 * Finds byte-identical recordings (same MD5), keeps one canonical copy per
 * group, and moves the rest into recordings/_duplicates/ (reversible — nothing
 * is deleted). Duplicates exist for two reasons:
 *
 *   1. Curated imports saved twice under different src numbers
 *      (e.g. stand-src07-01 ≡ stand-src11-01). Hand labels are trusted:
 *      the canonical copy keeps the label, redundant copies lose theirs.
 *
 *   2. The stale-segment agent bug (fixed in capture.mjs) re-uploaded the same
 *      old segment for days under fresh timestamps, and pre-quarantine
 *      auto-labeling could stamp any of them with a predicted label. Those
 *      labels are NOT trustworthy — they're removed from labels.json and
 *      demoted to review suggestions (autolabels.json) on the canonical copy.
 *
 * Duplicate files are also removed from labels.json so a clip can never appear
 * twice in training (or leak across the train/val split).
 *
 * Usage:
 *   node scripts/dedupe-recordings.mjs           # dry run — report only
 *   node scripts/dedupe-recordings.mjs --apply   # move files + update labels
 */

import crypto from 'crypto'
import path from 'path'
import fs from 'fs/promises'
import { createReadStream } from 'fs'
import { fileURLToPath } from 'url'
import { readLabels, updateLabels, updateAutoLabels } from '../lib/labelStore.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const REC_DIR   = path.join(__dirname, '..', 'recordings')
const QUAR_DIR  = path.join(REC_DIR, '_duplicates')
const APPLY     = process.argv.includes('--apply')

// Timestamped names come from the agent/manual grabs; identical content under
// different timestamps is physically impossible for genuine recordings, so
// any label on such a group came from the buggy auto-label path.
const isTimestamped = f => /\d{4}-\d{2}-\d{2}T\d{2}/.test(f)

function md5(file) {
  return new Promise((resolve, reject) => {
    const h = crypto.createHash('md5')
    createReadStream(file)
      .on('data', d => h.update(d))
      .on('end', () => resolve(h.digest('hex')))
      .on('error', reject)
  })
}

const files = (await fs.readdir(REC_DIR)).filter(f => f.endsWith('.webm'))
console.log(`Hashing ${files.length} recordings…`)

const byHash = new Map()
for (const f of files) {
  const hash = await md5(path.join(REC_DIR, f))
  if (!byHash.has(hash)) byHash.set(hash, [])
  byHash.get(hash).push(f)
}

const groups = [...byHash.values()].filter(g => g.length > 1)
console.log(`Found ${groups.length} duplicate groups (${groups.reduce((a, g) => a + g.length - 1, 0)} redundant files)\n`)

const labels = await readLabels()

const toQuarantine   = []  // filenames to move
const labelsToDrop   = []  // label entries to remove
const suggestionsToAdd = {} // canonical → suspect label for re-review
const labelMoves     = []  // [from, to] move a trusted label onto the canonical
const conflicts      = []

for (const group of groups) {
  const sorted   = [...group].sort()
  const labeled  = sorted.filter(f => labels[f])
  const labelSet = new Set(labeled.map(f => labels[f]))

  if (labelSet.size > 1) {
    conflicts.push({ group: sorted, labels: labeled.map(f => `${f}=${labels[f]}`) })
    continue
  }

  const trusted   = sorted.every(f => !isTimestamped(f)) // curated imports only
  const canonical = trusted ? (labeled[0] ?? sorted[0]) : sorted[0]
  const label     = labelSet.size === 1 ? [...labelSet][0] : null

  console.log(`${trusted ? 'curated' : 'stale-segment'} group — keeping ${canonical}${label ? ` (label: ${label}${trusted ? '' : ' → demoted to suggestion'})` : ''}`)
  for (const f of sorted) {
    if (f === canonical) continue
    console.log(`  quarantine ${f}${labels[f] ? `  (drops label "${labels[f]}")` : ''}`)
    toQuarantine.push(f)
    if (labels[f]) labelsToDrop.push(f)
  }

  if (label) {
    if (trusted) {
      // Keep the hand label, on the canonical file
      if (!labels[canonical]) labelMoves.push([labeled[0], canonical])
    } else {
      // Auto-label era: content provably recycled → label unverified.
      if (labels[canonical]) labelsToDrop.push(canonical)
      suggestionsToAdd[canonical] = label
      console.log(`  → ${canonical} queued for re-review as suggested "${label}"`)
    }
  }
}

if (conflicts.length) {
  console.log('\n⚠ Groups with CONFLICTING labels — resolve by hand, left untouched:')
  for (const c of conflicts) console.log(`  ${c.labels.join('  vs  ')}`)
}

console.log(`\nPlan: quarantine ${toQuarantine.length} files, drop ${new Set(labelsToDrop).size} label entries, ` +
            `move ${labelMoves.length} labels to canonicals, add ${Object.keys(suggestionsToAdd).length} re-review suggestions`)

if (!APPLY) {
  console.log('\nDry run — nothing changed. Re-run with --apply to execute.')
  process.exit(0)
}

await fs.mkdir(QUAR_DIR, { recursive: true })
for (const f of toQuarantine) {
  await fs.rename(path.join(REC_DIR, f), path.join(QUAR_DIR, f))
}

await updateLabels(map => {
  for (const [from, to] of labelMoves) map[to] = map[from]
  for (const f of labelsToDrop) delete map[f]
  for (const f of toQuarantine) delete map[f]
  return map
})
await updateAutoLabels(auto => {
  for (const f of toQuarantine) delete auto[f]
  for (const [file, label] of Object.entries(suggestionsToAdd)) auto[file] = label
  return auto
})

console.log(`\n✅ Applied. Redundant files are in recordings/_duplicates/ (restore by moving back).`)
const finalLabels = await readLabels()
const counts = {}
for (const l of Object.values(finalLabels)) counts[l] = (counts[l] || 0) + 1
console.log('Label counts now:', JSON.stringify(counts))
