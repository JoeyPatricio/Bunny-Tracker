// Canonical behavior vocabulary (ethogram v2). Mirrors server_py/shared/labels.py.
//
// Definitions and their published sources are in docs/ethogram.md. The
// ethogram is the specification; this file is the implementation. Do not add,
// rename, or reorder a class here without updating that document first.
//
// LABELS order is significant: it matches the classifier's output axis, so
// output index i means LABELS[i]. Reordering requires a retrain.
//
// LABELS holds the model's classes only. `out_of_view` is a coding category,
// not a behavior: it has to be recordable so clips with no visible rabbit stop
// being silently coded as resting, but there is nothing to learn from it, so
// it is excluded from the model's output space.

export const LABELS = [
  'feeding',
  'grooming',
  'locomotion',
  'locomotion_rapid',
  'rearing',
  'resting_lying',
  'resting_sitting',
]

export const CODING_ONLY = ['out_of_view']

export const ALL_CODES = [...LABELS, ...CODING_ONLY]

// The baseline states. v1 had one catch-all `normal`; v2 has two resting
// postures, so anything asking "is this more interesting than rest" tests
// membership here rather than equality against a single label.
export const RESTING_LABELS = ['resting_lying', 'resting_sitting']

// Single source of UI truth per code. The v1 studio hardcoded a button, a stat
// chip, a progress segment, a border class and a badge class per label, which
// is why growing from 5 codes to 8 meant touching the component in a dozen
// places. Everything is generated from this map now.
//
// `key` is the keyboard shortcut. Digits, not letters: A and D are already
// taken by filmstrip navigation, and eight letter mnemonics for these names
// would collide before they were memorable.
export const LABEL_META = {
  feeding:          { name: 'Feeding',          short: 'Feeding',  icon: '🥬', color: '#7dffd4', key: '1' },
  grooming:         { name: 'Grooming',         short: 'Grooming', icon: '🐾', color: '#dc82ff', key: '2' },
  locomotion:       { name: 'Locomotion',       short: 'Hopping',  icon: '🚶', color: '#88aaff', key: '3' },
  locomotion_rapid: { name: 'Rapid locomotion', short: 'Zoomies',  icon: '⚡', color: '#7dff7d', key: '4' },
  rearing:          { name: 'Rearing',          short: 'Rearing',  icon: '🦘', color: '#ff9f3c', key: '5' },
  resting_lying:    { name: 'Resting (lying)',  short: 'Lying',    icon: '😴', color: '#6f86d6', key: '6' },
  resting_sitting:  { name: 'Resting (sitting)',short: 'Sitting',  icon: '🪑', color: '#a0b4ff', key: '7' },
  out_of_view:      { name: 'Out of view',      short: 'No bunny', icon: '👻', color: '#7a8290', key: '0' },
}

export const LABEL_COLOR = Object.fromEntries(
  Object.entries(LABEL_META).map(([id, m]) => [id, m.color]),
)

export const LABEL_NAME = Object.fromEntries(
  Object.entries(LABEL_META).map(([id, m]) => [id, m.name]),
)

// Keyboard shortcut -> label id.
export const KEY_TO_LABEL = Object.fromEntries(
  Object.entries(LABEL_META).map(([id, m]) => [m.key, id]),
)

export const LABEL_PHRASE = {
  feeding: 'Bunny is eating',
  grooming: 'Bunny is grooming',
  locomotion: 'Bunny is hopping about',
  locomotion_rapid: 'Bunny has the zoomies',
  rearing: 'Bunny is rearing up',
  resting_lying: 'Bunny is lying down',
  resting_sitting: 'Bunny is sitting',
  out_of_view: 'No bunny in view',
}
