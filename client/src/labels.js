// Canonical behavior vocabulary (ethogram v2). Mirrors server/shared/labels.py.
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

// The docs/ethogram.md version these codes implement. Mirrors
// ETHOGRAM_VERSION in server/shared/labels.py.
export const ETHOGRAM_VERSION = '2.0'

// Operational definitions, lifted verbatim from docs/ethogram.md (markdown
// stripped). Rendered beside the label buttons so the coder has the criterion
// in front of them for every clip instead of recalling it: that is what keeps
// the resting_lying/resting_sitting and locomotion/locomotion_rapid
// boundaries stable across an afternoon of coding. null means the ethogram
// gives no such criterion for that code. If the document changes, change this
// text in the same commit and bump ETHOGRAM_VERSION.
export const LABEL_DEFINITION = {
  resting_lying: {
    definition: 'Trunk on the ground. Includes limbs tucked under, limbs outstretched, and contact-lying with a conspecific.',
    includes: null,
    excludes: 'a body-roll in progress (transient, code the surrounding state).',
    onset: 'trunk contacts ground and forward motion ceases.',
    offset: 'trunk lifts off the ground, or locomotion begins.',
  },
  resting_sitting: {
    definition: 'Upright and stationary, rear end and forepaws on the ground, trunk off the ground.',
    includes: 'Includes yawning. Yawns are not coded separately (see the scope decision at the top), and a rabbit that yawns is almost always sitting or lying. Code the posture and ignore the yawn.',
    excludes: null,
    onset: 'forepaws contact ground with trunk raised, motion ceases.',
    offset: 'forepaws leave the ground (becomes rearing), trunk lowers (becomes resting_lying), or locomotion begins.',
  },
  rearing: {
    definition: 'Sitting up on the hindlimbs with both forepaws off the ground.',
    includes: null,
    excludes: 'forepaws raised onto an object while the rabbit leans on it. That is still rearing only if unsupported; if supported, code rearing and flag it in notes (revisit if it turns out to be frequent).',
    onset: 'the second forepaw leaves the ground.',
    offset: 'either forepaw returns to the ground.',
  },
  locomotion: {
    definition: 'Ordinary forward movement: hopping or walking.',
    includes: null,
    excludes: 'rapid running, which is locomotion_rapid.',
    onset: 'first forward displacement of the body.',
    offset: '1 s with no net displacement.',
  },
  locomotion_rapid: {
    definition: 'Running: sustained rapid forward movement, typically a circuit of the room.',
    includes: 'Includes binkies. Frisky hops are not counted separately (see the scope decision at the top); a bout containing them is still locomotion_rapid. A binky performed from standing, with no running bout around it, is coded by the posture it returns to.',
    excludes: null,
    onset: 'gait changes from hopping to fully-stretched running.',
    offset: '1 s at hopping pace or slower.',
    boundary: 'bouts separated by under 2 s are one bout.',
  },
  grooming: {
    definition: 'Self-directed licking, nibbling, or forelimb passes over the body.',
    includes: 'Allogrooming: coded as grooming with an allo modifier, not a separate state. Only meaningful if more than one rabbit is in frame.',
    excludes: 'scratching (hindfoot to own body). That is its own event.',
    onset: 'first forelimb pass over the head, or first contact of tongue to fur. Air-boxing (NC3Rs: "fast forward flicking of forelimbs whilst rabbit sits upright on haunches") reliably precedes a full body groom and is coded as part of the grooming bout.',
    offset: '3 s with no grooming action. Grooming is naturally intermittent; a short gap is within a bout, not between bouts.',
  },
  feeding: {
    definition: 'Ingestion of food or water.',
    includes: 'Feeding, drinking, and nibbling litter are merged unless the food and water stations are separately identifiable in frame.',
    excludes: 'NC3Rs Coprophagy/re-ingestion is not coded. It is a normal and welfare-relevant behaviour but it is not distinguishable from grooming the hindquarters at this camera angle and resolution.',
    onset: 'head lowers to the food source and chewing begins.',
    offset: 'head lifts away from the food source for more than 2 s.',
  },
  out_of_view: {
    definition: 'No rabbit visible, or the focal rabbit is occluded for the majority of the coded interval. Not a behaviour. Excluded from training; counted and reported, because the proportion of unusable footage is a real property of the deployment and belongs in the results.',
    includes: null,
    excludes: null,
    onset: null,
    offset: null,
  },
}

// Folded into LABEL_META so every consumer reads one map. Kept as its own
// table above only because the aligned one-line-per-code layout of LABEL_META
// cannot hold paragraphs.
for (const id of ALL_CODES) Object.assign(LABEL_META[id], LABEL_DEFINITION[id])

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
