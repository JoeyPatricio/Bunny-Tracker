# Bunny Tracker ethogram (v2)

Status: **v2 states are in force.** `server_py/shared/labels.py` and
`client/src/labels.js` declare the seven states plus `out_of_view`, and all 112
v1 clips were unlabeled on 2026-09-25 for recoding against these definitions.
The v1 labels are archived in `server/labels.v1.json`.

**Scope decision, 2026-09-25:** the state/event split described below was NOT
implemented. Yawn is folded into `resting_sitting` and binky into
`locomotion_rapid`, and labels are stored as a flat `{filename: state}` string
map rather than the `{state, events[], quality}` object. The Events section is
kept as the upgrade path, not as a description of what ships. What this costs:
yawn and binky frequencies cannot be reported, and a clip where the rabbit
yawns while sitting is indistinguishable from one where it only sits. Adding
events later does not require recoding the states, only a second pass over the
clips, so this is deferrable without waste.

This document is the coding manual. It is meant to be frozen before any clip
is labeled under it, handed to a second observer unchanged, and cited in a
methods section. If a definition has to change after labeling starts, bump the
version and re-code everything.

## Source

Definitions are taken, and where noted collapsed, from the **NC3Rs / IAT / RSPCA
laboratory rabbit ethogram**, itself a compilation of:

- Morton DB et al. (2003). Refinements in rabbit husbandry: Second report of the
  BVAAWF/FRAME/RSPCA/UFAW joint working group on refinement. *Laboratory Animals*
  27: 301-329, Appendix 1, pp. 325-7.
- Held SDE, Turner RJ, Wootton RJ (2001). The behavioural repertoire of
  non-breeding group-housed female laboratory rabbits (*Oryctolagus cuniculus*).
  *Animal Welfare* 10(4): 437-443.
- Gunn D, Morton DB (1995). Inventory of the behaviour of New Zealand White
  rabbits in laboratory cages. *Applied Animal Behaviour Science* 45(3-4): 277-92.

Quoted definitions below are from the NC3Rs compilation. Deviations are marked
**[deviation]** with the reason.

## Why v1 is not usable as-is

| v1 label | Problem |
|---|---|
| `normal` | Not a behaviour. It is a residual class meaning "none of the other four", so it currently absorbs resting, locomotion, feeding, and frames with no rabbit in them. It cannot be defined, so it cannot be coded reliably, so it cannot be reported. |
| `standing` | Non-standard term. The published category is *rearing*. |
| `zoomies` | Non-standard term, and conflates a state (rapid locomotion) with an event (the binky/frisky hop that punctuates it). |
| `yawn` | A sub-second point event competing in the same softmax as multi-second states. It is structurally disadvantaged, which is most of why n=5 performs as badly as it does. |
| `grooming` | Closest to usable. Needs an explicit onset/offset criterion and a self/allo distinction. |

The single most important change is that **`normal` is deleted** and replaced by
the behaviours it was hiding.

## Prior art: how the one comparable paper codes rabbit behaviour

Adedeji et al. (2023), the closest published work (see the README references),
classify rabbit behaviour into seven vernacular classes: digging, humping and
mating, eating, periscoping, jumping and binkying, laying, flopping and
loafing. Their scheme is worth reading as a counterexample rather than a
model:

- No operational definitions, no source ethogram, and no inter-observer
  reliability. The class names are the whole specification.
- The categories are not mutually exclusive. "Laying" and "flopping and
  loafing" describe overlapping postures, and the paper reports its classifier
  confusing visually similar classes. "Humping and mating" is a social
  behaviour in an otherwise postural scheme.
- States and events are mixed in one softmax, the same problem v1 has here.
  "Jumping and binkying" is a point event sitting alongside "laying", a state.

Two of their classes map onto v2 and are worth keeping in view:
`periscoping` is the same posture as `rearing` (NC3Rs "Rearing alert"), and
`digging` is a real NC3Rs category ("prolonged paw-scraping at deep
substrate") that this ethogram omits only because there is no dig-friendly
substrate in frame. Add `digging` if that changes.

Their classifier also runs on single still images, so it has no way to express
the state/event distinction even in principle. Working from video is what
makes the split below available to this project.

## Design decisions

**1. Two streams, not one softmax.** States and events are coded and modeled
separately.

- **States** are mutually exclusive and exhaustive. Exactly one applies at any
  instant. They have meaningful durations.
- **Events** are brief (under ~2 s), are counted rather than timed, and occur
  *during* a state. A yawn happens while the rabbit is sitting; both are true.

This is standard ethological practice and it also fixes the modeling mismatch:
states are a classification problem scored with macro-F1 and a confusion matrix,
events are a detection problem scored with precision/recall at a temporal
tolerance. Rare events stop being a broken softmax class and become a
legitimately hard detection task, which is a better result to report either way.

**2. Granularity is capped by the sensor, and that is stated openly.** The
NC3Rs ethogram separates `Sleep` from `Dozing` from `Lying` by eye aperture and
ear position. At 224x224 from a room camera those are not resolvable. Categories
the camera cannot separate are collapsed, and each collapse is recorded below.
A reviewer will accept a coarser ethogram that is honestly justified. They will
not accept fine distinctions that the observer could not actually have made.

**3. `out_of_view` is a coding category, not a behaviour.** Clips where no
rabbit is visible, or the rabbit is occluded for the majority of the clip, get
this code and are excluded from training. Right now they are almost certainly
sitting in `normal`.

## States

Exactly one per coded interval. Ordered as they should appear in `LABELS`.

### `resting_lying`
Trunk on the ground. Includes limbs tucked under, limbs outstretched, and
contact-lying with a conspecific.

> NC3Rs *Resting*: "Lying, limbs tucked under: resting with trunk on ground,
> hindlimbs tucked under the forelimbs lying under or forward stretched from
> body. Lying, limbs outstretched: resting with body trunk on ground, all four
> limbs outstretched and belly exposed."

**[deviation]** NC3Rs `Sleep` ("lying or sitting with both eyes closed") and
`Dozing` ("eyes slightly to half open and one or both ears erect") are merged
into `resting_lying` / `resting_sitting`. Eye aperture is not resolvable at this
camera resolution and distance. Consequence: this study cannot report sleep.

- **Onset:** trunk contacts ground and forward motion ceases.
- **Offset:** trunk lifts off the ground, or locomotion begins.
- **Not:** a body-roll in progress (transient, code the surrounding state).

### `resting_sitting`
Upright and stationary, rear end and forepaws on the ground, trunk off the
ground.

> NC3Rs *Resting*: "Sitting: in upright stationary position, with rear end and
> forepaws on ground and ears down. Sitting alert: as above, but with ears erect."

**[deviation]** `Sitting` and `Sitting alert` are merged. Ear position is the
only discriminator and it is unreliable at this resolution. **Decide this from
your own footage before freezing:** if ear orientation is in fact legible in
your camera's frames, split them, because the alert posture is welfare-relevant
(it indicates vigilance and therefore possible stressors).

- **Onset:** forepaws contact ground with trunk raised, motion ceases.
- **Offset:** forepaws leave the ground (becomes `rearing`), trunk lowers
  (becomes `resting_lying`), or locomotion begins.
- **Includes yawning.** Yawns are not coded separately (see the scope decision
  at the top), and a rabbit that yawns is almost always sitting or lying. Code
  the posture and ignore the yawn.

### `rearing`
Sitting up on the hindlimbs with **both** forepaws off the ground.

> NC3Rs *Rearing*: "Sitting up on hind-limbs with both forepaws off the ground;
> ears partly or fully down." *Rearing alert*: "As above but with ears erect."

**[deviation]** `Rearing` and `Rearing alert` merged, same reason as above.
Replaces v1 `standing`.

- **Onset:** the second forepaw leaves the ground.
- **Offset:** either forepaw returns to the ground.
- **Not:** forepaws raised onto an object while the rabbit leans on it. That is
  still `rearing` only if unsupported; if supported, code `rearing` and flag it
  in notes (revisit if it turns out to be frequent).

### `locomotion`
Ordinary forward movement: hopping or walking.

> NC3Rs *Locomotory*: "Hopping: forward movement achieved by alternate extension
> of fore and hindlimbs. Distinguished from running by its slower speed and
> shorter distance covered per forward jump."

- **Onset:** first forward displacement of the body.
- **Offset:** 1 s with no net displacement.
- **Not:** rapid running, which is `locomotion_rapid`.

This class does not exist in v1. It is currently inside `normal`, and it is
almost certainly a large part of why `normal` and `zoomies` confuse.

### `locomotion_rapid`
Running: sustained rapid forward movement, typically a circuit of the room.

> NC3Rs *Locomotory*: "Running: rapid forward movement achieved by alternate,
> fully-stretched extension of fore and hindlimbs."

Replaces the state half of v1 `zoomies`. In the literature a bout of this
interspersed with binkies is a frenetic random activity period (FRAP). Binkies
within it are coded separately as events (see `binky`).

- **Onset:** gait changes from hopping to fully-stretched running.
- **Offset:** 1 s at hopping pace or slower.
- **Boundary rule:** bouts separated by under 2 s are one bout.
- **Includes binkies.** Frisky hops are not counted separately (see the scope
  decision at the top); a bout containing them is still `locomotion_rapid`. A
  binky performed from standing, with no running bout around it, is coded by
  the posture it returns to.

### `grooming`
Self-directed licking, nibbling, or forelimb passes over the body.

> NC3Rs *Grooming*: "Self-groom: A full body groom is usually preceded by
> air-boxing. The forelimbs are the licked and passed over the head and ears,
> prior to licking/nibbling of fur over the rest of the body. Allogroom: Rabbits
> may also lick the fur of another rabbit."

- **Onset:** first forelimb pass over the head, or first contact of tongue to
  fur. Air-boxing (NC3Rs: "fast forward flicking of forelimbs whilst rabbit sits
  upright on haunches") reliably precedes a full body groom and is coded as part
  of the grooming bout.
- **Offset:** 3 s with no grooming action. Grooming is naturally intermittent;
  a short gap is within a bout, not between bouts.
- **Allogrooming:** coded as `grooming` with an `allo` modifier, not a separate
  state. Only meaningful if more than one rabbit is in frame.
- **Not:** `scratching` (hindfoot to own body). That is its own event.

### `feeding`
Ingestion of food or water.

> NC3Rs *Feeding*: "Taking food material into mouth and chewing and swallowing,
> from food dispenser or floor." *Drinking*: "Lapping up water with tongue."
> *Nibbling litter*: "Picking up and nibbling litter, with or without ingestion."

**[deviation]** Feeding, drinking, and nibbling litter are merged unless the
food and water stations are separately identifiable in frame. If they are,
split into `feeding` and `drinking`, because drinking frequency is directly
relevant to the water-intake stretch goal in the roadmap.

**[deviation]** NC3Rs `Coprophagy/re-ingestion` is not coded. It is a normal and
welfare-relevant behaviour but it is not distinguishable from grooming the
hindquarters at this camera angle and resolution.

- **Onset:** head lowers to the food source and chewing begins.
- **Offset:** head lifts away from the food source for more than 2 s.

### `out_of_view`
No rabbit visible, or the focal rabbit is occluded for the majority of the coded
interval. Not a behaviour. Excluded from training; counted and reported, because
the proportion of unusable footage is a real property of the deployment and
belongs in the results.

## Events (not implemented, upgrade path)

Counted, not timed. Coded independently of the concurrent state. More than one
may occur in a clip. **None of this is in the codebase**; see the scope
decision at the top. It is kept because the argument for it does not go away,
and because adding it later is additive rather than a recode.

### `yawn`
Wide opening of the mouth with the head raised, jaw fully extended, typically
1 to 2 s, usually from a sitting or lying posture.

**Not in the NC3Rs compilation.** Yawning is conventionally grouped with
stretching as a comfort behaviour; the likely source in this lineage is the
Gunn & Morton (1995) inventory, which I have not verified directly. **Before
freezing this document, check Gunn & Morton (1995) and either cite it or mark
`yawn` as a locally defined category.** A locally defined category is
acceptable if it is declared as such.

- **Count:** one per mouth opening. A double yawn within 2 s counts as two.

### `binky`
The frisky hop or play gambol.

> NC3Rs *Play*: "Play gambolling or 'frisky hop': forward hopping/jumping
> accompanied by sideways tossing of the head/ears, shaking/twisting the body or
> kicking out with the feet."

- **Count:** one per airborne twist or kick-out. A running circuit containing
  three twists is `locomotion_rapid` with three `binky` events.

### `thumping` (optional)
> NC3Rs *Thumping*: "Loud thumping of the ground with the hind-foot (feet),
> usually when alarmed."

Include only if the capture pipeline records audio. It currently does not, and
thumping is much easier to score from audio than from video. Listed here so the
decision is recorded rather than silently skipped.

### `head_flick` and `scratching` (optional)
> NC3Rs *Play*: "Head flicking: flicking head sideways."
> NC3Rs *Scratching*: "Scratching at own body with a hindfoot."

Low priority. Add only if they turn out to be frequent enough to score
reliably.

## Annotation protocol

**Unit of coding.** Clips are 8 s (clipping mode) or 12 s (alert mode). The
rabbit can change state mid-clip, so a single label per clip is a weak label and
must be defined as such.

*Rule for v2:* code the **predominant state**, defined as the state occupying
the largest share of the clip, and only if that share is **at least 50%** of the
clip duration. If no state reaches 50%, code `mixed`. `mixed` clips are excluded
from training and reported as a count. Events are coded for the whole clip
regardless of state.

*Upgrade path:* true interval coding, with onset and offset timestamps per bout,
scored in [BORIS](https://www.boris.unito.it/) (the standard open-source
event-logging tool for ethology). That turns weak clip labels into frame-level
ground truth and makes the state stream a temporal segmentation problem rather
than a clip classification problem. It is strictly better and it is also
strictly more work. Decide before recoding the 112 existing clips, because
recoding twice is the expensive outcome.

**Focal animal.** If more than one rabbit can be in frame, a focal rule is
required. Default: code the rabbit nearest the frame centre at clip onset, and
record its identity. If the individuals are not visually distinguishable, say so
and code at the group level, which weakens but does not invalidate the design.

**Uncertainty.** A `quality` field with values `good` / `poor` / `unusable`.
Poor-quality clips stay in the dataset and are reported; unusable ones are
excluded. Never guess a label to avoid leaving a clip uncoded.

## Inter-observer reliability

Required for publication and impossible to retrofit.

1. Freeze this document. No edits after step 2 begins.
2. Train the second observer on 10 practice clips that are **not** part of the
   dataset. Discuss disagreements on those freely.
3. The second observer independently codes a random 20% sample, stratified by
   the primary observer's v2 labels, blind to those labels.
4. Report **Cohen's kappa**, unweighted (the categories are nominal), overall
   and per class. Target kappa >= 0.70; below 0.60 means a definition is
   ambiguous and needs rewriting, which means going back to step 1.
5. Report event agreement separately, as agreement on counts per clip.

## Migrating the 112 existing clips

Current v1 distribution: `normal` 45, `zoomies` 26, `grooming` 25,
`standing` 11, `yawn` 5.

| v1 | v2 | Can it be mapped automatically? |
|---|---|---|
| `grooming` | `grooming` | Probably, but verify each. |
| `standing` | `rearing` | Probably, but verify each. |
| `yawn` | state unknown + `yawn` event | **No.** The state was never recorded. |
| `zoomies` | `locomotion_rapid` + `binky` events | **No.** Event counts were never recorded, and some may be ordinary `locomotion`. |
| `normal` | `resting_lying` / `resting_sitting` / `locomotion` / `feeding` / `out_of_view` | **No.** This is the whole point. |

**Do not write a migration script.** A scripted mapping would preserve exactly
the residual-class error the redesign exists to remove. Re-view all 112 clips
under the frozen v2 definitions. At roughly 30 s each that is about an hour, and
it buys the ability to state in a methods section that the entire dataset was
coded under one protocol by a trained observer, which a partial migration cannot.

**Keep v1.** Write v2 to a new file rather than overwriting `server/labels.json`,
so the v1-to-v2 confusion table can be reported. Proposed schema, one entry per
clip:

```json
{
  "recording-clip-2026-09-24T14-03-11.webm": {
    "state": "resting_sitting",
    "events": ["yawn"],
    "focal": "bun_a",
    "quality": "good",
    "coder": "primary",
    "ethogram_version": "2.0"
  }
}
```

The bare-string v1 format cannot express an event co-occurring with a state, so
the schema change is forced by the state/event split, not optional.

## Code touchpoints when v2 goes in force

The label vocabulary is declared in five places and consumed in several more.
`LABELS` order is load-bearing: it maps classifier output indices to names.

- `server_py/shared/labels.py:8` (`LABELS`, `LABEL_PHRASE`)
- `client/src/labels.js:7` (`LABELS`, `LABEL_COLOR`, `LABEL_PHRASE`)
- `server_py/app/lib/valid_labels.py:3` (`VALID_LABELS`, server-side validation)
- `server_py/app/routers/notify.py:25` (alert email phrasing)
- `server/model/labels.json` (retired tfjs artifact, read-only)
- `client/src/components/LabelingStudio.jsx` (keybindings, filters, count chips,
  progress bar segments)
- `client/src/components/RecordingGallery.jsx:16` (filter list)
- `server_py/agent/capture.py` (`interest_score` treats index `normal` as the
  uninteresting class; with two resting states plus `out_of_view` it needs to
  treat a **set** of indices as uninteresting)

Consequences to accept up front:

- **The deployed ONNX head is a 5-class model and becomes incompatible.** v2 has
  7 states. A retrain is mandatory, not optional.
- The keybindings need reassigning. Seven states no longer fit one intuitive
  letter each; suggest number keys 1-7 for states plus letter keys for toggling
  events, since events are now independent of the state choice.
- `interest_score` and `AGENT_INTEREST_THRESHOLD` need recalibrating against the
  new class set. `training/calibrate_gate.py` already does exactly this and will
  work unchanged once the labels file and head are v2.

## Open questions

Settled 2026-09-25: the optional splits (`sitting` / `sitting alert`, `feeding`
/ `drinking`, `digging`) are all **out**. The class list is the seven states
plus `out_of_view`. Adding any of them later means recoding, so revisit before
the dataset gets large rather than after.

Still open:

1. How many individual rabbits, and are they visually distinguishable on
   camera? This determines whether a focal-animal rule is possible, and it is
   the question a reviewer will ask first.
2. Clip-level predominant-state coding, or full interval coding in BORIS? The
   50% predominant-state rule above is what is in force; BORIS is the upgrade.
3. Who is the second coder for the reliability pass, and when?
