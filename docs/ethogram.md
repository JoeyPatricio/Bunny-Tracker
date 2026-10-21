# Bunny Tracker ethogram (v2)

This is the coding manual for labeling clips. The seven states plus
`out_of_view` are declared in `server_py/shared/labels.py` and
`client/src/labels.js`. All 112 v1 clips were unlabeled on 2026-09-25 for
recoding under these definitions. The v1 labels are archived in
`server/labels.v1.json`.

Freeze this document before labeling starts. If a definition changes after
that, bump `ETHOGRAM_VERSION` in both label files, update the definition text
in `client/src/labels.js`, and recode everything.

**Scope decision (2026-09-25).** Only states are coded. Yawn is folded into
`resting_sitting` and binky into `locomotion_rapid`. Labels are a flat
`{filename: state}` map. This means yawn and binky counts can't be reported.
Adding events later only needs a second pass over the clips, not a recode of
the states. The design is under [Events](#events-not-implemented).

## Sources

Definitions come from the NC3Rs / IAT / RSPCA laboratory rabbit ethogram,
which compiles:

- Morton DB et al. (2003). Refinements in rabbit husbandry. *Laboratory
  Animals* 27: 301-329, Appendix 1.
- Held SDE, Turner RJ, Wootton RJ (2001). The behavioural repertoire of
  non-breeding group-housed female laboratory rabbits. *Animal Welfare* 10(4):
  437-443.
- Gunn D, Morton DB (1995). Inventory of the behaviour of New Zealand White
  rabbits in laboratory cages. *Applied Animal Behaviour Science* 45(3-4):
  277-92.

Quotes below are from the NC3Rs compilation. Places where I collapsed or
dropped a category are marked **[deviation]**.

## Why v1 was replaced

| v1 label | Problem |
|---|---|
| `normal` | Not a behaviour. It meant "none of the others" and soaked up resting, locomotion, feeding and empty frames. It can't be defined, so it can't be coded reliably. |
| `standing` | The published term is rearing. |
| `zoomies` | Mixed a state (running) with an event (the binky). |
| `yawn` | A sub-second event competing against multi-second states in one softmax. |
| `grooming` | Closest to usable. Needed onset and offset rules. |

The main change is deleting `normal` and replacing it with the behaviours it
was hiding.

Adedeji et al. (2023), the closest published work, uses seven vernacular
classes with no operational definitions or reliability measure, overlapping
categories, and states mixed with events. Two of their classes map here.
`periscoping` is `rearing`, and `digging` is a real NC3Rs category I left out
only because there's no diggable substrate in frame.

## Design rules

1. **The camera limits granularity.** At 224x224 from across the room I can't
   see eye aperture or reliably see ear position. Categories that depend on
   them are merged, and each merge is listed below.
2. **`out_of_view` is a coding category, not a behaviour.** It is excluded
   from training but counted, because the share of unusable footage is a real
   result.
3. **States are mutually exclusive.** Exactly one applies at any instant.

## States

Listed in the order they appear in `LABELS`.

### `resting_lying`
Trunk on the ground. Includes limbs tucked under, limbs outstretched, and
contact-lying with a conspecific.

> NC3Rs *Resting*: "Lying, limbs tucked under: resting with trunk on ground,
> hindlimbs tucked under the forelimbs lying under or forward stretched from
> body. Lying, limbs outstretched: resting with body trunk on ground, all four
> limbs outstretched and belly exposed."

**[deviation]** NC3Rs `Sleep` and `Dozing` are merged into the two resting
states because they depend on eye aperture. This study can't report sleep.

- **Onset:** trunk contacts ground and forward motion ceases.
- **Offset:** trunk lifts off the ground, or locomotion begins.
- **Not:** a body-roll in progress (transient, code the surrounding state).

### `resting_sitting`
Upright and stationary, rear end and forepaws on the ground, trunk off the
ground.

> NC3Rs *Resting*: "Sitting: in upright stationary position, with rear end and
> forepaws on ground and ears down. Sitting alert: as above, but with ears erect."

**[deviation]** `Sitting` and `Sitting alert` are merged because ear position
isn't reliable at this resolution.

- **Onset:** forepaws contact ground with trunk raised, motion ceases.
- **Offset:** forepaws leave the ground (becomes `rearing`), trunk lowers
  (becomes `resting_lying`), or locomotion begins.
- **Includes yawning.** Yawns are not coded separately (see the scope decision
  at the top), and a rabbit that yawns is almost always sitting or lying. Code
  the posture and ignore the yawn.

### `rearing`
Sitting up on the hindlimbs with **both** forepaws off the ground. Replaces
v1 `standing`.

> NC3Rs *Rearing*: "Sitting up on hind-limbs with both forepaws off the ground;
> ears partly or fully down." *Rearing alert*: "As above but with ears erect."

**[deviation]** `Rearing` and `Rearing alert` are merged for the same reason.

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

### `locomotion_rapid`
Running: sustained rapid forward movement, typically a circuit of the room.
Replaces v1 `zoomies`.

> NC3Rs *Locomotory*: "Running: rapid forward movement achieved by alternate,
> fully-stretched extension of fore and hindlimbs."

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
food and water stations are separately identifiable in frame.

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

## Events (not implemented)

Nothing here is in the code. Events would be brief (under about 2 s), counted
rather than timed, and coded alongside whatever state is happening. They'd be
scored as detection (precision/recall at a time tolerance) instead of
classification.

- `yawn`: wide mouth opening with the head raised, 1 to 2 s, usually while
  sitting or lying. Not in the NC3Rs compilation. Check Gunn & Morton (1995)
  before using it, or mark it as locally defined.
- `binky`: NC3Rs "frisky hop". One count per airborne twist or kick-out.
- `thumping`: only worth adding if capture records audio, which it doesn't.
- `head_flick`, `scratching`: only if they turn out to be common.

## Annotation protocol

**Unit of coding.** Clips are 8 s (clipping mode) or 12 s (alert mode). Code
the predominant state, meaning the one that covers the most of the clip, and
only if it covers at least 50%. If nothing reaches 50%, leave the clip
unlabeled and flag it uncertain. There is no `mixed` code.

Interval coding in [BORIS](https://www.boris.unito.it/), with onset and offset
per bout, would give frame-level labels. It's more work. Decide before
recoding the 112 clips so they only get recoded once.

**Focal animal.** If more than one rabbit is in frame, code the one nearest
the centre at clip onset. If the rabbits can't be told apart, code at group
level and say so.

**Uncertainty.** The studio stores an `uncertain` flag in
`server/labels.meta.json`. Uncertain labels stay in training. The reliability
report also computes agreement without them. If no state fits, leave the clip
unlabeled. Don't guess.

## Reliability protocol

This measures how consistently I code (intra-observer), not agreement between
two people.

1. Freeze this document.
2. Finish the first pass over the CV pool.
3. The studio draws a fixed 25% uniform sample (seed 0, rounded up). For 112
   clips that's 28.
4. Re-code them blind in reliability mode. First-pass labels, suggestions,
   filenames and metadata are hidden. Re-codes go to
   `server/labels.reliability.json`.
5. Run `training.reliability`. It writes overall and per-class Cohen's kappa,
   raw agreement and uncertainty counts to `server_py/models/reliability.json`.

No kappa has been measured yet. Re-code kappa is not an upper bound on
classifier kappa because the samples and vocabularies differ (the classifier
excludes `out_of_view`).

## Prospective temporal holdout

**Cut date: not set yet.** Set `HOLDOUT_CUTOFF` in `server/.env` to a date
(`YYYY-MM-DD`, midnight UTC) or an ISO-8601 timestamp and restart the server.
Until then every label goes to `labels.json`.

Once set:

- Clips captured before the cut are the CV pool. Labels go to
  `server/labels.json`.
- Clips captured on or after the cut are holdout. The studio badges them
  HOLDOUT and saves their labels to `server/labels.holdout.json`. Nothing that
  trains, tunes or calibrates reads that file.
- Capture time comes from the UTC timestamp in the filename, then the file
  mtime. A clip with neither counts as holdout.
- Setting the cut doesn't move existing labels. Re-labeling a post-cut clip
  moves just that clip.
- Holdout clips don't count toward the reliability sample or the clipping
  backlog cap.
- The holdout file gets opened once, at paper freeze, to measure drift.

Code is in `server_py/app/lib/holdout.py`, tested by
`server_py/tests/test_holdout.py`.

## Migrating the 112 v1 clips

v1 counts: `normal` 45, `zoomies` 26, `grooming` 25, `standing` 11, `yawn` 5.

`grooming` and `standing` mostly map to `grooming` and `rearing`. The others
can't be mapped because the information was never recorded. Don't write a
migration script. Re-watch all 112 clips under v2 (about an hour) so the whole
dataset is coded under one protocol.

`server/labels.json` stays flat:

```json
{
  "recording-clip-2026-09-24T14-03-11.webm": "resting_sitting"
}
```

Coder, time, ethogram version and uncertainty go in `server/labels.meta.json`.
`training.recoding_report` cross-tabs v1 against v2.

## Where the vocabulary lives

`LABELS` order matters because it maps classifier output indices to names.

- `server_py/shared/labels.py` (`LABELS`, `LABEL_PHRASE`)
- `client/src/labels.js` (`LABELS`, `LABEL_COLOR`, `LABEL_PHRASE`, `LABEL_DEFINITION`)
- `server_py/app/lib/valid_labels.py` (server-side validation)
- `server_py/app/routers/notify.py` (alert email wording)
- `client/src/components/LabelingStudio.jsx` and `RecordingGallery.jsx`
- `server_py/agent/capture.py` (`interest_score` treats the resting states as
  uninteresting)

The old five-class ONNX head can't serve v2, so a retrain is required.
`training/calibrate_gate.py` recalibrates `AGENT_INTEREST_THRESHOLD` for the
new classes.

## Open questions

Decided 2026-09-25: no `sitting alert`, `drinking` or `digging` splits. Adding
one later means recoding.

1. How many rabbits are on camera, and can I tell them apart?
2. Stick with predominant-state clip labels, or move to BORIS interval coding?
3. When does the blind re-code happen, and will a second observer code a
   sample?
