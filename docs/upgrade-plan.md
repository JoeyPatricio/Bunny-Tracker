# Bunny Tracker: project upgrade plan

## Context

The goal is a project good enough to publish from. The ethogram v2 migration has
landed and all 112 clips are unlabeled awaiting recoding. That change exposed how
much of the surrounding machinery was built for a vocabulary that no longer
exists, and an audit turned up problems that block a credible paper regardless of
vocabulary.

Three findings drive everything below.

**There is no test set.** `split_by_group` in `server_py/training/dataset.py`
produces only train/val, and that val set simultaneously drives early stopping,
checkpoint selection, the window sweep, the fp32-vs-int8 comparison, and the
headline number. Every accuracy this project has reported is a selection score.
The current val split is 15 clips over 7 classes, roughly 2 per class.
(`docs/stretch-goals.md` still cites 39, from the older label set.)

**The accuracy gate has no teeth.** `models/gate_result.json` currently reads
`"gate_passed": false` and the model shipped anyway, because `export_onnx.py`
writes into `models/` in place and there is no promote step.

**Two live leakage bugs.** `train.py:181-183` computes the `emb_mean`/`emb_std`
standardization buffers, and `class_weights` at line 176, across all training
entries; under any fold scheme these must be per-fold or the buffers baked into
the exported head have seen the evaluation data. Worse,
`training/calibrate_gate.py:54` selects `AGENT_INTEREST_THRESHOLD` by running the
deployed model over every labeled clip including its own training data. That
threshold decides when you get emailed, and it is fitted in-sample today.

**Outcome:** numbers that are trustworthy, results that reproduce from a clean
clone, and enough of a methods story to survive review.

## Decisions already taken

- **Scope:** everything, staged. Each phase independently useful, stoppable at any
  boundary.
- **License:** stays MIT. Ultralytics YOLO (AGPL-3.0) ruled out.
- **`docs/stretch-goals.md`:** Phases A (Pi) and B (sensors) stand unchanged.
  **Phase C is superseded.** Its negative-result claim rests on selection scores
  over a vocabulary that no longer exists and cannot be claimed as written.
- **`notes/`:** the four missing documents get reconstructed into `docs/` and the
  30+ dangling references repointed.
- **Relabeling:** after the tooling lands, before any modeling.

---

## Phase 0: Unbreak the pipeline and pin the environment

**1 day.** Blocks everything: export currently raises a size mismatch and every
loader dies on the v1 manifest.

| Change | File |
|---|---|
| `num_classes=5` to `len(LABELS)` | `training/diagnose_meanstd.py:42` |
| `n=8` to `manifest["window"]` | `training/evaluate.py:54` |
| Filter calibration clips against `LABELS`, keeping `out_of_view` out of int8 calibration | `training/export_onnx.py:62-66` |
| `h_n.squeeze(0)` to `h_n[-1]`, so the exporter cannot collapse the batch axis | `training/model.py:70` |
| Re-export, assert `logits` is `['batch', 7]`. Fall back to `dynamo=True` plus `onnxscript` if the legacy exporter still folds it | `training/export_onnx.py:52-57` |
| Keep strict `load_state_dict` and let it fail loudly until the 7-class retrain lands. Document why | `training/export_onnx.py:49` |
| Move to `scripts/legacy/js_baseline_v1.mjs`, fix the hardcoded Windows ffmpeg path, make the label list a `--labels` arg, output to `models/js_baseline.v1.json` | `scripts/js_baseline.mjs:27,29` |

**Delete and regenerate**, do not migrate: `models/cache/`, `head_state.pt`
(5-class `fc: 5x32`), `train_result.json`, `gate_result.json`.

**Environment.** `pyproject.toml` with `[project]`, `[dependency-groups] train`,
`[tool.ruff]`, `[tool.pytest.ini_options] testpaths=["server_py/ci"]`. Commit
`uv.lock`. Declare the cu130 torch index explicitly under `[tool.uv.sources]`
plus a `cpu` extra for CI, because bare `torch` currently pulls a multi-GB CUDA
wheel and ~15 `nvidia_*` packages. `.venv-train` was built by `uv` and has no
`pip`, so the README's setup command cannot work: standardise on `uv` and make
the README true. `onnxscript` is declared but not installed.

**New docs.** `docs/pipeline.md` (the 8-step command order, which venv each step
needs, what each writes: this exists nowhere today). `docs/provenance.md` mapping
`notes/fixes.md installment N` to description and landing commit, then bulk
repoint the references. The numbering is fully recoverable from the test check
names: 1.x security, 2.x capture loop, 3.x robustness, 4.x model/auth/ffmpeg,
5.1 training guards.

**Risks.** `h_n[-1]` may not fix the static axis if the exporter folds elsewhere;
the Phase 3 G2 batch check makes that verifiable and `dynamo=True` is the
fallback. The uv/torch index change could break `.venv-train`; build a fresh venv
from the lock in a scratch dir and diff before touching the real one.

**Verify.** New `tests/test_export_guards.py` in the existing check-script style:
a mismatched head width raises; `head.onnx` output is `['batch', len(LABELS)]`;
`pick_calibration_frames()` returns no `out_of_view` clips. Plus
`uv sync --frozen` resolves clean.

---

## Phase 1: Test hygiene, runner, CI

**1 day.** Gives a single green/red signal before the training pipeline gets
rewritten.

**Extract the duplicated helper.** `check(...)` is copy-pasted verbatim into five
files. New `server_py/tests/checks.py` exporting `check`, `skip` (a third outcome
that does not append to `failures`), and `finish`. The scripts already do
`sys.path.insert(0, parent.parent)`, so the import resolves.

**pytest as a runner, not a rewrite.** Single file `server_py/ci/test_scripts.py`
parameterized over the seven scripts, each running `subprocess.run([venv_python,
script])` and asserting exit 0, with stdout attached on failure.
`testpaths = ["server_py/ci"]` so pytest never walks `server_py/tests/`. This
matters: those scripts define `async def test_*` at module level, and pytest
collecting them without an async plugin would emit skips that look like passes,
which is worse than today. Tag `test_server_robustness.py` as `local_data` and
the rest `hermetic`. The scripts stay independently runnable; that is the point.

**Fix the flaky timing test** at `tests/test_server_robustness.py:71-104`
properly rather than by loosening the guard:

- The real assertion is structural and cannot be flaky: patch `_scan_recordings`
  to record `threading.get_ident()`, call the real handler, assert no call ran on
  the event-loop thread. That is what the test was always trying to say.
- Auto-calibrate the batch instead of hardcoding `SCAN_BATCH = 60`: warm the
  dentry cache, take a median of 5 timed calls, compute the batch needed to
  exceed the floor, cap it.
- Measure the timer floor on this machine instead of asserting a Windows 15.6ms
  constant. Linux will come out nearer 6-8ms.
- `skip`, not `fail`, when even the capped batch cannot stall the loop. A
  measurement that cannot be made is not a regression.
- Drop the 20-real-clips precondition: seed a temp dir with synthetic zero-byte
  `.webm` files and repoint `rec.RECORDINGS_DIR`, the pattern already used at
  lines 120-130. scandir cost is per-entry not per-byte, so the measurement is
  identical and the script becomes CI-eligible.

**CI.** GitHub Actions, two jobs: `hermetic` (`uv sync`, `pytest -m "not
local_data"`, CPU torch index so a GPU-less runner does not pull 3GB of wheels)
and `lint` (`ruff check`). Land `ruff check` and `ruff format` as separate
commits, format alone and mechanical, or the diff is review-hostile.

A seeded trainer on an unpinned dependency set is not reproducible. `uv.lock`
plus a CI run proving it resolves is what makes the seeding from the last session
actually mean something.

---

## Phase 2: Labeling tooling, then the recoding pass

**1.5 days tooling, ~3 hours coding.** Front-loaded. Blocks everything after it.

All in `client/src/components/LabelingStudio.jsx` and `client/src/labels.js`:

- **Inline operational definitions.** Extend `LABEL_META` with `definition`,
  `includes`, `excludes`, `onset`, `offset`, lifted verbatim from
  `docs/ethogram.md`, rendered beside the label buttons. `labels.js` is already
  the single source of UI truth and generates every chip, button and badge, so
  this needs no new plumbing. **Highest-value change here:** a coder with the
  criterion in front of them for all 112 clips is measurably more consistent than
  one recalling it, and it is what makes the `resting_lying`/`resting_sitting`
  and `locomotion`/`locomotion_rapid` boundaries survive an afternoon.
- **Playback control.** 0.25x/0.5x/1x, loop, frame-step. `locomotion` versus
  `locomotion_rapid` is a speed judgement and is not reliably codeable at 1x.
- **Uncertainty flag**, a toggle not a label, written to a sidecar. Ambiguous
  clips stay findable, and the paper gets an honest "n flagged uncertain" line.
- **Per-class progress** replacing the single bar, so a thin class surfaces
  *during* coding when the remedy is still cheap.
- **Reliability mode**: a blind re-code serving a seeded random 25% with the
  existing label hidden, writing `server/labels.reliability.json`. **This must
  exist before the recoding pass.** Intra-observer agreement cannot be
  retrofitted once you have seen your own answers.

**Server side.** Add a `server/labels.meta.json` sidecar
(`{filename: {ethogram_version, coder, coded_at, uncertain}}`) with its own store
mirroring `label_store.py`'s `create_store(..., guard_wipe=True)`. Keep
`server/labels.json` as the flat `{filename: label}` map so eight consumers stay
untouched; `docs/ethogram.md` already committed to the flat schema as an explicit
scope decision. Gitignore the new sidecars.

**Then recode all 112 clips**, then the 25% blind re-code, then
`training/reliability.py` computing `cohen_kappa_score` overall and per class into
`models/reliability.json`. That number is the ceiling on what any classifier can
achieve and belongs in the methods section.

**Risk.** Recoding may leave a class under 5 clips; `rearing` and `feeding` are
the likely candidates (v1's `standing` had 11, `yawn` had 5). The per-class
display surfaces it while a targeted Clipping Mode harvest is still cheap. Decide
merge-versus-collect before Phase 3, and bump the ethogram version if you merge.

---

## Phase 3: Cache v2, fold harness, gate, promote

**2-3 days. The core phase.**

### Evaluation: repeated nested grouped CV, not a three-way split

**Verified, not assumed:** `group_key` buckets the 112 clips into **53 groups
across 15 capture days**, largest 7 clips, 29 singletons.
`StratifiedGroupKFold(n_splits=5)` on the real structure yields five clean folds
of 21-25 clips with every class present in every fold.

A 20% test set would be ~22 clips, ~3 per class, with a Wilson 95% interval near
±20 points. It could not distinguish a real 10-point gain from noise, which is
the one job a test set exists to do, and it would cost 20% of the training data
at exactly the sample size where the learning curve is steepest. Grouped k-fold
instead gives every clip an out-of-fold prediction, so the pooled OOF confusion
matrix has support of the full 112.

**Nesting.** Within each outer fold, split the outer-train groups again and stop
on inner-val loss. Window sweep, learning rate and augmentation policy are
selected on inner folds per outer fold; if outer folds disagree on window, that
instability is itself a result. fp32-vs-int8 leaves the science path entirely and
becomes a deployment check. Headline is pooled OOF macro-F1, balanced accuracy
and kappa, plus mean and sd across repeats. Cost is trivial: the head trains on
cached embeddings in seconds, so 5 repeats x 5 outer x 4 inner x 3 windows is
minutes.

Add a hard precondition, because this will bite after recoding: if
`min(class_counts) < n_splits` or a class's clips all sit in one group, fail with
an actionable message naming the class and the remedy, never a silent zero-support
fold.

**Instead of a test set, designate a prospective temporal holdout.** Freeze the
112 as the CV pool; every clip captured after a cut date goes to
`server/labels.holdout.json` and is opened once, at paper-freeze. This costs zero
clips today and tests distribution shift over time, which is the actual
deployment failure mode for a fixed camera in a room that changes. Record the cut
date in `docs/ethogram.md`.

### Cache v2: split-agnostic, two tiers

The current cache bakes the split in at extraction time because augmented
variants are generated before embedding. But the backbone is frozen, so
embeddings do not depend on the split at all; only the choice of which variants to
generate does.

```
models/cache/
  frames/<clip>.npy      # raw uint8 [TRAIN_FRAMES,224,224,3], decode once
  emb/<clip>.v<k>.npy    # [TRAIN_FRAMES,1280], variant k (k=0 identity)
  manifest.json          # schema:2, clips:[{filename,label,group,variants}],
                         # window, train_frames, backbone, labels,
                         # labels_sha256, ethogram_version, aug_policy
```

No `train`/`val` keys. Generate all variants for all clips (112 x 3 x 16 = 5,376
backbone forwards, seconds on the 3060). At fold time: train clips get all
variants and all window starts; eval clips get variant 0 and window `[0, W]`
only, reproducing today's eval protocol exactly so numbers stay comparable. The
frames tier means changing augmentation policy or backbone re-runs only the embed
pass, not 112 ffmpeg decodes.

**Invariant to assert in a test:** an augmented variant of a clip must never
appear in a fold where that clip is being evaluated.

### Decode: one ffmpeg process per clip, not OpenCV

`extract_frames` spawns 17 processes per clip (one duration probe, 16 frame
reads). Replace with a single ffmpeg invocation using a `select` filter on the
target timestamps, piping all N frames as one rawvideo stream. Same binary, same
`scale` filter, same `pix_fmt`, so byte-parity is preserved by construction,
which matters because `tests/test_motion_parity.py` and the JS-parity story
depend on it. Keep the old function as `read_frame_single` and add a parity
script asserting byte-identical arrays across all 112 clips.

### The rest of Phase 3

- **`training/splits.py`**: `group_key` moved here as the single definition,
  `make_folds()` on `StratifiedGroupKFold`, the min-support precondition.
  `split_by_group` stays in `dataset.py` as legacy and imports `group_key`.
- **`training/metrics.py`**: macro-F1, balanced accuracy, kappa, classification
  report, pooled OOF confusion matrix, Wilson intervals on per-class recall.
  `check_degenerate()` moves here out of `calibrate_gate.py` so it runs in both
  places. Replaces `per_class_report`, currently duplicated verbatim in
  `train.py:101-110` and `evaluate.py:33-44`.
- **`training/crossval.py`**: the nested CV, writing `models/cv_result.json` and
  appending to `models/runs/index.jsonl` (run id, git SHA, seed, labels hash,
  ethogram version, arm, metrics). **Fixes the per-fold `emb_mean`/`emb_std`/
  `class_weights` leakage.**
- **`calibrate_gate.py`**: threshold selected on OOF probabilities, fixing the
  in-sample bug.
- **`train.py`**: keeps its CLI, gains `--fold`/`--all-clips`. The final
  deployable model trains on all labeled clips with CV-selected hyperparameters.
  Its own reported number is demoted to a pointer at `cv_result.json` so nobody
  can quote a selection score again.
- **`verify_predictor_parity.py`**: update for schema 2; it reads
  `manifest["val"]`, which disappears.

### What replaces the dead accuracy gate

`js_baseline.mjs` hardcodes v1 labels and the tfjs model physically cannot
predict 7 classes. Split the one gate into two artifacts with different jobs.

**Science, in `crossval.py`:** evaluate arms on identical folds so they are
paired.

| Arm | Tests |
|---|---|
| `majority` | chance floor, 1/7 balanced accuracy by construction |
| `static_logreg` | mean/std pooled embeddings into sklearn LogisticRegression |
| `static_meanstd_mlp` | the `diagnose_meanstd.py` architecture, folded in as an arm |
| `gru` | the shipping model |

**Primary comparator is `static_logreg`.** It uses the same frozen embeddings, so
the only variable is temporal aggregation, which is exactly the claim the project
makes. It retrains per fold so it cannot go stale or out of sync with the
vocabulary, and it is paired on identical folds so you can run a paired test
rather than comparing two point estimates. The old js_baseline had none of those
properties even when it worked. Lead with kappa in the paper, because it is
directly comparable to the human-human kappa from Phase 2, giving a measured
ceiling.

**Deployment, in a rewritten `evaluate.py`:** G1 head width equals `len(LABELS)`
(move `agent/capture.py:430`'s guard here so it fires at build time, not at 3am);
G2 batch axis actually batches; G3 ONNX fp32 matches the torch head within 1e-4
on cached embeddings (stronger than today's check, which compares only final
accuracies and so passes when two different models happen to score the same); G4
int8 macro-F1 within 0.03 of fp32, catching the per-tensor depthwise collapse the
codebase already documents at `export_onnx.py:96-99`; G5 non-degenerate; G6
candidate beats `static_logreg` on the same folds; G7 candidate is not worse than
the deployed snapshot by more than one fold-sd.

**Give the gate teeth.** `export_onnx.py` writes to `models/candidate/`. New
`training/promote.py` refuses unless `gate_passed` is true and artifact SHA-256s
match, archives the current `models/*.onnx` to `models/archive/<iso8601>/`, then
moves candidate into place. `--force "<reason>"` writes the reason into the
promoted provenance so an override stays visible.

**Free paper figure:** `training/recoding_report.py` cross-tabulating
`server/labels.v1.json["labels"]` against the new `server/labels.json`. A 7x5
table showing exactly what `normal` was hiding, which is the strongest empirical
support for the migration, at zero compute. Both files already exist. Note
`labels.v1.json` is nested, so readers must go through `["labels"]`.

**Verify.** `training/test_crossval_guards.py`: every clip in exactly one outer
eval fold and no training set for that fold; no group on both sides of any fold;
a synthetic 3-clip class raises the actionable error; `emb_mean` for fold k is
bit-identical to `emb_mean` from fold k's train clips alone; the majority arm
yields exactly 1/7; `promote.py` refuses on a false gate and on a hash mismatch.

**Stopping here already gives you** a defensible headline number, a gate that can
block a deploy, a threshold calibrated without leakage, and a kappa against a
measured human ceiling. That is most of a paper's methods section.

---

## Phase 4: Albumentations

**1 day.** Replace `flip_frames`/`jitter_frames`/`make_variants` in
`dataset.py:97-119` with `ReplayCompose`, which applies one sampled transform
identically across every frame of a clip: exactly the constraint `make_variants`
hand-implements. Adds rotation, perspective, motion blur and `CoarseDropout` for
occlusion, which the hand-rolled version cannot express. Raise the variant bank
from 3 to ~8 now that generation is cheap.

**Risk.** Aggressive augmentation destroys posture signal: a rotated rabbit is not
a valid `resting_lying`, and motion blur is adversarial to
`locomotion`/`locomotion_rapid`. Ship as a paired arm on identical folds and keep
whichever wins. Record `aug_policy` in the manifest so results stay attributable.

---

## Phase 5: Crop / ROI arm

**1-2 days.** Cheap, license-clean, plausibly the largest single accuracy lever at
n=112, which is why it comes before pose.

Median background from the cached frames, threshold the difference reusing
`shared/motion.py`'s pooled-luma detector, take the connected component's bounding
box unioned across the clip with a margin, crop, feed the same pipeline at the
same 224². Removing background nuisance variation and raising effective
resolution on the animal costs **zero runtime dependencies and zero inference
cost** if the ROI is static. Run as arm `gru_crop` against `gru_full` so the claim
is measured. `torchvision` detection (BSD) offline only if a static ROI
underperforms.

**Risk.** The rabbit leaves the ROI or the camera is bumped. Log the fraction of
frames where the motion blob falls outside, fall back to full-frame past a
threshold, recompute the background on a schedule.

---

## Phase 6: Pose arm

**Days to weeks**, dominated by keypoint annotation. Stoppable after the offline
step.

SLEAP for keypoints (**verify the license before committing**; believed BSD-3,
DeepLabCut is LGPL-3.0), then your own `training/pose_features.py` producing ~40
geometric features (inter-keypoint distances, body-axis angle, ear-tip-to-spine
angle, per-keypoint velocity and acceleration, bbox area rate, all normalised by
body length), then an sklearn classifier as an arm on the same folds.

The sample-efficiency argument is strong: ~40 geometric features versus 1280-d
embeddings at n=112. Expect pose to win on `rearing` and the
`resting_lying`/`resting_sitting` boundary, because those are pure posture
distinctions a mean-pooled ImageNet embedding has no reason to encode, and they
are exactly the NC3Rs definitions that specify posture and ear position.

**Pose augments, it does not replace.** Do not add pose to the runtime venv,
`inference/`, or `agent/capture.py` in this plan. The runtime venv staying
torch-free and ONNX-only is one of the better decisions in the codebase. A
SLEAP-family backbone is heavier than the 2.9MB int8 model, and
`AGENT_FRAME_SECONDS=1500ms` on a Pi 5 is the whole budget. If pose wins
decisively, the realistic shape is two-tier: the cheap embedding path triggers
every frame, pose runs only on frames that already cleared the interest
threshold. Gate that on a measured `scripts/bench_pi.py` result.

---

## Phase 7: Paper artifacts

**1-2 days.** `training/figures.py` emitting to `paper/figures/`: learning curve,
head ablation, quantization table, OOF confusion matrix, the v1-to-v2 recoding
cross-tab, kappa against human kappa. Every number traceable to a JSON under
`models/`, which is `docs/stretch-goals.md` §15's own verification criterion.

Rewrite that file's Phase C. Keep A and B. The new claim is a methods
contribution: an open-source, edge-deployable, continuous home monitoring system
for a genuinely understudied species, with a published-ethogram vocabulary,
inter-observer reliability, grouped cross-validation, and paired comparison
against multiple baselines. That is defensible whichever way the accuracy lands,
which the negative-result framing was not.

---

## What to cut

**OpenCV for frame decode.** The measured cost is 59s across 112 clips, once, and
after Phase 3's frames tier, once ever. Against that, cv2 seek on VP8/VP9 `.webm`
is least trustworthy exactly where this project needs it, in a codebase whose
parity story depends on sampling identical timestamps. The single-ffmpeg-process
fix is strictly better and adds no dependency. (`opencv-python-headless` arrives
anyway as an albumentations transitive dep in the training venv; that is fine,
just keep it out of `requirements.txt`.) The MOG2 comparison against
`shared/motion.py` is a nice paper figure, not a product need, and must never
replace the parity-tested detector.

**BORIS.** Right tool, wrong stage. `docs/ethogram.md` made an explicit scope
decision for clip-level flat coding, implemented across eight consumers. BORIS
produces interval data none of them can read, so adopting it means changing the
schema, API, UI, loaders and writing a converter, to annotate for a model that
classifies fixed 8-frame windows. Keep it where the ethogram already puts it: the
upgrade path.

**DVC.** The real requirement is "which labels produced which model", satisfied by
~15 lines: a SHA-256 of `labels.json` plus ethogram version and `LABELS` written
into the manifest and every result JSON. DVC adds a remote, a content cache, a
lockfile and a second mental model of what is checked out, for 112 clips and four
JSON files on one machine.

**Weights & Biases**, and **MLflow for now.** W&B is a cloud service with an
account and telemetry in a project whose stated value is self-hosted and offline.
MLflow is a server and two stores for what will be a few dozen runs. Make
`models/runs/index.jsonl` a mandatory Phase 3 deliverable instead: 20 lines,
greppable, diffable, and exactly the schema MLflow would ingest later. Revisit in
local file-store mode if run count passes a few hundred.

**SimBA and B-SOiD.** Both believed GPL-3.0 (verify), both opinionated full
pipelines wanting their own layout and config, both fighting a torch 2.14+cu130
environment. Their actual contribution, geometric features into a tree ensemble,
is ~150 lines of numpy plus `sklearn.ensemble` that you can own and ship under
MIT. Take the idea, not the dependency.

**DeepEthogram as a dependency.** Biggest effort-to-value mismatch on the list: a
full pipeline with its own GUI, data format, hydra config and pinned older
torch/CUDA that will fight `.venv-train`, wanting frame-level labels across
thousands of frames, which is a second annotation project on top of Phase 2. The
output is one number reported once, and a real chance of ending in "couldn't
reproduce". **Keep it as a citation**, and if the temporal-signal question turns
out to matter, reimplement the idea (optic-flow features plus a sequence model) as
an arm in `crossval.py` for a day instead of a week.

**Converting the test suite to pytest.** Large diff, no new coverage, and a real
risk of async-collection false passes. The runner and the shared helper get the
benefit.

**Keep with conviction:** scikit-learn (already installed, entirely unused, best
value-per-hour on the list), albumentations behind a paired ablation, SLEAP as an
offline research arm, pytest as a runner only.

---

## Verification

End to end, this is done when:

1. `pytest -m "not local_data"` exits 0 from a clean clone against `uv.lock`, and
   the robustness test passes 20 consecutive runs.
2. A full retrain runs: `extract_frames -> crossval -> train -> export_onnx ->
   evaluate -> promote -> verify_predictor_parity -> calibrate_gate`, producing a
   7-class head that `agent/capture.py`'s guard accepts.
3. `./run.sh all` starts server and agent, the agent loads the model instead of
   falling back to motion-only, and the dashboard shows live v2 predictions.
4. The headline is pooled OOF macro-F1 with a confidence interval, from data never
   touched during selection, alongside `majority` and `static_logreg` on identical
   folds.
5. Cohen's kappa for the blind re-code is recorded in `docs/ethogram.md`.
6. `promote.py` demonstrably refuses a failing candidate.
7. Every number in the paper draft traces to a file under `server_py/models/`.

## Stop points

Phases 0-1 make the project buildable and testable. Phase 2 produces the dataset.
Phase 3 is what makes the numbers publishable. Phases 4-7 improve a project that
is already sound. **Stopping after Phase 3 leaves a coherent, honest, reproducible
system with a trustworthy evaluation and a real dataset**, which is further than
most projects of this kind get.

## First action on execution

Copy this plan to `~/bunny-tracker-upgrade-plan.md` as requested, and commit a
copy to `docs/upgrade-plan.md` so it travels with the repo.
