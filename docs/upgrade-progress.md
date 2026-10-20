# Upgrade plan progress

Updated 2026-10-19 on `overnight/upgrade-plan`. This run used Opus 5.5 in
herdr pane `w4:p8` and GPT-6.1-Sol in `w4:p9`, coordinated from `w4:p1`.
No commits or remote writes were made. The two edits already present in
`agent/ffmpeg_io.py` and `scripts/bench_pi.py` were preserved.

## Current state

The remaining offline tooling is implemented. The research and deployment
milestones are not complete. This checkout contains zero recordings and no
`server/labels.json`, reliability sample or blind re-codes. Synthetic checks
verify the machinery and do not provide rabbit behavior results.

| Phase | Implementation | Measurement still required |
|---|---|---|
| 0 environment/export | `pyproject.toml`, frozen `uv.lock`, CPU-only train group, generated requirements exports, strict candidate export guards | A seven-class fit on recoded clips |
| 1 checks/CI | Check-script runner, fixed test-only security credentials, locked environments in CI | Remote CI is for Joe to run after publishing |
| 2 labeling | Existing inline definitions, playback controls, uncertainty, class progress and blind sample tooling retained | Primary coding, resolved manual ambiguities, blind 25% re-code and measured kappa |
| 3 evaluation/deployment | Existing nested grouped CV, cache v2, OOF calibration and G1-G7 retained, with selected-arm provenance and group-bootstrap macro-F1 intervals | Real CV, final fitting, export, gate, promote and live inference |
| 4 augmentation | ReplayCompose policy verified, paired policy arms, final full-view training via `--arm` | Paired legacy/albu comparison on real labels |
| 5 ROI | Group-local median background, connected-component ROI, higher-resolution crop extraction, per-class fallback reporting, paired `gru@crop` | Real comparison and a separately validated matching runtime ROI before any crop deployment |
| 6 pose | Offline 40-feature geometry and nested sklearn arm on the reference folds, annotation provenance and OOF/fold validation | Reviewed keypoints or a documented independent pose estimator and actual paired evaluation |
| 7 paper | Measured learning-curve producer, six SVG/PDF figure exports with source hashes, Phase C rewrite and paper instructions | All source reports, paper claim, manuscript, artifact freeze and publishing |

## Changes in this run

The lock covers runtime Python 3.14 and the CPU training group on Python 3.12.
It installs torch only for training. The generated requirements files are
compatibility exports. `docs/pipeline.md` documents their generator commands.
A fresh scratch runtime and training environment were installed from this
lock without replacing the existing environments.

Every CV arm records its cache policy and view. Final training reads the
selected full-view arm's cache and writes that provenance into its result.
Export requires an all-clips fit, unchanged labels and the current ethogram,
refuses the live model directory, and carries the selected arm into the
candidate manifest. G6 checks the selected arm against static logistic
regression and verifies vocabulary, ethogram and provenance. A quick smoke
run cannot supply deployment evidence.

The agent requires promotion metadata and uses its selected window. A
conflicting window override disables inference with a diagnostic. The Training
panel describes the promoted model and its OOF macro-F1, rather than a later
unpromoted training run or a legacy selection score. Runtime parity uses the
promoted policy/window and rejects empty or mismatched caches.

Crop remains an offline arm. Its background uses only each source group's
frames, so backgrounds never cross folds. Low-motion and large-box fallbacks
are recorded and reported. Singleton resting clips can disappear into their
own median background. A per-clip offline ROI is not a validated streaming ROI,
so final training and export refuse crop heads.

Macro-F1 confidence intervals resample complete source groups across the same
OOF repeats. They are conditional on the fitted predictions and exclude
retraining uncertainty. Repeat SD is reported separately. A constant nonzero
paired difference produces no estimable p-value, recorded as null.

The pose arm accepts complete, reviewed single-animal tracks and does not
install or import a pose framework. The learning-curve producer keeps the
outer evaluation pool fixed, samples nested whole training groups and rebuilds
inner folds for every size. Figure export validates recorded reports/OOF data
before writing anything and records source/output hashes and plotted values.

`docs/ethogram.md` now describes the implemented flat label schema and uniform
25% intra-observer re-code. It records remaining coding-manual decisions rather
than implying nonexistent mixed/event/quality fields or independent-observer
measurements. Human re-code kappa is not a mathematical classifier ceiling.

## Verification

- Fresh frozen runtime and CPU training environments installed successfully.
  Runtime has no torch. Training reports `torch==2.14.0+cpu`, with no CUDA.
- The initial 14-script hermetic suite passed in those environments.
- The server robustness script passed 20 consecutive runs.
- The integrated hermetic runner passed 20 scripts, with the real-data decode
  script deselected, in 154.63 seconds. The three scripts covering the final
  deployment, export and evaluation fixes passed again in 19.84 seconds.
- Client production build, Ruff and `git diff --check` passed.
- Synthetic ONNX export, quantization, gate, promote and ring-buffer parity
  exercise the real numeric path with a small stub backbone.
- Real-data decode parity, a real retrain, live camera inference and Pi
  measurements were not run because the required data/hardware are absent.

## Remaining prerequisites

1. Restore the recordings on the machine used for coding. Do not automatically
   map v1 labels. The archive `server/labels.v1.json` remains available.
2. Resolve supported-forepaw `rearing` wording, the unavailable notes/allo
   fields, focal-animal identity and the handling of clips with no predominant
   state. Freeze the manual and update UI definitions if they change.
3. Complete primary coding and the fixed blind 25% re-code. Resolve thin-class
   support before grouped CV. Record measured agreement in the ethogram.
4. Choose and document `HOLDOUT_CUTOFF`. It remains unset. No date or historical
   labels were changed by this run. Labels already in the CV file are not moved
   merely by setting a cutoff.
5. Run the documented real-data pipeline and registered experiments. Do not
   interpret synthetic or quick-run scores as results. Default to legacy/full
   preprocessing until a real comparison supports a change.
6. Measure a Raspberry Pi directly before a Pi latency claim. Pose annotation
   and runtime ROI deployment need their own real-data validation.
7. Freeze reports, regenerate figures, write the paper and publish under Joe's
   name. Publishing remains Joe's responsibility.

## Earlier implementation history

The previous run landed labeling tooling in `1120143`, candidate promotion and
reports in `f83c07e`, cache/decode v2 in `feb4093`, nested CV in `48f8529`,
deployment gates in `99b75f0`, and holdout routing in `dab6376`. Partial Phase 4
work was included in `676f531`. Its old progress entry described it as
uncommitted and unverified. This run completed and verified that work.

The previous progress log's real-data measurements described another data
snapshot: 112 clips, 53 source groups and 15 capture days, with 111 WebM files
used for decode parity. Those measurements were not repeated in this checkout.
