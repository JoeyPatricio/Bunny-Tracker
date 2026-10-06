# Upgrade plan: progress log

Tracks execution of [upgrade-plan.md](upgrade-plan.md). Overnight run started
2026-10-05 on branch `overnight/upgrade-plan` (never pushed, never on `main`).
Each verified work package is one commit on that branch.

## Decisions taken for this run

- Commits are allowed on `overnight/upgrade-plan` for this run only. Review,
  squash or drop in the morning.
- Temporal holdout cut date: **placeholder**. The mechanism is built with a
  TODO date; clips captured before a date is chosen are in an ambiguous state.
- Scope: Phases 2 and 3 first, then Phases 4 and 5 as code-only arms, marked
  unevaluated until the recoding pass is done.
- Reliability mode: default coder `joe` (editable), blind re-code sample is 25%
  drawn once with seed 0 and stored.
- Phase 1 departures: requirements files instead of `uv sync`/`uv.lock` (Phase 0
  decision in `pyproject.toml`), no `ruff format`, robustness test is hermetic.

## Status

| Package | State | Commit |
|---|---|---|
| Phase 0 | done before this run | c339783 |
| Phase 1 lint fixes | done, verified | 39c6542 |
| Phase 1 runner, CI, robustness | done, verified | 9f34d02 |
| Phase 2 tooling | done, verified (9 passed, ruff clean, client builds); UI not checked in a browser | 1120143 |
| Phase 3 splits + metrics | done, verified (8 passed, ruff clean) | dc39c5f |
| Phase 3 candidate export, promote.py, recoding + reliability reports | done, verified (10 passed, ruff clean) | f83c07e |
| Phase 3 cache v2 + single-ffmpeg decode | done, verified (12 passed incl. local_data, ruff clean) | feb4093 |
| Phase 3 crossval, train --fold/--all-clips, OOF gate calibration | done, verified (12 passed + 1 local_data deselected, ruff clean) | 48f8529 |
| Phase 3 evaluate.py deployment gates G1-G7 | done, verified (13 passed + 1 local_data deselected, ruff clean) | 99b75f0 |
| Phase 3 temporal holdout mechanism (placeholder date) | done, verified (14 passed + 1 deselected, ruff clean, client builds) | dab6376 |
| Phase 4 albumentations (code only, unevaluated) | STOPPED at usage limit, partial and unverified, left uncommitted in the working tree | |
| Phase 5 crop/ROI arm | not started | |

## Measured on the real data

From `server/labels.v1.json` (read-only): 112 clips, 53 groups, 15 days,
largest group 7, 29 singletons. v1 per class (clips/groups): grooming 25/20,
normal 45/26, standing 11/9, yawn 5/5, zoomies 26/11. Five outer folds of
23/22/25/21/21 clips. A v2 class with fewer than 5 groups will trip the
`splits.py` precondition by design.

Decode parity: the single-ffmpeg decode is byte-identical to the old
per-frame decode on all 111 `.webm` files in `server/recordings/`, at 8 and 16
frames. 88.4s down to 11.6s.

## Departures from the plan

- Cache v2 eval entries come from a separate per-window eval sample
  (`emb/<clip>.eval<W>.npy`), not the first W of the 16 training frames. The
  plan's "variant 0, window [0, W]" would not have reproduced the schema-1
  eval protocol, which spreads W frames over the whole clip.
- `promote.py --force` bypasses the gate only. The artifact hash check can
  never be forced.
- `verify_predictor_parity.py` now reads `quant_result.json`; it was reading
  `gate_result.json`, which `evaluate.py` stopped writing in Phase 0.

Crossval cost on synthetic data: `--quick` 28s; full default (112 clips,
5 repeats x 5 outer x 4 inner, `--jobs 6`) 37.5 minutes worst case, because
perfectly separable synthetic data never early-stops. The plan's "minutes"
estimate was optimistic; real data has not been timed. Paired test is the
Nadeau-Bengio corrected resampled t-test (plain Wilcoxon reported as an
uncorrected secondary).

Gate choices: G4 compares int8 vs fp32 macro-F1 over every cached clip's
eval frames (in-sample for the `--all-clips` model; only the difference is
gated). G6 requires gru > static_logreg on pooled OOF macro-F1 and reports
the corrected paired p without requiring significance, because at n=112 the
corrected test would block real gains. G7 passes with "no deployed baseline"
on the first promote.

## Where the run stopped

Usage ran out during Phase 4. Its partial work is uncommitted on the branch:
`requirements-train*.txt`, `training/augment.py`, `training/test_augment.py`,
and edits to `training/cache.py`, `crossval.py`, `extract_frames.py`,
`docs/pipeline.md`. It was never verified, and albumentations may or may not
have been installed into `.venv-train`. Either review and finish it, or
discard it with `git restore` / `git clean` on those paths.

## Blocked on Joe

- Recode all 112 clips, then the blind 25% re-code (Phase 2).
- Merge-versus-collect for any class under 5 clips (before Phase 3 results).
- Temporal holdout cut date.

## Phase 2 notes for Joe

- Reliability mode refuses to draw its sample (409) while any clip is
  unlabeled, and never redraws. Sample size is ceil(25%), 28 of 112.
- Keys in the studio: `,` and `.` step a frame, space plays or pauses, `u`
  toggles uncertain. Classes under 5 clips show a warning in per-class progress.
- Daily backups now include `labels.meta.json`, `labels.reliability.json` and
  `labels.reliability_sample.json`.
- The UI has only been build-checked. Open the studio and try it before coding
  all 112 clips.

## Noticed, not changed

- `docs/ethogram.md` references a `mixed` code that does not exist, specifies a
  second observer on a stratified 20% (the plan uses a blind 25% self re-code),
  has a self-contradictory `rearing` "Not:" line, and its coprophagy exclusion
  names no code to use instead. Definitions were copied into the UI verbatim.

- Committed `models/head.onnx` is the stale 5-class head with static batch 1.
  `agent/test_capture_offline.py` runs it through `AgentDecisionLoop`, which
  has no width guard, and prints v2 names for 5-class indices.
- `scripts/bench_pi.py` prints "docs/pipeline.md docs/pipeline.md,".
