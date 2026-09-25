# Provenance: the fix log and the port plan

## Why this file exists

Roughly 40 code comments and test docstrings cite four documents that are not in
this repository and never were:

| Cited as | Citations | Status |
|---|---|---|
| `notes/fixes.md` | 25 | never committed |
| `notes/environment.md` | 7 | never committed |
| `notes/python-port-plan.md` | 4 | never committed |
| `notes/architecture.md` | 4 | never committed |

`.gitignore` excludes `notes/` as local working notes, and
`docs/stretch-goals.md` reinforces that they hold machine-specific paths. The
result is that every regression test in this project opens by citing a document
no one who clones the repo can read, and a reviewer asking "what is fixes.md
3.1" has no way to find out.

**This file is reconstructed, not recovered.** The originals are not available
here. Everything below is derived from the check names and docstrings in the
test suite, which carry the numbering faithfully because each check is prefixed
with its item number. Where a description is inferred rather than quoted, it
says so.

## fixes.md: the regression log

Each installment is a batch of fixes with a dedicated regression test. The test
is the authoritative record of what the fix was, since it asserts the broken
behaviour is gone.

### Installment 1: security

Test: `server_py/tests/test_security_regressions.py` (runtime venv)

| Item | Defect |
|---|---|
| 1.1 | The SPA fallback served any file on disk. Path traversal via encoded `%2e%2e` segments. |
| 1.2 | `compare_digest` raised `TypeError` on a non-ASCII `str`, turning a wrong password into a 500 instead of a 401. |
| 1.3 | `startswith('backups')` was bypassable via path normalization, exposing model backups. |
| 1.4 | The admin guard's 401 short-circuited past the security-header middleware, so error responses shipped without headers. |
| 1.5 | `request.body()` was read with no size cap and no content-type check on the frame upload endpoint. |

### Installment 2: the ffmpeg capture loop

Test: `server_py/tests/test_capture_loop.py`. Code: `server_py/agent/ffmpeg_io.py`

| Item | Defect |
|---|---|
| 2.1 | The reader stopped draining stdout while `on_frame` ran, so a slow handler stalled the pipe and ffmpeg blocked. |
| 2.2 | The consumer loop re-read `self.proc` each iteration, so a stale process exit nulled the *new* process after a restart. Readers also had to be torn down before `start()` rebound state. |

### Installment 3: server robustness

Test: `server_py/tests/test_server_robustness.py`

| Item | Defect |
|---|---|
| 3.1 | The recordings directory walk ran on the event loop, stalling every other request. Moved to `asyncio.to_thread`; see `app/routers/recordings.py`. |
| 3.2 | A typo in numeric env parsing raised at import and took the whole server down. Now `config._numberish` falls back with a warning. |

### Installment 4: model, auth, ffmpeg

No single dedicated test; the items are cited from the code they fixed.

| Item | Defect | Cited at |
|---|---|---|
| 4.2 | The model metadata route still described the dead tfjs model. | `app/routers/model.py` |
| 4.3 | The camera agent needed its own service credential, separate from the human cookie session, so it can never lock out the admin. | `app/auth.py` |
| 4.5b | No MJPEG pipe parser to port; the agent reads rawvideo instead. | `agent/ffmpeg_io.py` |

Items 4.1 and 4.4 are not referenced anywhere in the code and could not be
reconstructed.

### Installment 5: training guards

Test: `server_py/training/test_train_guards.py` (**training venv**, needs torch)

| Item | Defect |
|---|---|
| 5.1 | Degenerate inputs to `training/train.py` failed with opaque errors: an empty val split surfaced as `ZeroDivisionError` on the loss average after a full run, and a NaN loss every epoch left `best_state` as `None` and raised `TypeError` from `load_state_dict` after all epochs had burned. Both now raise `SystemExit` naming the actual cause. |

### Installment 0: export guards

Not part of the original numbering. Added during the ethogram v2 upgrade
(`docs/upgrade-plan.md` Phase 0) and numbered 0.x to sit before the historical
items without renumbering them.

Test: `server_py/training/test_export_guards.py` (training venv)

| Item | Defect |
|---|---|
| 0.1 | `head.onnx` exported with a static batch axis, so the head could not be evaluated in batches. torch 2.14 defaults to the dynamo exporter, which specializes a batch-1 example input to a literal `1` and rejects a dynamic `window` axis outright. `export_onnx.py` now pins `dynamo=False`. |
| 0.2 | `export_head` must keep `load_state_dict` strict. `strict=False` would "succeed" against a stale checkpoint by leaving the final layer at its random init, exporting a model that predicts noise confidently. |
| 0.3 | int8 calibration frames were drawn without filtering on `LABELS`, so `out_of_view` clips could set the quantized backbone's dynamic range. |
| 0.4 | `training/model.py` hardcoded `NUM_CLASSES = 5`, which survived the ethogram v2 migration to 7 classes. |

## python-port-plan.md

Cited by `training/export_onnx.py` (§3.4), `training/model.py` (§3.1),
`inference/predictor.py` (§3.1), `scripts/legacy/js_baseline_v1.mjs` (§3.5) and
`scripts/dump_agent_frames.py`.

The document covered the Node-to-Python rewrite (commit `3e46234`, "Rewrite
backend in Python, move inference to ONNX Runtime"). The sections still
referenced:

- **§3.1 Two-stage architecture.** Frozen backbone as a per-frame embedder plus
  a small learned temporal head, exported as two ONNX graphs rather than one,
  so the agent embeds each frame once and runs the cheap head every step.
- **§3.3 Dataset handling.** Group-aware splitting ported from the browser
  trainer's `groupKey`/`splitByGroup`, keeping "the good parts" and fixing its
  known problems.
- **§3.4 Export.** ONNX export plus int8 quantization of the backbone only.
- **§3.5 Accuracy gate.** The retrained model had to meet or beat the tfjs
  baseline. **This gate is retired**; see `docs/upgrade-plan.md`.

The rest of the document is not recoverable from the code.

## architecture.md

Cited by `docs/stretch-goals.md` for "debt item 1": the deployed ONNX classifier
was less accurate than the tfjs model it replaced, 61.5% against 74.4%.

**Both numbers are now known to be unusable.** They were measured on a
validation split that also drove early stopping and checkpoint selection, so
they are selection scores rather than held-out estimates, and they describe a
five-class vocabulary that no longer exists. They are preserved in
`server_py/models/archive/v1-ethogram/` for provenance and must not be cited as
accuracy.

## environment.md

Cited by `docs/stretch-goals.md`, `scripts/bench_pi.py`, and
`training/test_train_guards.py` for the two-venv rule.

Its recoverable content is now in **`docs/pipeline.md`**, which documents both
venvs, their Python versions, what may and may not be installed in each, and
why the runtime venv stays torch-free.

## What to do with the citations

Leave the item numbers in place. They are precise, they map to real tests, and
renumbering would break the correspondence with the check names printed at
runtime. Repoint the *path* only: `notes/fixes.md 3.1` becomes
`docs/provenance.md, fixes.md 3.1`.
