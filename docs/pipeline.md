# The training pipeline, end to end

The full command order existed nowhere before this document. It was distributed
across per-file `Usage:` docstrings, several of which disagreed with each other
and all of which used Windows paths on what is now a Linux box. `README.md`
pointed at `training/train.py`'s docstring "for the full pipeline"; that
docstring is two lines and documents only its own invocation.

## The two-venv rule

Two virtual environments, on two different Python versions, and the separation
is deliberate.

| | `server_py/.venv` | `server_py/.venv-train` |
|---|---|---|
| Python | 3.14 | 3.12 |
| Contains torch | **no, ever** | yes, with CUDA |
| Runs | server, agent, `tests/`, `calibrate_gate.py` | everything in `training/` |
| Deployed to the Pi | yes | no |
| Requirements | `requirements.txt` | `requirements-train.txt` |

The runtime venv stays torch-free because it is what ships to the Raspberry Pi,
where inference is ONNX Runtime on CPU. Putting torch in it would add gigabytes
to a deployment that cannot use them. The Python versions differ because torch's
wheel coverage lags the interpreter the runtime targets.

Both venvs were created with `uv`, so **neither has `pip`**. Use:

```bash
uv pip install --python server_py/.venv/bin/python       -r server_py/requirements.txt
uv pip install --python server_py/.venv-train/bin/python -r server_py/requirements-train.txt
```

## The order

Run from `server_py/`. Steps 1 through 6 use the training venv; step 7 uses the
runtime venv, deliberately, because it exercises the same stack the agent does
including int8 quantization.

```bash
TRAIN=.venv-train/bin/python
RUN=.venv/bin/python
```

**1. Extract and embed.** Reads `server/labels.json` and `server/recordings/`,
writes the schema-2 cache under `models/cache/`:

| Path | Contents |
|---|---|
| `frames/<clip>.npy` | raw uint8 `[16,224,224,3]`, the training sample |
| `frames/<clip>.eval<W>.npy` | raw uint8 `[W,224,224,3]`, the eval sample |
| `emb/<clip>.v<k>.npy` | `[16,1280]` embeddings of variant k (k=0 is the original) |
| `emb/<clip>.eval<W>.npy` | `[W,1280]` embeddings of the eval sample |
| `manifest.json` | `schema: 2`, one entry per clip, plus `labels_sha256`, `ethogram_version`, `backbone`, `aug_policy` |

The cache is split-agnostic: every labeled clip gets every variant and an eval
sample, and the train/eval choice is made per fold at training time
(`training/cache.py`). The frames tier is reused on a re-run, so changing the
augmentation or the backbone re-runs only the embed pass. Each clip is decoded
with one ffmpeg process per sample (`shared/frames.read_frames`), byte-identical
to the old one-process-per-frame decode; `training/check_decode_parity.py`
checks that over every recording. A schema-1 cache is refused rather than
overwritten: archive it first (see "After an ethogram change").

```bash
$TRAIN -m training.extract_frames [--seed N] [--eval-windows 8 12] [--cache-dir DIR]
```

`--eval-windows` lists every window length step 2 may be asked to evaluate at.
An eval sample is W frames spread across the whole clip, the protocol schema 1
used, so a window sweep needs one per length.

**2. Fit the temporal head.** Reads the cache, writes `models/head_state.pt`
and `models/train_result.json`. Until `crossval.py` lands this still trains on
the legacy single train/val split, rebuilt deterministically from the manifest
by `dataset.split_by_group`. Its val accuracy is a selection score.

```bash
$TRAIN -m training.train [--window N] [--seed N]
```

**3. Export to ONNX.** Reads `head_state.pt` and `train_result.json`, writes
`models/backbone_fp32.onnx`, `models/head.onnx`, `models/backbone_int8.onnx`
(plus `.data` sidecars for the two large graphs).

```bash
$TRAIN -m training.export_onnx
```

**There is no separate deploy step: this one writes in place and IS the
deploy.** `agent/capture.py` and `app/routers/model.py` read
`models/backbone_int8.onnx` and `models/head.onnx` directly. That is why
`models/gate_result.json` could read `"gate_passed": false` while the model was
live. A `models/candidate/` staging directory and a `promote.py` that enforces
the gate land in Phase 3 of `upgrade-plan.md`.

**4. Check the export.** Accuracy and confusion matrix for the fp32 and int8
pair, plus the int8-versus-fp32 quantization check. Writes
`models/quant_result.json`.

```bash
$TRAIN -m training.evaluate
```

These are **selection-split numbers, not held-out accuracy.** The same split
drove early stopping and checkpoint selection. Do not report them as accuracy.

**5. Verify runtime parity.** Confirms `inference/predictor.py`'s ring-buffer
path reproduces step 4's fp32 number exactly.

```bash
$TRAIN -m training.verify_predictor_parity
```

**6. Guard regressions.** Neither needs a model or a camera.

```bash
$TRAIN -m training.test_train_guards
$TRAIN -m training.test_export_guards
```

**7. Calibrate the alert threshold.** Runs the deployed int8 pair over every
labeled clip and reports what each `AGENT_INTEREST_THRESHOLD` would have caught.

```bash
$RUN -m training.calibrate_gate [--refresh] [--window N]
```

**Known limitation:** this currently calibrates on every labeled clip,
including the ones the model trained on, so the threshold is fitted in-sample.
Phase 3 moves it to out-of-fold probabilities. Until then, read its recall
column as optimistic.

## Off the path

- `training/diagnose_meanstd.py` is a diagnostic, not a pipeline step. It trains
  the old JS system's mean/std-pooling head on the current embeddings to
  separate "is the new backbone better" from "is the GRU head data-starved".
  Phase 3 folds it in as a proper cross-validated arm.
- `scripts/legacy/js_baseline_v1.mjs` reproduces the ethogram-v1 tfjs number
  (74.4%) and nothing else. It is retired from the pipeline: the tfjs model
  predicts five v1 classes and cannot predict v2's seven, so comparing a v2
  retrain against it was meaningless. Defaults to the archived v1 manifest in
  `models/archive/v1-ethogram/`.
- `scripts/bench_pi.py` measures inference latency on the target hardware.

## Tests

Every check-script runs through one pytest runner, from `server_py/`:

```bash
uvx --from pytest==9.1.1 pytest                       # all of them
uvx --from pytest==9.1.1 pytest -m "not local_data"   # what CI runs
```

`ci/test_scripts.py` runs each script as a subprocess under the venv it belongs
to (runtime for `tests/`, training for `training/test_*.py`) and asserts exit 0,
attaching the script's output on failure. pytest never collects the scripts
themselves; see that file's docstring for why. The scripts stay runnable on
their own, which is the faster loop when one is red:

```bash
.venv/bin/python tests/test_motion_parity.py
.venv/bin/python tests/test_security_regressions.py   # needs client/dist/ built
.venv/bin/python tests/test_server_robustness.py
.venv/bin/python tests/test_capture_loop.py
.venv/bin/python tests/test_clipping_mode.py
.venv-train/bin/python training/test_export_guards.py
.venv-train/bin/python training/test_train_guards.py
```

Each prints `PASS`, `FAIL` or `SKIP` per check through the shared helper in
`tests/checks.py`. A `SKIP` is a measurement this machine cannot make, not a
regression: `test_server_robustness.py`'s timing check skips if even its capped
batch cannot stall the event loop past the measured timer floor. Its
structural check, that the listing scan never runs on the event-loop thread,
always runs.

`test_security_regressions.py` reads `ADMIN_PASSWORD` and `AGENT_TOKEN` from
`server/.env`. Without them it fails two checks (login returns 503; the
stream-frame route cannot be reached). CI sets placeholder values.

`agent/test_capture_offline.py` is not in the runner. It is a smoke harness
that replays fixture frames through `AgentDecisionLoop` with the ONNX pair on
disk and asserts nothing beyond not crashing.

CI (`.github/workflows/ci.yml`) runs the `not local_data` set on a fresh
runner, building both venvs from the pinned requirements files (CPU torch) and
the client, plus `ruff check` as a separate job.

## After an ethogram change

Changing `shared/labels.py` invalidates the whole chain. The cache holds the old
label strings, the checkpoint has the old class count, and the exported head has
the old output width.

1. Archive, do not migrate: move `models/cache/`, `head_state.pt`,
   `train_result.json` and any result JSONs to
   `models/archive/<ethogram-version>/`.
2. Recode every clip against the new definitions. Do not write a mapping script;
   a scripted migration preserves exactly the errors a vocabulary change exists
   to remove.
3. Re-run from step 1.

The agent refuses to load a head whose output width disagrees with `LABELS` and
falls back to motion-only with a message saying so. Clipping Mode is unaffected,
which is the point: it never uses the model, so footage can still be harvested
while the dataset is rebuilt.
