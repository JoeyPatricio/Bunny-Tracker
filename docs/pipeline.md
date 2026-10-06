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

Run from `server_py/`.

```
extract_frames -> crossval -> train --all-clips -> export_onnx -> evaluate
               -> promote -> verify_predictor_parity -> calibrate_gate
```

Steps 1 to 5 and 7 use the training venv. Promote and calibrate_gate use the
runtime venv: neither needs torch, and promote is what runs on the box that
serves the model.

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

**2. Cross-validate.** Reads the cache, writes `models/cv_result.json`,
`models/cv_oof.npz` and appends to `models/runs/index.jsonl`. **This is the
only step whose numbers are results.**

```bash
$TRAIN -m training.crossval [--repeats 5] [--n-splits 5] [--n-inner 4] [--seed N]
```

Repeated, nested, grouped CV (`training/splits.py`) over four arms on
identical folds: `majority`, `static_logreg`, `static_meanstd_mlp` and `gru`
(the shipping head). Inner folds choose each arm's window (from the cache's
`--eval-windows`), learning rate or C, and epoch count; the arm is then
refitted on the outer-train clips and scored once on the outer eval clips,
which are never used for stopping or selection. Standardization buffers and
class weights are computed per fit from that fit's training clips only.

`cv_result.json` holds, per arm, the pooled out-of-fold macro-F1, balanced
accuracy and kappa (mean and sd over repeats), per-class recall with Wilson
intervals, the per-outer-fold selections (folds disagreeing on the window is a
result), and per-fold scores. `paired` compares `gru` against `static_logreg`
fold by fold with the Nadeau-Bengio corrected resampled t-test, plus an
uncorrected Wilcoxon for reference. `--quick` (one config per arm, 15-epoch
cap) is for smoke runs only.

It refuses to run if any class has fewer clips or source groups than
`--n-splits`, naming the class and the remedy.

**3. Fit the deployable head.** Reads the cache and `cv_result.json`, writes
`models/head_state.pt` and `models/train_result.json`.

```bash
$TRAIN -m training.train --all-clips [--seed N]
```

Trains on every labeled clip with the CV-selected config: the most common
window and learning rate across outer folds, and the median epoch count.
There is no held-out data in this mode, so `train_result.json` records the
config and a pointer to `cv_result.json` and no accuracy at all. It refuses a
`cv_result.json` computed on different labels than the cache.

`--fold R:K` fits one outer fold with its CV-selected config and prints that
fold's report, for debugging. With no flag, `train.py` still fits the legacy
single train/val split; its val accuracy is printed as a selection score and
not written anywhere.

**4. Export the candidate.** Reads `head_state.pt` and `train_result.json`,
writes `models/candidate/` (`backbone_fp32.onnx`, `backbone_int8.onnx`,
`head.onnx`, their `.data` sidecars, and `manifest.json` with each artifact's
sha256, the git SHA, labels, ethogram version and window). Nothing the agent
loads is touched.

```bash
$TRAIN -m training.export_onnx
```

**5. Evaluate (being rewritten: deployment gates G1-G7).** Today it reports
fp32 and int8 accuracy on the legacy split's val clips and writes
`models/quant_result.json`; the rewrite evaluates `models/candidate/` and
writes `models/candidate/gate_result.json`, which promote requires.

```bash
$TRAIN -m training.evaluate
```

**6. Promote.** Refuses unless `candidate/gate_result.json` says
`gate_passed: true` and every artifact's sha256 matches the candidate
manifest. Archives the deployed `models/*.onnx` to
`models/archive/<timestamp>/`, installs the candidate, and writes
`models/promoted.json`. `--force "<reason>"` overrides the gate (never the hash
check) and records the reason. Restart the agent afterwards: it loads the
model at startup.

```bash
$RUN -m training.promote [--force "reason"]
```

**7. Verify runtime parity.** Confirms `inference/predictor.py`'s ring-buffer
path reproduces the export's fp32 number exactly.

```bash
$TRAIN -m training.verify_predictor_parity
```

**8. Calibrate the alert threshold.** Reads the `gru` arm's out-of-fold
probabilities from `models/cv_oof.npz` and reports what each
`AGENT_INTEREST_THRESHOLD` would have caught and woken you for. Every
probability comes from a fold that never trained on that clip, so the
threshold is not fitted in-sample. It loads no ONNX model.

```bash
$RUN -m training.calibrate_gate [--oof models/cv_oof.npz]
```

## Dataset reports

Both run on the runtime venv and write to `models/`.

- `training/recoding_report.py` cross-tabulates the v1 labels
  (`server/labels.v1.json`) against the v2 recode into
  `models/recoding_report.json`: what each v1 class became.
- `training/reliability.py` computes Cohen's kappa, overall and one-vs-rest per
  class, between the first pass (`server/labels.json`) and the blind 25%
  re-code (`server/labels.reliability.json`) into `models/reliability.json`.
  It refuses until the re-code is complete. This kappa is the ceiling any
  classifier can be compared against.

## Off the path

- `training/diagnose_meanstd.py` is the old single-split diagnostic. Its
  architecture now runs as crossval's `static_meanstd_mlp` arm, which is the
  number to use; the script itself still reports on the legacy split.
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
.venv/bin/python tests/test_labeling_sidecars.py
.venv/bin/python training/test_promote_reports.py
.venv-train/bin/python training/test_export_guards.py
.venv-train/bin/python training/test_train_guards.py
.venv-train/bin/python training/test_splits_metrics.py
.venv-train/bin/python training/test_cache_v2.py
.venv-train/bin/python training/test_crossval_guards.py      # also runs calibrate_gate under .venv
.venv-train/bin/python training/check_decode_parity.py       # local_data: needs server/recordings/
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
