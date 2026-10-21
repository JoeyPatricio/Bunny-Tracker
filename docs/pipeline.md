# The training pipeline

Commands run from `server/`. Producing real results needs recoded clips in
`data/recordings/` and `data/labels.json`. Without them only the hermetic
tests run.

## The two-venv rule

| | `.venv` | `.venv-train` |
|---|---|---|
| Python | 3.14 | 3.12 |
| torch | no | yes, CPU build |
| Runs | server, agent, `tests/`, `promote`, `calibrate_gate` | the rest of `training/` |
| Deployed to the Pi | yes | no |

The runtime venv ships to the Pi, where inference is ONNX Runtime on CPU, so
it stays torch-free. The training venv is on 3.12 because torch wheels lag
behind new Python releases.

`pyproject.toml` defines dependencies and `uv.lock` pins both environments.
torch and torchvision come from a CPU-only index. No CUDA packages get
installed.

```bash
uv sync --frozen --python 3.14 --group dev
UV_PROJECT_ENVIRONMENT=.venv-train uv sync --frozen --python 3.12 --group train
```

Drop `--group dev` for a deployment. To change a dependency, edit
`pyproject.toml`, run `uv lock`, then regenerate the requirements files
(they're generated, don't edit them by hand):

```bash
uv export --frozen --no-hashes --no-annotate --no-emit-project --output-file requirements.txt
uv export --frozen --no-hashes --no-annotate --no-emit-project --emit-index-url --group train --output-file requirements-train.txt
uv export --frozen --no-hashes --no-annotate --no-emit-project --emit-index-url --group train --output-file requirements-train-cpu.txt
```

Prefer `uv sync --frozen`. The requirements format can't keep the per-package
CPU index.

## The order

```
extract_frames -> crossval -> train --all-clips -> export_onnx -> evaluate
               -> promote -> verify_predictor_parity -> calibrate_gate
```

```bash
TRAIN=.venv-train/bin/python
RUN=.venv/bin/python
```

### 1. Extract and embed

```bash
$TRAIN -m training.extract_frames [--eval-windows 8 12] [--aug-policy legacy|albu_v1] [--view full|crop]
```

Reads the labels and recordings and writes a cache to `models/cache/`: raw
frames, 1280-d embeddings for each augmented variant, eval samples for each
window length in `--eval-windows`, and a `manifest.json` with the label hash,
ethogram version, backbone and augmentation policy. The cache doesn't depend
on any split. Folds are picked at training time. Re-runs reuse the frames, so
changing the backbone or augmentation only re-embeds. An old schema-1 cache is
refused, not overwritten (see [After an ethogram change](#after-an-ethogram-change)).

Two experimental options, neither evaluated on real labels yet:

- **`--aug-policy albu_v1`** adds mild rotation, perspective, blur and
  occlusion (8 variants vs. 3 for `legacy`). Build both, then run
  `crossval --aug-policies legacy albu_v1` to compare them on the same folds.
  Only adopt it if it wins.
- **`--view crop`** crops each clip to a box around detected motion. Compare
  with `crossval --views full crop`. It's offline only. Runtime has no crop
  step, so train and export refuse crop heads.

### 2. Cross-validate

```bash
$TRAIN -m training.crossval [--repeats 5] [--n-splits 5] [--n-inner 4]
```

**This is the only step whose numbers are results.** It runs repeated, nested,
grouped CV over four arms on the same folds: `majority`, `static_logreg`,
`static_meanstd_mlp` and `gru` (the shipping head). Inner folds pick each
arm's window, learning rate and epochs. Outer eval clips are never used for
selection or stopping.

Output is `models/cv_result.json`, `models/cv_oof.npz`, and a line in
`models/runs/index.jsonl`. Each arm gets out-of-fold macro-F1, balanced
accuracy and kappa (mean and sd over repeats), per-class recall with Wilson
intervals, and a group bootstrap 95% CI on macro-F1. `gru` vs.
`static_logreg` is compared with the Nadeau-Bengio corrected t-test.

It refuses to run if a class has fewer clips or source groups than
`--n-splits`. `--quick` is for smoke tests only.

### 3. Fit the deployable head

```bash
$TRAIN -m training.train --all-clips [--arm gru|gru@albu_v1]
```

Trains on every clip using the config CV selected. With no held-out data,
`train_result.json` records the config and a pointer to the CV run, not an
accuracy. It refuses a CV result computed on different labels. `--fold R:K`
fits a single outer fold for debugging.

### 4. Export

```bash
$TRAIN -m training.export_onnx
```

Writes `models/candidate/`: fp32 and int8 backbones, `head.onnx`, and a
manifest with hashes, git SHA, labels, ethogram version and window. Doesn't
touch anything the agent loads.

### 5. Evaluate (the deployment gate)

```bash
$TRAIN -m training.evaluate
```

Writes `candidate/gate_result.json` and exits 1 if any gate fails.

| Gate | Fails when |
|---|---|
| G1 | head output width isn't `len(LABELS)` |
| G2 | batched output differs from one-at-a-time output (1e-5) |
| G3 | ONNX head differs from the torch head (1e-4) |
| G4 | int8 macro-F1 is more than 0.03 below fp32 |
| G5 | int8 predicts one class for every clip |
| G6 | GRU doesn't beat `static_logreg` in CV, provenance mismatches, or the CV run was `--quick` |
| G7 | OOF macro-F1 is more than one sd below the deployed model's |

G4 and G5 run in-sample, so only the fp32 vs. int8 difference matters there.
G7 passes when nothing is deployed yet. The reasoning for each gate is in
`training/evaluate.py`.

### 6. Promote

```bash
$RUN -m training.promote [--force "reason"]
```

Requires a passing gate and matching hashes. Archives the current models to
`models/archive/<timestamp>/`, installs the candidate, and writes
`models/promoted.json`. `--force` skips the gate but never the hash check, and
records why. Restart the agent afterwards.

### 7. Verify runtime parity

```bash
$TRAIN -m training.verify_predictor_parity
```

Checks that the agent's frame-by-frame ring buffer gives the same
probabilities as a batched run (within 1e-5).

### 8. Calibrate the alert threshold

```bash
$RUN -m training.calibrate_gate [--arm gru@albu_v1]
```

Uses out-of-fold probabilities from `cv_oof.npz` to show what each
`AGENT_INTEREST_THRESHOLD` would have caught. Calibrate the arm you actually
deployed.

## Dataset reports

Both use the runtime venv and write to `models/`.

- `training.recoding_report` cross-tabs v1 labels against the v2 recode.
- `training.reliability` computes Cohen's kappa between the first pass and the
  blind 25% re-code. It refuses until the re-code is complete.

## Other scripts

- `training/diagnose_meanstd.py` is the old single-split diagnostic. Use
  crossval's `static_meanstd_mlp` arm instead.
- `scripts/legacy/js_baseline_v1.mjs` reproduces the retired v1 tfjs number
  (74.4%). It predicts v1's five classes, so don't compare v2 against it.
- `scripts/bench_pi.py` measures inference latency on the Pi.
- `training.pose_features`, `training.learning_curve` and `training.figures`
  are for the paper. They error on missing inputs instead of filling in
  placeholders.

## Tests

```bash
.venv/bin/python -m pytest                       # everything
.venv/bin/python -m pytest -m "not local_data"   # what CI runs
```

`ci/test_scripts.py` runs each check script as a subprocess in the right venv
(`tests/` in the runtime venv, `training/test_*.py` in the training venv) and
expects exit 0. Each script also runs on its own, which is quicker when one
fails:

```bash
.venv/bin/python tests/test_capture_loop.py
.venv-train/bin/python training/test_evaluate_gates.py
```

Scripts print `PASS`, `FAIL` or `SKIP` per check. `SKIP` means the machine
couldn't make the measurement, not a regression. `check_decode_parity.py`
needs local recordings. `agent/test_capture_offline.py` is a smoke harness
and isn't in the runner.

CI (`.github/workflows/ci.yml`) builds both venvs from `uv.lock`, builds the
client, runs the `not local_data` set, and runs `ruff check`.

## After an ethogram change

Changing `shared/labels.py` invalidates the cache, the checkpoint and the
exported head.

1. Move `models/cache/`, `head_state.pt`, `train_result.json` and the result
   JSONs to `models/archive/<ethogram-version>/`.
2. Recode every clip. Don't write a mapping script.
3. Re-run from step 1.

Until a new head is promoted, the agent refuses the old one and runs
motion-only. Clipping Mode doesn't use the model, so footage keeps coming in
while the dataset is rebuilt.
