# Offline experiments and paper figures

No animal experiment has been run for these additions. The repository does not
contain the recoded labels, complete keypoints, or measured learning-curve runs.
Synthetic tests exercise the code but provide no evidence about rabbit behavior.
Run these commands from `server_py/` with the CPU training environment described
in [the pipeline](../docs/pipeline.md). Nothing here belongs in the runtime venv.
Figure export needs `matplotlib==3.11.2` in the training environment.

## Pose input

`training.pose_features` accepts JSON with these required fields:

| Field | Contract |
|---|---|
| `schema` | Integer `1` |
| `coordinate_system` | `image_pixels`, x right and y down |
| `node_names` | In order: `nose`, `neck`, `rump`, `left_ear_tip`, `right_ear_tip`, `tail_base` |
| `source` | `kind` and `description` describing the annotation protocol or independent model |
| `clips` | One entry for every clip in the reference embedding manifest, with no duplicates or extra clips |
| `clips[].filename` | Exact filename in the manifest |
| `clips[].track_id` | Nonempty string identifying one stable animal track |
| `clips[].timestamps` | Strictly increasing finite timestamps in seconds, at least three |
| `clips[].keypoints` | Finite `[frames, 6, 2]` coordinates in the declared node order |

Use manually reviewed annotations (`source.kind = manual`) or a model trained
outside the entire CV pool (`source.kind = external_model`). For an external
model, record its model/weights version and training dataset in `description` and
`training_data`, and explicitly declare `cv_pool_excluded = true`. This declaration
must cover all clips and related recording sessions. A pose estimator trained on
this pool needs its own nested training protocol, which this CLI does not supply.
The reader validates the declaration and cannot independently audit its truth.

The track must correspond to the rabbit whose behavior the clip label describes.
Resolve identity switches and occluded keypoints during annotation review. Missing
coordinates, degenerate body axes, and incomplete clip coverage stop the run.
There is no automatic interpolation, confidence threshold, or choice of rabbit.
Use a fixed timestamp sampling protocol across clips and record it in `source`.
Derivative features depend on that protocol and its temporal coverage.

The 40 features are eight normalized distances with mean and standard deviation,
body-axis sine and cosine with mean and standard deviation, each ear's cosine to
the body axis with mean and standard deviation, mean speed and acceleration for
each of six keypoints, bounding-box area mean and standard deviation, mean absolute
area rate, and body-length coefficient of variation. The scale is the clip's median
neck-to-rump distance. Speed uses body lengths per second and acceleration uses
body lengths per second squared. Area uses squared body lengths. Axis orientation
and the axis-aligned bounding box retain the camera's orientation. Features are
computed within each clip and use no population statistics.

```bash
.venv-train/bin/python -m training.pose_features \
  --keypoints models/pose_keypoints.json \
  --manifest models/cache/manifest.json \
  --reference models/cv_result.json --oof models/cv_oof.npz \
  --out-dir models/pose
```

This regenerates the reference nested grouped folds with the same clip order,
seed, split counts, and repeats. It verifies label hashes, ethogram, reference
OOF metrics and fold metrics before fitting. Standardization and class balancing
are fitted only on each fit's training subset. Inner-fold log loss selects `C`
from `0.01`, `0.1`, `1.0`. Each selected model is refitted on outer training clips
and scored once on the outer evaluation clips. The paired comparator defaults to
`static_logreg`. This is a comparison of information sources, since pose uses
additional keypoint information. It does not isolate temporal aggregation.

`pose_result.json` records the feature schema, complete inner/outer membership,
input SHA-256s, implementation hash, package versions and paired fold differences.
`pose_oof.npz` contains `pose_logreg_P`, `y` and `filenames` with the reference OOF
layout. The corrected resampled test uses mean evaluation/training fold ratios.
A constant nonzero fold difference has no estimable p-value and is recorded as
null, rather than interpreted as certainty.

SLEAP is an optional external annotation tool. Its upstream
[LICENSE](https://github.com/talmolab/sleap/blob/develop/LICENSE) identified the
license as Clear BSD when checked on 2026-10-19, rather than the upgrade plan's
assumed BSD-3. Keep any installation in a separate environment and record the
specific release, license and export procedure with the annotations. This path
neither installs nor imports SLEAP. No pose framework is needed for these features.

## Measured learning curve

```bash
.venv-train/bin/python -m training.learning_curve \
  --cache-dir models/cache --manifest models/cache/manifest.json \
  --reference models/cv_result.json --oof models/cv_oof.npz \
  --fractions 0.25 0.5 0.75 1 --arms static_logreg gru \
  --out-dir models/learning
```

The producer reuses `crossval.run_fold` and the reference run's windows,
regularization searches, epoch cap and patience. Each outer evaluation set stays
fixed. Fractions select seeded prefixes of complete outer-training source groups,
so larger subsets contain smaller subsets. Each subset gets its own stratified
grouped inner folds. No dropped training group moves into evaluation. Unsupported
class/group counts fail before any model is fitted. Collect more independent
groups or explicitly request fewer training sizes if a small subset cannot support
the registered inner-fold design. Do not reduce split counts to force it through.

Runs are sequential with one thread per fit. Full GRU experiments can take much
longer than the synthetic tests. A reference `--quick` run produces quick
learning-curve runs and is only a smoke test, never publication evidence.

Each `models/learning/point-N/` contains an actual `cv_result.json` and
`cv_oof.npz`. Results record group fractions, actual training clip/group counts
for every outer fit, full split membership, reference input hashes and code
hashes. `models/learning/learning_curve.json` indexes these measured artifacts.
Pose and learning reports include a source-group bootstrap interval for pooled
macro-F1, conditional on fitted OOF predictions and excluding retraining uncertainty.
Figure export verifies the recorded counts against training membership and checks
that outer evaluation membership agrees across all points.

## Figure generation

```bash
# Figures with existing primary CV/OOF reports:
.venv-train/bin/python -m training.figures \
  --only head_ablation oof_confusion --arm gru --repeat 0

# All six require all reports listed below:
.venv-train/bin/python -m training.figures
```

| Figure | Inputs under `server_py/models/` | Producer |
|---|---|---|
| `head_ablation` | `cv_result.json`, `cv_oof.npz` | `training.crossval` |
| `oof_confusion` | Same, with selected arm and repeat | `training.crossval` |
| `recoding` | `recoding_report.json` | `training.recoding_report` |
| `kappa` | Primary CV/OOF plus `reliability.json` | `training.crossval`, `training.reliability` |
| `quantization` | `candidate/quant_result.json` | `training.evaluate` |
| `learning_curve` | `learning/learning_curve.json` and indexed CV/OOF files | `training.learning_curve` |

Defaults write SVG and PDF to `paper/figures/`, plus a JSON sidecar containing the
unrounded plotted values, all source paths and hashes, source run/label/ethogram
identifiers when available, generator hash and output hashes. Source references
are also embedded in SVG/PDF metadata. Quantization additionally exports CSV.
JSON/OOF metric disagreement stops generation. All requested inputs are validated
before output creation. `--only` permits an explicit partial export. Missing
requested inputs are errors and never become empty or simulated plots.

Bars and learning curves show mean and standard deviation of pooled OOF metrics
across repeats. These error bars are descriptive variation, not confidence
intervals. The confusion matrix shows one repeat to keep the clip count real.
The reliability point is a blind intra-observer subset, with no estimated
interval, and may include coding categories outside the model vocabulary. It
is not an inter-observer measurement or a mathematical ceiling on model kappa.
The quantization scores are in-sample deployment checks of the final all-clips
model. The current report contains no per-tensor result or measured latency,
so neither is fabricated or extrapolated.

## Paper freeze

Joe chooses the claim, venue and publishing date after the measured runs. Freeze
the code revision, environment lock, clip/label manifest, annotation provenance,
CV/OOF reports, learning-curve/pose reports, reliability and recoding reports,
candidate gate/quantization reports and ONNX artifact hashes. Regenerate figures
from that snapshot and use their sidecar values for manuscript numbers. Never
transfer the archived v1 selection scores into v2 results.

A release of frozen embeddings can reproduce head fitting without raw recordings.
It does not reproduce video decoding, alternate backbones, crop extraction or
pose estimation. Embeddings are not a proven anonymization method. Review privacy
before releasing data. The single-camera environment, small sample, targeted clip
collection, clip-level weak labels, individual animal ambiguity and intra-observer
reliability design remain limitations to describe, regardless of which arm wins.
