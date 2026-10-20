"""Repeated, nested, grouped cross-validation over the schema-2 cache.

This is where the project's reported numbers come from (docs/upgrade-plan.md,
Phase 3). Every arm is evaluated on the identical folds from
training/splits.make_folds, so arms are paired and their per-fold differences
can be tested directly.

Arms:
    majority            Predicts the outer-train majority class. The floor:
                        balanced accuracy is exactly 1/len(LABELS) per fold.
    static_logreg       mean||std pooled embeddings into LogisticRegression.
                        The primary comparator: same frozen embeddings as the
                        GRU, so the only variable is temporal aggregation.
    static_meanstd_mlp  diagnose_meanstd.py's MeanStdClassifier on the same
                        pooled features.
    gru                 The shipping TemporalHead (train.fit_head).

Nesting. Inside each outer fold, the outer-train clips are split again into
inner folds. Each arm's hyperparameters (window, learning rate or C, and for
the neural arms the epoch count) are chosen by mean inner-val loss, then the
arm is refitted on the full outer-train set with that config for exactly the
chosen number of epochs, and scored once on the outer eval clips. The outer
eval clips are never used for stopping or selection. The selections are
recorded per outer fold, because folds disagreeing on the window is a result.

Leakage. Every quantity fitted from data comes from that fit's training clips
only: the GRU's emb_mean/emb_std buffers and class weights (train.fit_head),
the static arms' feature scaler and class weights, the majority class. For an
inner fit that is the inner-train clips, for an outer fit the outer-train
clips. test_crossval_guards.py checks the GRU's buffers and weights per fold.

Pooled out-of-fold. Within one repeat every clip is evaluated exactly once, so
the repeat's pooled confusion matrix has the whole dataset as its support. The
headline per arm is the mean and sd over repeats of the pooled macro-F1,
balanced accuracy and kappa. Per-class recall intervals are Wilson intervals
on the dataset's real clip count per class (mean recall over repeats times the
class support), not on the repeat-summed matrix, whose support is inflated by
counting each clip once per repeat.

Paired test, gru minus static_logreg. The unit is one outer fold of one
repeat (macro-F1 on that fold's eval clips). Fold scores from repeated k-fold
are not independent: every pair of folds shares most of its training data, so
a plain paired t-test or Wilcoxon treats correlated differences as independent
and is overconfident. The primary test is therefore the Nadeau-Bengio
corrected resampled t-test (Nadeau and Bengio 2003; Bouckaert and Frank
2004), which inflates the variance by n_eval/n_train for exactly this case.
Wilcoxon signed-rank on the same differences is reported alongside as a
secondary, uncorrected check.

Outputs (in --out-dir, default models/):
    cv_result.json      config, per-arm metrics, selections, per-fold numbers,
                        the paired comparison
    cv_oof.npz          out-of-fold probabilities per arm, [repeats, clips,
                        classes], plus y and filenames. calibrate_gate.py
                        reads the gru arm.
    runs/index.jsonl    one line per arm per run, appended

Usage (from server_py/, training venv):
    .venv-train/bin/python -m training.crossval [--repeats 5] [--n-splits 5]
        [--n-inner 4] [--seed 0] [--quick] [--jobs N] [--cache-dir DIR] [--out-dir DIR]
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.labels import LABELS
from training import train as T
from training.cache import DEFAULT_CACHE_DIR, LEGACY_POLICY, load_manifest, split_entries
from training.dataset import class_weights
from training.diagnose_meanstd import MeanStdClassifier
from training.metrics import classification_report, confusion_matrix, wilson_interval
from training.splits import make_folds

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ARMS = ("majority", "static_logreg", "static_meanstd_mlp", "gru")
PRIMARY_COMPARISON = ("gru", "static_logreg")

# Search spaces. Small on purpose: at ~112 clips every extra config is another
# chance for the inner folds to select noise.
GRU_LRS = (3e-4, 1e-3)
MLP_LRS = (1e-3, 3e-4)
LOGREG_CS = (0.01, 0.1, 1.0)


# ── features for the static arms ─────────────────────────────────────────

def pooled(window: np.ndarray) -> np.ndarray:
    """[N,1280] -> [2560] mean||std, diagnose_meanstd.mean_std_features."""
    return np.concatenate([window.mean(axis=0), window.std(axis=0)])


def static_features(entries: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """One pooled feature per training window (train entries) or per clip
    (eval entries, which carry one window). Pooling each W-frame window, not
    the whole 16-frame sample, so train and eval features summarise the same
    number of frames."""
    xs, ys = [], []
    for e in entries:
        for start, end in e["windows"]:
            xs.append(pooled(e["embeddings"][start:end]))
            ys.append(LABELS.index(e["label"]))
    return np.stack(xs).astype(np.float32), np.array(ys)


def scale(x_train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """Standardize with the training rows' statistics only."""
    mean, std = x_train.mean(axis=0), x_train.std(axis=0) + 1e-6
    return [(x - mean) / std for x in (x_train, *others)]


def log_loss(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))))


# ── one fit per arm: returns eval probabilities and the inner-val loss ───

def fit_majority(train_entries, eval_entries, **_) -> dict:
    counts = np.bincount([LABELS.index(e["label"]) for e in train_entries], minlength=len(LABELS))
    probs = np.zeros((len(eval_entries), len(LABELS)))
    probs[:, int(counts.argmax())] = 1.0
    return {"probs": probs}


def fit_logreg(train_entries, eval_entries, *, C: float, seed: int, **_) -> dict:
    x_tr, y_tr = static_features(train_entries)
    x_ev, y_ev = static_features(eval_entries)
    x_tr, x_ev = scale(x_tr, x_ev)
    weights = class_weights(train_entries).numpy()
    model = LogisticRegression(C=C, max_iter=2000, random_state=seed,
                               class_weight={k: float(w) for k, w in enumerate(weights)})
    model.fit(x_tr, y_tr)
    probs = np.zeros((len(x_ev), len(LABELS)))
    probs[:, model.classes_] = model.predict_proba(x_ev)
    return {"probs": probs, "val_loss": log_loss(probs, y_ev)}


def fit_mlp(train_entries, eval_entries, *, lr: float, seed: int, epochs: int | None,
            patience: int, early_stop: bool, **_) -> dict:
    x_tr, y_tr = static_features(train_entries)
    x_ev, y_ev = static_features(eval_entries)
    x_tr, x_ev = scale(x_tr, x_ev)
    weights = class_weights(train_entries)
    to_t = torch.from_numpy
    fit = T.fit_classifier(
        MeanStdClassifier(), to_t(x_tr), to_t(y_tr).long(),
        to_t(x_ev) if early_stop else None, to_t(y_ev).long() if early_stop else None,
        weights=weights, lr=lr, seed=seed, epochs=epochs, patience=patience,
    )
    with torch.no_grad():
        probs = torch.softmax(fit["model"](to_t(x_ev)), dim=1).numpy()
    return {"probs": probs, "val_loss": fit["best_val_loss"], "best_epoch": fit["best_epoch"]}


def fit_gru(train_entries, eval_entries, *, lr: float, seed: int, epochs: int | None,
            patience: int, early_stop: bool, **_) -> dict:
    fit = T.fit_head(train_entries, eval_entries if early_stop else None, lr=lr, seed=seed,
                     epochs=epochs, patience=patience)
    x_ev, _ = T.eval_tensors(eval_entries)
    with torch.no_grad():
        probs = torch.softmax(fit["head"](x_ev), dim=1).numpy()
    return {"probs": probs, "val_loss": fit["best_val_loss"], "best_epoch": fit["best_epoch"],
            "emb_mean": fit["emb_mean"], "emb_std": fit["emb_std"],
            "class_weights": fit["class_weights"]}


FITTERS = {"majority": fit_majority, "static_logreg": fit_logreg,
           "static_meanstd_mlp": fit_mlp, "gru": fit_gru}


def search_space(arm: str, windows: list[int], quick: bool) -> list[dict]:
    if arm == "majority":
        return [{"window": windows[0]}]
    if arm == "static_logreg":
        cs = LOGREG_CS[1:2] if quick else LOGREG_CS
        return [{"window": w, "C": c} for w in windows for c in cs]
    lrs = (GRU_LRS if arm == "gru" else MLP_LRS)[: 1 if quick else None]
    return [{"window": w, "lr": lr} for w in windows for lr in lrs]


def fit_seed(seed: int, *parts: int) -> int:
    """A fit's seed from the run seed and its position, so any single fit can
    be reproduced alone and does not depend on fit order."""
    return int(np.random.SeedSequence([seed, *parts]).generate_state(1)[0] % 2**31)


# ── the nested loop ──────────────────────────────────────────────────────

def run_fold(arm: str, manifest: dict, fold, windows: list[int], *, seed: int, quick: bool,
             epochs: int, patience: int, cache_dir: Path, on_fit=None) -> dict:
    """Select on inner folds, refit on outer-train, score outer-eval.

    One thread per fold, always, whether folds run in parallel or not. The
    thread count (torch's, and BLAS's under sklearn) changes floating-point
    reduction order, so pinning it is what makes a result independent of
    --jobs. These models are small enough that one thread loses little, and
    multi-threaded BLAS on them was mostly oversubscription: logreg ran ~19x
    faster pinned. The parallelism is across folds instead.
    """
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):  # BLAS under sklearn, same reasoning
        return _run_fold(arm, manifest, fold, windows, seed=seed, quick=quick, epochs=epochs,
                         patience=patience, cache_dir=cache_dir, on_fit=on_fit)


def _run_fold(arm, manifest, fold, windows, *, seed, quick, epochs, patience, cache_dir, on_fit):
    configs = search_space(arm, windows, quick)
    needs_epochs = arm in ("gru", "static_meanstd_mlp")
    scored = []
    for c_idx, config in enumerate(configs):
        if arm == "majority":
            scored.append((0.0, config, None))
            break
        losses, best_epochs = [], []
        for j, (inner_train, inner_val) in enumerate(fold.inner):
            tr, ev = split_entries(manifest, inner_train, inner_val, window=config["window"],
                                   cache_dir=cache_dir)
            res = FITTERS[arm](tr, ev, **config, seed=fit_seed(seed, fold.repeat, fold.outer, j, c_idx),
                               epochs=epochs, patience=patience, early_stop=True)
            losses.append(res["val_loss"])
            if needs_epochs:
                best_epochs.append(res["best_epoch"] + 1)
            if on_fit:
                on_fit(arm, fold, "inner", inner_train, res)
        scored.append((float(np.mean(losses)), config, best_epochs))

    inner_loss, config, best_epochs = min(scored, key=lambda s: s[0])
    selection = {"repeat": fold.repeat, "outer": fold.outer, **config}
    if arm != "majority":
        selection["inner_val_loss"] = inner_loss
    if needs_epochs:
        selection["epochs"] = int(round(float(np.median(best_epochs))))

    tr, ev = split_entries(manifest, fold.train, fold.eval, window=config["window"], cache_dir=cache_dir)
    params = {k: v for k, v in config.items() if k != "window"}
    res = FITTERS[arm](tr, ev, **params, seed=fit_seed(seed, fold.repeat, fold.outer, 99),
                       epochs=selection.get("epochs"), patience=patience, early_stop=False)
    if on_fit:
        on_fit(arm, fold, "outer", fold.train, res)
    y = np.array([LABELS.index(e["label"]) for e in ev])
    preds = res["probs"].argmax(axis=1)
    report = classification_report(confusion_matrix(y.tolist(), preds.tolist()))
    return {"selection": selection, "probs": res["probs"], "y": y, "eval_idx": np.asarray(fold.eval),
            "fold_metrics": {k: report[k] for k in ("n", "macro_f1", "balanced_accuracy", "kappa")}}


def corrected_resampled_t(diffs: np.ndarray, n_train: float, n_eval: float) -> dict:
    """Nadeau-Bengio corrected resampled t-test on per-fold differences."""
    j = len(diffs)
    mean = float(diffs.mean())
    var = float(diffs.var(ddof=1)) if j > 1 else 0.0
    denom = math.sqrt((1 / j + n_eval / n_train) * var) if var > 0 else 0.0
    if denom == 0:
        return {"t": None, "df": j - 1, "p_two_sided": 1.0 if mean == 0 else 0.0}
    t = mean / denom
    return {"t": t, "df": j - 1, "p_two_sided": float(2 * stats.t.sf(abs(t), df=j - 1))}


def paired_comparison(per_fold: dict, a: str, b: str, n_clips: int, n_splits: int) -> dict:
    da = np.array([f["macro_f1"] for f in per_fold[a]])
    db = np.array([f["macro_f1"] for f in per_fold[b]])
    d = da - db
    n_eval = n_clips / n_splits
    out = {
        "metric": "macro_f1 per outer fold",
        "a": a, "b": b,
        "differences": d.tolist(),
        "mean_difference": float(d.mean()),
        "sd_difference": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        "folds_a_better": int((d > 0).sum()), "folds_b_better": int((d < 0).sum()),
        "ties": int((d == 0).sum()),
        "corrected_resampled_t": corrected_resampled_t(d, n_clips - n_eval, n_eval),
    }
    if (d != 0).any():
        w = stats.wilcoxon(d)
        out["wilcoxon_uncorrected"] = {"statistic": float(w.statistic), "p_two_sided": float(w.pvalue)}
    else:
        out["wilcoxon_uncorrected"] = None
    return out


def summarise(arm_runs: list[dict], y_all: np.ndarray, repeats: int) -> tuple[dict, np.ndarray]:
    n = len(y_all)
    oof = np.full((repeats, n, len(LABELS)), np.nan)
    for run in arm_runs:
        r = run["selection"]["repeat"]
        if not np.isnan(oof[r, run["eval_idx"]]).all():
            raise ValueError("a clip received an out-of-fold prediction from two folds")
        oof[r, run["eval_idx"]] = run["probs"]
    if np.isnan(oof).any():
        raise ValueError("some clip received no out-of-fold prediction")

    per_repeat = []
    for r in range(repeats):
        cm = confusion_matrix(y_all.tolist(), oof[r].argmax(axis=1).tolist())
        per_repeat.append(classification_report(cm))

    def mean_sd(key):
        vals = np.array([rep[key] for rep in per_repeat])
        return {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "per_repeat": vals.tolist()}

    support = np.bincount(y_all, minlength=len(LABELS))
    per_class = {}
    for k, label in enumerate(LABELS):
        recall = float(np.mean([rep["per_class"][label]["recall"] for rep in per_repeat]))
        lo, hi = wilson_interval(int(round(recall * support[k])), int(support[k]))
        per_class[label] = {"recall_mean": recall, "recall_ci95": [lo, hi], "support": int(support[k])}

    summary = {
        "pooled_oof": {key: mean_sd(key) for key in ("macro_f1", "balanced_accuracy", "kappa", "accuracy")},
        "per_class": per_class,
        "per_repeat_confusion": [rep["confusion_matrix"] for rep in per_repeat],
        "selections": [run["selection"] for run in arm_runs],
        "per_fold": [{"repeat": run["selection"]["repeat"], "outer": run["selection"]["outer"],
                      **run["fold_metrics"]} for run in arm_runs],
    }
    return summary, oof


def git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def check_policies_comparable(a: dict, b: dict, name_a: str, name_b: str) -> None:
    """Two policy manifests are only a paired comparison if everything except
    the augmentation is identical: same clips in the same order (so fold
    indices mean the same clips), same labels, backbone and eval tiers."""
    same = {
        "clips": [(c["filename"], c["label"]) for c in a["clips"]]
                 == [(c["filename"], c["label"]) for c in b["clips"]],
        **{k: a.get(k) == b.get(k) for k in ("labels_sha256", "backbone", "eval_windows", "window",
                                             "train_frames", "labels")},
    }
    bad = [k for k, ok in same.items() if not ok]
    if bad:
        raise SystemExit(
            f"Aug policies {name_a!r} and {name_b!r} were extracted from different inputs "
            f"(differs: {bad}). Rebuild both with training/extract_frames.py against the same "
            f"labels.json and backbone before comparing them."
        )


def crossval(*, cache_dir: Path = DEFAULT_CACHE_DIR, out_dir: Path = MODELS_DIR, repeats: int = 5,
             n_splits: int = 5, n_inner: int = 4, seed: int = 0, quick: bool = False,
             arms=ARMS, on_fit=None, jobs: int = 1, aug_policies: list[str] | None = None) -> dict:
    """`jobs` outer folds run in parallel processes. `on_fit` (a test hook that
    sees every fit) needs the fits in this process, so it forces jobs=1.

    `aug_policies` (Phase 4): the first policy is primary and every arm runs on
    its cache, under the usual arm names, so evaluate.py, train.py and
    calibrate_gate.py are unaffected. Each further policy adds one arm,
    `gru@<policy>`, fitted on the identical folds with the identical fit
    seeds, so the only difference between it and `gru` is the augmentation.
    The corrected paired test for each lands in result["paired_aug_policy"].
    Only gru gets the extra arms: it is the shipping model, and the question
    Phase 4 asks is whether augmentation helps it.
    """
    started = time.time()
    jobs = 1 if on_fit is not None else max(1, jobs)
    saved_threads = torch.get_num_threads()
    policies = list(dict.fromkeys(aug_policies or [LEGACY_POLICY]))
    manifest = load_manifest(cache_dir, policies[0])
    extra_manifests = {pol: load_manifest(cache_dir, pol) for pol in policies[1:]}
    for pol, other in extra_manifests.items():
        check_policies_comparable(manifest, other, policies[0], pol)
    clips = manifest["clips"]
    filenames = [c["filename"] for c in clips]
    labels = [c["label"] for c in clips]
    folds = make_folds(filenames, labels, n_splits=n_splits, n_inner=n_inner, repeats=repeats, seed=seed)
    windows = sorted(int(w) for w in manifest["eval_windows"])
    if quick:
        windows = [manifest["window"]] if manifest["window"] in windows else windows[:1]
    epochs = 15 if quick else T.EPOCHS
    patience = 5 if quick else T.PATIENCE
    y_all = np.array([LABELS.index(lab) for lab in labels])

    # (result name, fitter arm, manifest). Extra-policy arms reuse gru's fitter.
    plan = [(arm, arm, manifest) for arm in arms]
    if "gru" in arms:
        plan += [(f"gru@{pol}", "gru", m) for pol, m in extra_manifests.items()]

    results, oof_arrays, per_fold = {}, {}, {}
    for name, arm, arm_manifest in plan:
        arm_started = time.time()
        kwargs = dict(seed=seed, quick=quick, epochs=epochs, patience=patience, cache_dir=cache_dir)
        if jobs == 1:
            runs = [run_fold(arm, arm_manifest, fold, windows, on_fit=on_fit, **kwargs) for fold in folds]
            torch.set_num_threads(saved_threads)
        else:
            runs = Parallel(n_jobs=jobs)(delayed(run_fold)(arm, arm_manifest, fold, windows, **kwargs)
                                         for fold in folds)
        summary, oof = summarise(runs, y_all, repeats)
        results[name] = summary
        oof_arrays[name] = oof
        per_fold[name] = summary["per_fold"]
        po = summary["pooled_oof"]
        print(f"{name:20s} macro-F1 {po['macro_f1']['mean']:.3f} +/- {po['macro_f1']['sd']:.3f}  "
              f"bal-acc {po['balanced_accuracy']['mean']:.3f}  kappa {po['kappa']['mean']:.3f}  "
              f"({time.time() - arm_started:.0f}s)", flush=True)

    created = datetime.now(UTC)
    sha = git_sha()
    run_id = f"{created.strftime('%Y%m%dT%H%M%SZ')}-{(sha or 'nogit')[:8]}-s{seed}"
    result = {
        "run_id": run_id,
        "created_at": created.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "git_sha": sha,
        "labels_sha256": manifest["labels_sha256"],
        "ethogram_version": manifest.get("ethogram_version"),
        "labels": list(LABELS),
        "n_clips": len(clips),
        "config": {"repeats": repeats, "n_splits": n_splits, "n_inner": n_inner, "seed": seed, "jobs": jobs,
                   "quick": quick, "windows": windows, "epochs_cap": epochs, "patience": patience,
                   "aug_policy": manifest.get("aug_policy"), "backbone": manifest.get("backbone")},
        "headline": "pooled_oof, mean over repeats. Every number here is out-of-fold: "
                    "no outer eval clip was used for stopping or selection.",
        "arms": results,
    }
    a, b = PRIMARY_COMPARISON
    if a in results and b in results:
        result["paired"] = paired_comparison(per_fold, a, b, len(clips), n_splits)
    if extra_manifests:
        result["aug_policy_arms"] = {name: (policies[0] if name == arm else name.split("@", 1)[1])
                                     for name, arm, _ in plan if arm == "gru"}
        result["paired_aug_policy"] = {
            pol: paired_comparison(per_fold, f"gru@{pol}", "gru", len(clips), n_splits)
            for pol in extra_manifests if f"gru@{pol}" in results
        }
        result["aug_policy_status"] = ("UNEVALUATED until a crossval run on the recoded v2 labels; "
                                       "see training/augment.py")
    result["runtime_seconds"] = round(time.time() - started, 1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cv_result.json").write_text(json.dumps(result, indent=2))
    np.savez(out_dir / "cv_oof.npz", y=y_all, filenames=np.array(filenames),
             **{f"{arm}_P": oof for arm, oof in oof_arrays.items()})
    (out_dir / "runs").mkdir(exist_ok=True)
    with (out_dir / "runs" / "index.jsonl").open("a") as f:
        for arm, summary in results.items():
            f.write(json.dumps({
                "run_id": run_id, "created_at": result["created_at"], "git_sha": sha, "seed": seed,
                "labels_sha256": manifest["labels_sha256"],
                "ethogram_version": manifest.get("ethogram_version"), "arm": arm,
                "aug_policy": result.get("aug_policy_arms", {}).get(arm, policies[0]),
                "config": result["config"],
                "metrics": {k: v["mean"] for k, v in summary["pooled_oof"].items()},
            }) + "\n")
    for pol, pc in result.get("paired_aug_policy", {}).items():
        print(f"\ngru@{pol} - gru ({policies[0]}): mean macro-F1 difference {pc['mean_difference']:+.3f}, "
              f"corrected resampled t p={pc['corrected_resampled_t']['p_two_sided']:.3f}")
    if "paired" in result:
        p = result["paired"]
        print(f"\n{a} - {b}: mean macro-F1 difference {p['mean_difference']:+.3f} over "
              f"{len(p['differences'])} folds, corrected resampled t p={p['corrected_resampled_t']['p_two_sided']:.3f}")
    print(f"\nWrote {out_dir / 'cv_result.json'}, cv_oof.npz, runs/index.jsonl ({result['runtime_seconds']}s)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-inner", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                        help="one config per arm, 15-epoch cap: for tests and smoke runs, not results")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="outer folds fitted in parallel (results do not depend on it)")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--aug-policies", nargs="+", default=None, metavar="POLICY",
                        help="first is primary (all arms); each further policy adds a paired "
                             "gru@<policy> arm. Build each with extract_frames --aug-policy.")
    args = parser.parse_args()
    crossval(cache_dir=args.cache_dir, out_dir=args.out_dir, repeats=args.repeats,
             n_splits=args.n_splits, n_inner=args.n_inner, seed=args.seed, quick=args.quick,
             arms=args.arms, jobs=args.jobs, aug_policies=args.aug_policies)


if __name__ == "__main__":
    main()
