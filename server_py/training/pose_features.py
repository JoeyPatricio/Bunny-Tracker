"""Offline geometric pose arm, paired with an existing nested grouped CV run.

Consumes explicit, complete single-animal tracks. No pose estimator, torch,
video decoder or runtime integration is imported.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sklearn
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.labels import ETHOGRAM_VERSION, LABELS
from training.metrics import classification_report, confusion_matrix, grouped_macro_f1_interval
from training.splits import group_key, make_folds

NODES = ("nose", "neck", "rump", "left_ear_tip", "right_ear_tip", "tail_base")
DISTANCES = (("nose", "rump"), ("nose", "neck"), ("left_ear_tip", "right_ear_tip"),
             ("neck", "left_ear_tip"), ("neck", "right_ear_tip"), ("rump", "tail_base"),
             ("nose", "left_ear_tip"), ("nose", "right_ear_tip"))
FEATURE_NAMES = tuple(
    [f"distance_{a}_{b}_{s}" for a, b in DISTANCES for s in ("mean", "std")]
    + [f"body_axis_{axis}_{s}" for axis in ("sin", "cos") for s in ("mean", "std")]
    + [f"{ear}_axis_cos_{s}" for ear in ("left_ear_tip", "right_ear_tip") for s in ("mean", "std")]
    + [f"{node}_speed_mean" for node in NODES]
    + [f"{node}_acceleration_mean" for node in NODES]
    + ["bbox_area_mean", "bbox_area_std", "bbox_area_abs_rate_mean", "body_length_cv"]
)
CS = (0.01, 0.1, 1.0)
METRICS = ("macro_f1", "balanced_accuracy", "kappa", "accuracy")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometric_features(points, timestamps) -> np.ndarray:
    p, t = np.asarray(points, dtype=float), np.asarray(timestamps, dtype=float)
    if p.ndim != 3 or p.shape[1:] != (len(NODES), 2) or len(p) < 3:
        raise ValueError(f"keypoints must be [T, {len(NODES)}, 2] with T >= 3, in NODES order")
    if t.shape != (len(p),) or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError("timestamps must be finite, strictly increasing seconds, one per frame")
    if not np.isfinite(p).all():
        raise ValueError("keypoints contain missing/nonfinite coordinates. Complete or review the track first")
    axis = p[:, 1] - p[:, 2]
    length = np.linalg.norm(axis, axis=1)
    if (length <= 1e-8).any():
        raise ValueError("neck and rump coincide. Correct the body-axis annotations")
    scale = np.median(length)
    values = []

    def moments(x):
        values.extend([float(x.mean()), float(x.std())])

    for a, b in DISTANCES:
        moments(np.linalg.norm(p[:, NODES.index(a)] - p[:, NODES.index(b)], axis=1) / scale)
    moments(axis[:, 1] / length)
    moments(axis[:, 0] / length)
    for ear in (3, 4):
        vector = p[:, ear] - p[:, 1]
        norm = np.linalg.norm(vector, axis=1)
        if (norm <= 1e-8).any():
            raise ValueError(f"{NODES[ear]} coincides with neck. Correct the ear annotations")
        moments(np.sum(vector * axis, axis=1) / (norm * length))
    velocity = np.gradient(p, t, axis=0, edge_order=2) / scale
    acceleration = np.gradient(velocity, t, axis=0, edge_order=2)
    values.extend(np.linalg.norm(velocity, axis=2).mean(axis=0).tolist())
    values.extend(np.linalg.norm(acceleration, axis=2).mean(axis=0).tolist())
    area = np.prod(np.ptp(p, axis=1), axis=1) / scale**2
    moments(area)
    values.extend([float(np.abs(np.gradient(area, t, edge_order=2)).mean()),
                   float(length.std() / length.mean())])
    result = np.asarray(values)
    if not np.isfinite(result).all():
        raise ValueError("pose features overflowed. Check coordinate units and timestamp spacing")
    return result


def load_tracks(path: Path, manifest: dict) -> tuple[np.ndarray, dict]:
    data = json.loads(path.read_text())
    if data.get("schema") != 1 or data.get("node_names") != list(NODES):
        raise ValueError(f"pose schema must be 1 and node_names must be {list(NODES)}")
    if data.get("coordinate_system") != "image_pixels":
        raise ValueError("coordinate_system must be image_pixels with x right and y down")
    source = data.get("source", {})
    if source.get("kind") not in ("manual", "external_model") or not source.get("description"):
        raise ValueError("source must name kind manual or external_model and describe annotation/model provenance")
    if source["kind"] == "external_model" and (source.get("cv_pool_excluded") is not True
                                               or not source.get("training_data")):
        raise ValueError("external_model source needs training_data and cv_pool_excluded=true. "
                         "Pose fitting on this CV pool requires a separate nested pose-training experiment")
    tracks = {}
    for clip in data.get("clips", []):
        name = clip["filename"]
        if name in tracks:
            raise ValueError(f"duplicate pose clip {name}")
        if not isinstance(clip.get("track_id"), str) or not clip["track_id"].strip():
            raise ValueError(f"{name}: specify a stable single-animal track_id and review identity switches")
        try:
            tracks[name] = geometric_features(clip["keypoints"], clip["timestamps"])
        except ValueError as err:
            raise ValueError(f"{name}: {err}") from err
    names = [c["filename"] for c in manifest["clips"]]
    missing, extra = set(names) - tracks.keys(), tracks.keys() - set(names)
    if missing or extra:
        raise ValueError(f"pose clips must match the complete reference pool. "
                         f"Missing: {sorted(missing)[:5]}, extra: {sorted(extra)[:5]}. "
                         "Annotate the missing clips rather than silently changing the CV pool")
    return np.stack([tracks[name] for name in names]), source


def validate_oof(data: dict, arrays: dict, *, arm: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    labels = data["labels"]
    names, y = arrays["filenames"].tolist(), np.asarray(arrays["y"])
    if len(names) != len(set(names)) or len(names) != data["n_clips"]:
        raise ValueError("OOF filenames must be unique and match result n_clips")
    if y.shape != (len(names),) or y.dtype.kind not in "iu" or (y < 0).any() or (y >= len(labels)).any():
        raise ValueError("OOF y must contain one valid integer label ID per clip")
    p = np.asarray(arrays[f"{arm}_P"])
    expected = (data["config"]["repeats"], len(names), len(labels))
    if p.shape != expected or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError(f"{arm} OOF probabilities must be finite in [0,1] with shape {expected}")
    if not np.allclose(p.sum(axis=2), 1, atol=1e-6):
        raise ValueError(f"{arm} OOF probabilities must sum to one")
    summary = data["arms"][arm]
    reports = [classification_report(confusion_matrix(y, r.argmax(1), len(labels)), labels) for r in p]
    for key in METRICS:
        actual = np.array([r[key] for r in reports])
        saved = summary["pooled_oof"][key]
        sd = float(actual.std(ddof=1)) if len(actual) > 1 else 0.0
        if not (np.allclose(actual, saved["per_repeat"], atol=1e-8)
                and np.isclose(actual.mean(), saved["mean"], atol=1e-8)
                and np.isclose(sd, saved["sd"], atol=1e-8)):
            raise ValueError(f"{arm} {key} disagrees between result JSON and OOF. Use files from the same run")
    if not np.array_equal([r["confusion_matrix"] for r in reports], summary["per_repeat_confusion"]):
        raise ValueError(f"{arm} confusion matrices disagree between result JSON and OOF")
    return names, y, p


def reference_folds(manifest: dict, reference: dict, arrays: dict, arm: str):
    if manifest.get("schema") != 2 or manifest.get("labels") != LABELS or reference["labels"] != LABELS:
        raise ValueError("reference manifest must use cache schema 2 and the current model vocabulary")
    for key in ("labels_sha256", "ethogram_version"):
        if not manifest.get(key) or manifest[key] != reference.get(key):
            raise ValueError(f"reference manifest and CV result disagree on {key}. Rebuild on the same labels")
    if reference["ethogram_version"] != ETHOGRAM_VERSION:
        raise ValueError("reference uses an older ethogram. Recode and rerun CV first")
    names, y, p = validate_oof(reference, arrays, arm=arm)
    clips = manifest["clips"]
    if names != [c["filename"] for c in clips] or not np.array_equal(y, [LABELS.index(c["label"]) for c in clips]):
        raise ValueError("reference OOF filename order/labels differ from the manifest")
    config = reference["config"]
    folds = make_folds(names, [c["label"] for c in clips],
                       **{k: config[k] for k in ("seed", "repeats", "n_splits", "n_inner")})
    saved = reference["arms"][arm]["per_fold"]
    by_fold = {(s["repeat"], s["outer"]): s for s in saved}
    if len(saved) != len(folds) or len(by_fold) != len(folds):
        raise ValueError("reference per_fold entries must cover every repeat/outer fold exactly once")
    for fold in folds:
        metrics = classification_report(confusion_matrix(y[fold.eval], p[fold.repeat, fold.eval].argmax(1)))
        previous = by_fold.get((fold.repeat, fold.outer), {})
        if any(not np.isclose(metrics[k], previous.get(k, np.nan), atol=1e-8) for k in ("macro_f1", "balanced_accuracy", "kappa")):
            raise ValueError("regenerated folds disagree with reference fold metrics. "
                             "Use the original split configuration, manifest order and sklearn version")
    return folds, y, p


def fit_pose(x, y, train_idx, eval_idx, *, c: float, seed: int):
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=c, class_weight="balanced", max_iter=2000, random_state=seed))
    model.fit(x[train_idx], y[train_idx])
    probs = np.zeros((len(eval_idx), len(LABELS)))
    probs[:, model[-1].classes_] = model.predict_proba(x[eval_idx])
    return probs, model


def fold_membership(folds, filenames) -> list[dict]:
    return [{"repeat": f.repeat, "outer": f.outer,
             "train": [filenames[i] for i in f.train], "eval": [filenames[i] for i in f.eval],
             "inner": [{"train": [filenames[i] for i in a], "eval": [filenames[i] for i in b]}
                       for a, b in f.inner]} for f in folds]


def run(*, keypoints: Path, manifest_path: Path, reference_path: Path,
        oof_path: Path, out_dir: Path, comparator: str = "static_logreg", on_fit=None) -> dict:
    paths = {"keypoints": keypoints, "manifest": manifest_path,
             "reference_result": reference_path, "reference_oof": oof_path}
    provenance = {k: {"path": str(p.resolve()), "sha256": sha256(p)} for k, p in paths.items()}
    manifest, reference = json.loads(manifest_path.read_text()), json.loads(reference_path.read_text())
    with np.load(oof_path, allow_pickle=False) as archive:
        arrays = dict(archive)
    folds, y, baseline = reference_folds(manifest, reference, arrays, comparator)
    x, source = load_tracks(keypoints, manifest)
    names = arrays["filenames"].tolist()
    config = reference["config"]
    p = np.full_like(baseline, np.nan, dtype=float)
    selections, per_fold, differences = [], [], []
    with threadpool_limits(limits=1):
        for fold in folds:
            losses = []
            for c in CS:
                inner_loss = []
                for tr, ev in fold.inner:
                    probs, model = fit_pose(x, y, tr, ev, c=c, seed=config["seed"])
                    if on_fit:
                        on_fit(fold, tr, ev, model)
                    inner_loss.append(float(-np.log(np.clip(probs[np.arange(len(ev)), y[ev]], 1e-12, 1)).mean()))
                losses.append(float(np.mean(inner_loss)))
            chosen = int(np.argmin(losses))
            probs, model = fit_pose(x, y, fold.train, fold.eval, c=CS[chosen], seed=config["seed"])
            if on_fit:
                on_fit(fold, fold.train, fold.eval, model)
            p[fold.repeat, fold.eval] = probs
            metrics = classification_report(confusion_matrix(y[fold.eval], probs.argmax(1)))
            other = classification_report(confusion_matrix(y[fold.eval], baseline[fold.repeat, fold.eval].argmax(1)))
            differences.append(metrics["macro_f1"] - other["macro_f1"])
            per_fold.append({"repeat": fold.repeat, "outer": fold.outer, **metrics})
            selections.append({"repeat": fold.repeat, "outer": fold.outer, "C": CS[chosen],
                               "inner_log_loss": losses, "candidate_C": list(CS)})
    if not np.isfinite(p).all():
        raise ValueError("not every clip received an OOF pose prediction")
    reports = [classification_report(confusion_matrix(y, row.argmax(1))) for row in p]
    pooled = {}
    for metric in METRICS:
        values = np.array([rep[metric] for rep in reports])
        pooled[metric] = {"mean": float(values.mean()),
                          "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                          "per_repeat": values.tolist()}
    pooled["macro_f1"]["group_bootstrap"] = grouped_macro_f1_interval(
        y, p, [group_key(name) for name in names], seed=config["seed"])
    d = np.asarray(differences)
    # Correct correlated fold differences using the actual evaluation/training ratios.
    ratio = float(np.mean([len(f.eval) / len(f.train) for f in folds]))
    variance = float(d.var(ddof=1)) if len(d) > 1 else 0.0
    se = np.sqrt((1 / len(d) + ratio) * variance)
    test = {"standard_error": float(se), "df": len(d) - 1,
            "p_two_sided": float(2 * stats.t.sf(abs(d.mean() / se), len(d) - 1)) if se else (1.0 if d.mean() == 0 else None),
            "note": "zero-variance nonzero differences have no estimable p-value" if not se and d.mean() != 0 else None}
    created = datetime.now(UTC)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                         cwd=Path(__file__).resolve().parent)
    membership = fold_membership(folds, names)
    result = {
        "schema": 1, "run_id": f"pose-{created.strftime('%Y%m%dT%H%M%S%fZ')}",
        "created_at": created.isoformat(), "git_sha": git.stdout.strip(),
        "labels": list(LABELS), "labels_sha256": manifest["labels_sha256"],
        "ethogram_version": manifest["ethogram_version"], "n_clips": len(names),
        "config": {k: config[k] for k in ("seed", "repeats", "n_splits", "n_inner")},
        "feature_schema": {"version": 1, "names": list(FEATURE_NAMES), "body_scale": "clip median neck-rump distance"},
        "pose_source": source, "inputs": provenance,
        "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__},
        "implementation_sha256": sha256(Path(__file__)),
        "folds": membership,
        "folds_sha256": hashlib.sha256(json.dumps(membership, sort_keys=True).encode()).hexdigest(),
        "arms": {"pose_logreg": {"pooled_oof": pooled, "per_fold": per_fold, "selections": selections,
                                   "per_repeat_confusion": [rep["confusion_matrix"] for rep in reports]}},
        "paired": {"a": "pose_logreg", "b": comparator, "reference_run_id": reference["run_id"],
                   "metric": "macro_f1 per outer fold", "differences": differences,
                   "mean_difference": float(d.mean()), "corrected_resampled_t": test},
        "status": "offline only. Keypoints are an additional information source, not a temporal-head ablation",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "pose_oof.npz", filenames=np.array(names), y=y, pose_logreg_P=p)
    result["oof_sha256"] = sha256(out_dir / "pose_oof.npz")
    (out_dir / "pose_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keypoints", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("models/cache/manifest.json"))
    parser.add_argument("--reference", type=Path, default=Path("models/cv_result.json"))
    parser.add_argument("--oof", type=Path, default=Path("models/cv_oof.npz"))
    parser.add_argument("--comparator", default="static_logreg")
    parser.add_argument("--out-dir", type=Path, default=Path("models/pose"))
    args = parser.parse_args()
    try:
        result = run(keypoints=args.keypoints, manifest_path=args.manifest,
                     reference_path=args.reference, oof_path=args.oof,
                     out_dir=args.out_dir, comparator=args.comparator)
    except (OSError, ValueError, KeyError) as err:
        parser.exit(2, f"pose_features: {err}.\n")
    print(f"Wrote {args.out_dir / 'pose_result.json'} ({len(FEATURE_NAMES)} features, {result['n_clips']} clips)")


if __name__ == "__main__":
    main()
