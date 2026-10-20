"""Hermetic geometry, leakage and paired-OOF guards for the offline pose arm."""
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.labels import ETHOGRAM_VERSION, LABELS
from tests.checks import check, finish
from training import pose_features as P
from training.metrics import classification_report, confusion_matrix
from training.splits import make_folds

CFG = {"repeats": 2, "n_splits": 3, "n_inner": 2, "seed": 3,
       "backbone": "synthetic", "aug_policy": "legacy", "quick": True,
       "epochs_cap": 2, "patience": 1, "windows": [4]}


def track(label=0, group=0):
    times = np.array([0., .13, .4, .9, 1.5, 2.])
    skeleton = np.array([[3, 1], [2, 1], [0, 1], [2.2, -.7 - label * .15],
                         [2.8, -.8 - label * .1], [-.5, 1.2]])
    points = skeleton[None] + np.stack([times * (label + 1) * .3, times * 0], axis=1)[:, None]
    points += group * .03
    return points, times


def reference(tmp, *, with_cache=False):
    clips, tracks = [], []
    rng = np.random.default_rng(1)
    emb = tmp / "emb"
    if with_cache:
        emb.mkdir()
    for group in range(12):
        for label, name in enumerate(LABELS):
            filename = f"recording-pose-src{group}-{name}.webm"
            variant, ev = f"{filename}.v0.npy", f"{filename}.eval4.npy"
            clips.append({"filename": filename, "label": name, "group": f"pose-src{group}",
                          "variants": [variant], "eval": {"4": ev}})
            points, times = track(label, group)
            tracks.append({"filename": filename, "track_id": "rabbit-1", "keypoints": points.tolist(),
                           "timestamps": times.tolist()})
            if with_cache:
                x = rng.normal(0, .1, (4, 1280)).astype(np.float32)
                x[:, label::len(LABELS)] += 2
                np.save(emb / variant, x)
                np.save(emb / ev, x)
    manifest = {"schema": 2, "labels": list(LABELS), "labels_sha256": "ab" * 32,
                "ethogram_version": ETHOGRAM_VERSION, "clips": clips,
                "backbone": "synthetic", "aug_policy": "legacy", "window": 4,
                "eval_windows": [4], "train_frames": 4}
    names = [clip["filename"] for clip in clips]
    y = np.array([LABELS.index(clip["label"]) for clip in clips])
    probabilities = np.full((CFG["repeats"], len(clips), len(LABELS)), .01)
    for r in range(CFG["repeats"]):
        pred = y.copy()
        pred[np.arange(len(y)) % (4 + r) == 0] = (pred[np.arange(len(y)) % (4 + r) == 0] + 1) % len(LABELS)
        probabilities[r, np.arange(len(y)), pred] = .94
    folds = make_folds(names, [clip["label"] for clip in clips],
                       **{k: CFG[k] for k in ("seed", "repeats", "n_splits", "n_inner")})
    reports = [classification_report(confusion_matrix(y, p.argmax(1))) for p in probabilities]
    pooled = {key: {"mean": float(np.mean([rep[key] for rep in reports])),
                    "sd": float(np.std([rep[key] for rep in reports], ddof=1)),
                    "per_repeat": [rep[key] for rep in reports]} for key in P.METRICS}
    per_fold = []
    for fold in folds:
        report = classification_report(confusion_matrix(y[fold.eval], probabilities[fold.repeat, fold.eval].argmax(1)))
        per_fold.append({"repeat": fold.repeat, "outer": fold.outer,
                         **{key: report[key] for key in ("n", "macro_f1", "balanced_accuracy", "kappa")}})
    result = {"run_id": "synthetic-reference", "git_sha": "synthetic", "config": CFG.copy(),
              "labels": list(LABELS), "labels_sha256": manifest["labels_sha256"],
              "ethogram_version": ETHOGRAM_VERSION, "n_clips": len(clips),
              "arms": {"static_logreg": {"pooled_oof": pooled, "per_fold": per_fold,
                                         "per_repeat_confusion": [rep["confusion_matrix"] for rep in reports]}}}
    annotations = {"schema": 1, "coordinate_system": "image_pixels", "node_names": list(P.NODES),
                   "source": {"kind": "manual", "description": "synthetic test fixture, not animal annotations"},
                   "clips": tracks}
    for filename, data in (("manifest.json", manifest), ("cv_result.json", result), ("keypoints.json", annotations)):
        (tmp / filename).write_text(json.dumps(data))
    arrays = {"filenames": np.array(names), "y": y, "static_logreg_P": probabilities}
    np.savez(tmp / "cv_oof.npz", **arrays)
    return manifest, result, arrays, folds


def rejects(name, fn, text):
    try:
        fn()
    except (ValueError, SystemExit) as err:
        check(name, text in str(err), str(err))
    else:
        check(name, False, "did not reject")


def test_geometry():
    points, times = track(2)
    features = P.geometric_features(points, times)
    check("40 finite named geometric features", len(P.FEATURE_NAMES) == len(features) == 40 and np.isfinite(features).all())
    check("translation and uniform image scale preserve normalized geometry and derivatives",
          np.allclose(features, P.geometric_features(points * 13 + [201, 315], times), atol=1e-10))
    check("nonuniform timestamps give physical speed per body length",
          np.isclose(features[P.FEATURE_NAMES.index("nose_speed_mean")], .45))
    check("constant velocity has zero acceleration",
          abs(features[P.FEATURE_NAMES.index("nose_acceleration_mean")]) < 1e-12)
    check("body axis uses bounded sin/cos rather than a wrapped angle",
          features[P.FEATURE_NAMES.index("body_axis_cos_mean")] == 1
          and features[P.FEATURE_NAMES.index("body_axis_sin_mean")] == 0)
    quadratic = points.copy()
    quadratic[:, :, 0] += times[:, None] ** 2
    actual = P.geometric_features(quadratic, times)[P.FEATURE_NAMES.index("nose_acceleration_mean")]
    check("quadratic translation has acceleration 2/body_length", np.isclose(actual, 1))
    bad = points.copy()
    bad[0, 0, 0] = np.nan
    rejects("missing keypoints require annotation review", lambda: P.geometric_features(bad, times), "Complete or review")
    rejects("duplicate timestamps fail", lambda: P.geometric_features(points, [0, 0, 1, 2, 3, 4]), "strictly increasing")
    bad = points.copy()
    bad[:, 1] = bad[:, 2]
    rejects("zero body axis fails", lambda: P.geometric_features(bad, times), "neck and rump coincide")


def test_paired_run(tmp):
    manifest, ref, arrays, folds = reference(tmp)
    x, _ = P.load_tracks(tmp / "keypoints.json", manifest)
    fitted = []

    def on_fit(fold, tr, ev, model):
        fitted.append((fold, tr, ev))
        if not np.allclose(model[0].mean_, x[tr].mean(0)):
            raise AssertionError("scaler saw held-out rows")
        if set(tr) & set(ev) or set(tr) & set(fold.eval):
            raise AssertionError("evaluation rows entered training")

    # Outer fitting has eval=fold.eval, inner fitting has disjoint eval.
    args = dict(keypoints=tmp / "keypoints.json", manifest_path=tmp / "manifest.json",
                reference_path=tmp / "cv_result.json", oof_path=tmp / "cv_oof.npz")
    result = P.run(**args, out_dir=tmp / "out", on_fit=on_fit)
    check("every inner and outer scaler uses only its training clips", len(fitted) == len(folds) * 7)
    with np.load(tmp / "out" / "pose_oof.npz") as archive:
        p = archive["pose_logreg_P"].copy()
        check("complete normalized paired pose OOF", p.shape == arrays["static_logreg_P"].shape
              and np.isfinite(p).all() and np.allclose(p.sum(2), 1))
        P.validate_oof(result, dict(archive), arm="pose_logreg")
    actual = []
    for fold in folds:
        a = classification_report(confusion_matrix(arrays["y"][fold.eval], p[fold.repeat, fold.eval].argmax(1)))
        b = classification_report(confusion_matrix(arrays["y"][fold.eval], arrays["static_logreg_P"][fold.repeat, fold.eval].argmax(1)))
        actual.append(a["macro_f1"] - b["macro_f1"])
    check("paired differences derive from identical held-out clips", np.allclose(actual, result["paired"]["differences"]))
    check("explicit nested membership and input hashes retained", result["folds"] == P.fold_membership(folds, arrays["filenames"].tolist())
          and result["inputs"]["keypoints"]["sha256"] == P.sha256(tmp / "keypoints.json"))
    P.run(**args, out_dir=tmp / "again")
    with np.load(tmp / "again" / "pose_oof.npz") as archive:
        check("pose probabilities reproduce exactly", np.array_equal(p, archive["pose_logreg_P"]))
    wrong = copy.deepcopy(ref)
    wrong["arms"]["static_logreg"]["per_fold"][0]["macro_f1"] = .123
    rejects("reference fold mismatch fails before pose fitting", lambda: P.reference_folds(manifest, wrong, arrays, "static_logreg"), "regenerated folds")
    wrong = dict(arrays, y=arrays["y"][::-1])
    rejects("wrong reference OOF rejected", lambda: P.reference_folds(manifest, ref, wrong, "static_logreg"), "disagrees")
    annotations = json.loads((tmp / "keypoints.json").read_text())
    annotations["clips"].pop()
    (tmp / "missing.json").write_text(json.dumps(annotations))
    rejects("pose pool cannot silently drop an unannotated clip", lambda: P.load_tracks(tmp / "missing.json", manifest), "Missing:")
    annotations["source"] = {"kind": "external_model", "description": "test"}
    (tmp / "leaky.json").write_text(json.dumps(annotations))
    rejects("pool-trained keypoint model cannot masquerade as independent", lambda: P.load_tracks(tmp / "leaky.json", manifest), "cv_pool_excluded")
    check("pose imports no torch or pose framework", "torch" not in sys.modules and "sleap" not in sys.modules)


if __name__ == "__main__":
    test_geometry()
    with tempfile.TemporaryDirectory(prefix="bunny-pose-test-") as directory:
        test_paired_run(Path(directory))
    finish("offline pose")
