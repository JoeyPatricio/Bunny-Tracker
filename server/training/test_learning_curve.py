"""Hermetic grouped subset invariants and a measured sklearn learning curve."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.labels import LABELS
from tests.checks import check, finish
from training.learning_curve import run, training_subsets
from training.pose_features import validate_oof
from training.splits import group_key
from training.test_pose_features import CFG, reference, rejects


def test_subsets(tmp):
    manifest, _, arrays, folds = reference(tmp, with_cache=True)
    names, labels = arrays["filenames"].tolist(), [c["label"] for c in manifest["clips"]]
    plans = training_subsets(folds, names, labels, fractions=[.5, 1], seed=CFG["seed"], n_inner=2)
    once = training_subsets(folds, names, labels, fractions=[.5, 1], seed=CFG["seed"], n_inner=2)
    small, full = plans[.5], plans[1.]
    check("group subsets are deterministic", all(np.array_equal(a.train, b.train) for a, b in zip(small, once[.5], strict=True)))
    check("training sizes are nested and full size restores reference outer train",
          all(set(a.train) < set(b.train) and np.array_equal(b.train, original.train)
              for a, b, original in zip(small, full, folds, strict=True)))
    check("every training size uses the original outer eval clips",
          all(np.array_equal(sub.eval, original.eval) for plan in plans.values()
              for sub, original in zip(plan, folds, strict=True)))
    groups = np.array([group_key(name) for name in names])
    whole, inner = True, True
    for sub in small:
        selected_groups = set(groups[sub.train])
        original = next(f for f in folds if (f.repeat, f.outer) == (sub.repeat, sub.outer))
        whole &= set(sub.train) == {i for i in original.train if groups[i] in selected_groups}
        for a, b in sub.inner:
            inner &= (not set(groups[a]) & set(groups[b]) and set(a) | set(b) == set(sub.train)
                      and not (set(a) | set(b)) & set(sub.eval)
                      and set(labels[i] for i in b) == set(LABELS))
    check("no source group is partially sampled", whole)
    check("selection is nested inside sampled outer training groups", inner)
    rejects("small unsupported subsets fail with remedy", lambda: training_subsets(folds, names, labels, fractions=[.01, 1], seed=3, n_inner=2), "Clipping Mode")
    rejects("identical group counts rejected", lambda: training_subsets(folds, names, labels, fractions=[.99, 1], seed=3, n_inner=2), "identical numbers")
    rejects("out of range fractions fail", lambda: training_subsets(folds, names, labels, fractions=[0, 1], seed=3, n_inner=2), "(0, 1]")
    return manifest, arrays, plans


def test_actual_fits(tmp, arrays, plans):
    spec = run(cache_dir=tmp, manifest_path=tmp / "manifest.json",
               reference_path=tmp / "cv_result.json", oof_path=tmp / "cv_oof.npz",
               out_dir=tmp / "learning", fractions=[.5, 1], arms=["static_logreg", "majority"])
    complete = True
    for point, fraction in zip(spec["points"], [.5, 1], strict=True):
        result = json.loads((tmp / "learning" / point["result"]).read_text())
        with np.load(tmp / "learning" / point["oof"]) as archive:
            complete &= np.array_equal(archive["filenames"], arrays["filenames"])
            for arm in result["arms"]:
                validate_oof(result, dict(archive), arm=arm)
        counts = result["config"]["learning_curve"]["n_train_per_fold"]
        complete &= counts == [len(f.train) for f in plans[fraction]]
        interval = result["arms"]["static_logreg"]["pooled_oof"]["macro_f1"]["group_bootstrap"]
        complete &= interval["n_groups"] == 12 and len(interval["ci95"]) == 2
        # The single-frame synthetic embedding has one class-indicator channel.
        complete &= result["arms"]["static_logreg"]["pooled_oof"]["macro_f1"]["mean"] > .9
    check("real sklearn fits produce complete OOF and recorded training counts", complete)
    check("producer indexes measured result and OOF files", (tmp / "learning" / "learning_curve.json").is_file())
    from training.figures import generate

    generate(models_dir=tmp, out_dir=tmp / "actual-curve-figure", kinds=["learning_curve"])
    check("measured learning reports feed standalone figure export", (tmp / "actual-curve-figure" / "learning_curve.pdf").is_file())


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="bunny-learning-test-") as directory:
        tmp = Path(directory)
        _, arrays, plans = test_subsets(tmp)
        test_actual_fits(tmp, arrays, plans)
    finish("learning curve")
