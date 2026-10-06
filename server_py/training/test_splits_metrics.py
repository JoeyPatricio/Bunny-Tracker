"""Regression gate for training/splits.py and training/metrics.py (docs/upgrade-plan.md, Phase 3).

The fold checks are the leakage guarantees the whole evaluation rests on: a
clip evaluated in a fold it also trained in, or a source group on both sides
of a split, would inflate every number cv_result.json reports and nothing
downstream could tell. The metric checks pin the numpy implementations to
sklearn, since the runtime venv that imports check_degenerate cannot have
sklearn.

Hermetic: synthetic filenames only, nothing read from server/.

Usage (from server_py/, training venv):
    .venv-train/bin/python training/test_splits_metrics.py
"""
import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn import metrics as skm

from tests.checks import check, finish
from training import metrics as M
from training.splits import group_key, make_folds

CLASSES = ["a", "b", "c"]


def synthetic_pool(per_class: dict[str, int], groups_per_class: int, seed: int = 0):
    """Filenames whose group_key is the date-hour, spread across `groups_per_class`
    hours per class, mixing classes within an hour the way real sessions do."""
    rng = np.random.default_rng(seed)
    filenames, labels = [], []
    for ci, (label, n) in enumerate(per_class.items()):
        for i in range(n):
            hour = (ci * 3 + i % groups_per_class) % 24
            day = 1 + (ci * 3 + i % groups_per_class) // 24
            filenames.append(f"recording-clip-2026-09-{day:02d}T{hour:02d}-{rng.integers(60):02d}-00-{ci}{i:03d}Z.webm")
            labels.append(label)
    return filenames, labels


def raises_naming(fn, *needles) -> tuple[bool, str]:
    try:
        fn()
    except SystemExit as err:
        msg = str(err)
        return all(n in msg for n in needles), msg
    return False, "did not raise"


def test_outer_folds_partition_the_pool() -> None:
    """Every clip lands in exactly one outer eval fold per repeat, never in its own training set."""
    fns, labels = synthetic_pool({"a": 30, "b": 20, "c": 15}, groups_per_class=8)
    folds = make_folds(fns, labels, n_splits=5, n_inner=3, repeats=3, seed=7, classes=CLASSES)
    check("3 repeats x 5 outer folds", len(folds) == 15, f"got {len(folds)}")
    for r in range(3):
        rep = [f for f in folds if f.repeat == r]
        evals = np.concatenate([f.eval for f in rep])
        check(f"repeat {r}: every clip evaluated exactly once",
              sorted(evals.tolist()) == list(range(len(fns))))
        check(f"repeat {r}: no clip trains in the fold that evaluates it",
              all(not set(f.train.tolist()) & set(f.eval.tolist()) for f in rep))
        check(f"repeat {r}: train and eval together cover the pool",
              all(len(f.train) + len(f.eval) == len(fns) for f in rep))


def test_groups_never_straddle() -> None:
    """No source group appears on both sides of any outer or inner split."""
    fns, labels = synthetic_pool({"a": 30, "b": 20, "c": 15}, groups_per_class=8)
    groups = np.array([group_key(f) for f in fns])
    folds = make_folds(fns, labels, n_splits=5, n_inner=3, repeats=2, seed=1, classes=CLASSES)
    outer_ok = all(not set(groups[f.train]) & set(groups[f.eval]) for f in folds)
    inner_ok = all(not set(groups[a]) & set(groups[b]) for f in folds for a, b in f.inner)
    check("no group on both sides of an outer split", outer_ok)
    check("no group on both sides of an inner split", inner_ok)


def test_inner_folds_stay_inside_outer_train() -> None:
    """Inner folds draw only on outer-train clips, so selection never sees outer-eval clips."""
    fns, labels = synthetic_pool({"a": 30, "b": 20, "c": 15}, groups_per_class=8)
    folds = make_folds(fns, labels, n_splits=5, n_inner=3, repeats=2, seed=2, classes=CLASSES)
    subset = all(set(a.tolist()) | set(b.tolist()) <= set(f.train.tolist())
                 for f in folds for a, b in f.inner)
    disjoint_eval = all(not (set(a.tolist()) | set(b.tolist())) & set(f.eval.tolist())
                        for f in folds for a, b in f.inner)
    partition = all(sorted(np.concatenate([b for _, b in f.inner]).tolist()) == sorted(f.train.tolist())
                    for f in folds)
    check("inner clips are a subset of outer-train", subset)
    check("no outer-eval clip appears in any inner fold", disjoint_eval)
    check("inner val folds partition outer-train", partition)
    check("every outer fold has n_inner inner folds", all(len(f.inner) == 3 for f in folds))


def test_deterministic_given_seed() -> None:
    """Same seed, same folds; different seed or repeat, different folds."""
    fns, labels = synthetic_pool({"a": 30, "b": 20, "c": 15}, groups_per_class=8)
    kw = dict(n_splits=5, n_inner=3, repeats=2, classes=CLASSES)
    one = make_folds(fns, labels, seed=3, **kw)
    two = make_folds(fns, labels, seed=3, **kw)
    other = make_folds(fns, labels, seed=4, **kw)
    same = all(np.array_equal(x.eval, y.eval) and all(np.array_equal(a, c) for (a, _), (c, _) in zip(x.inner, y.inner, strict=True))
               for x, y in zip(one, two, strict=True))
    check("same seed reproduces every outer and inner fold", same)
    check("a different seed changes the folds",
          any(not np.array_equal(x.eval, y.eval) for x, y in zip(one, other, strict=True)))
    check("repeats differ from each other",
          any(not np.array_equal(one[k].eval, one[5 + k].eval) for k in range(5)))


def test_thin_class_fails_loudly() -> None:
    """A class with fewer clips than folds raises an actionable error naming it."""
    fns, labels = synthetic_pool({"a": 30, "b": 20, "c": 3}, groups_per_class=8)
    ok, msg = raises_naming(lambda: make_folds(fns, labels, n_splits=5, classes=CLASSES),
                            "'c'", "3 clip", "Clipping Mode", "merge")
    check("3-clip class raises naming the class, its count and the remedy", ok, msg)


def test_single_group_class_fails_loudly() -> None:
    """A class whose clips all come from one source group raises, even with plenty of clips."""
    fns, labels = synthetic_pool({"a": 30, "b": 20, "c": 12}, groups_per_class=8)
    fns = [f if lab != "c" else f"recording-clip-2026-09-20T05-00-00-{i:03d}Z.webm"
           for i, (f, lab) in enumerate(zip(fns, labels, strict=True))]
    ok, msg = raises_naming(lambda: make_folds(fns, labels, n_splits=5, classes=CLASSES),
                            "'c'", "1 source group", "Clipping Mode")
    check("single-group class raises naming the class and the group count", ok, msg)

    ok, msg = raises_naming(lambda: make_folds(fns, labels, n_splits=5, classes=CLASSES + ["d"]),
                            "'c'")
    check("a vocabulary class is checked even when it is not the last one", ok, msg)


def test_absent_vocabulary_class_fails_loudly() -> None:
    """A class in the vocabulary with no clips at all is a zero-support fold too."""
    fns, labels = synthetic_pool({"a": 30, "b": 20, "c": 15}, groups_per_class=8)
    ok, msg = raises_naming(lambda: make_folds(fns, labels, n_splits=5, classes=CLASSES + ["d"]),
                            "'d'", "0 clip")
    check("missing vocabulary class raises", ok, msg)


def test_metrics_match_sklearn() -> None:
    """numpy macro-F1, balanced accuracy and kappa match sklearn on random predictions."""
    rng = np.random.default_rng(0)
    worst = {"macro_f1": 0.0, "balanced_accuracy": 0.0, "kappa": 0.0, "precision": 0.0, "recall": 0.0}
    for _ in range(200):
        n_classes = int(rng.integers(2, 8))
        n = int(rng.integers(5, 120))
        y = rng.integers(0, n_classes, n)
        # Mix of skilled and random predictions, so agreement varies widely.
        p = np.where(rng.random(n) < rng.random(), y, rng.integers(0, n_classes, n))
        cm = M.confusion_matrix(y, p, n_classes)
        # sklearn warns when a random y_pred names a class y_true lacks; that
        # case is exactly the one being compared, so the warning is noise here.
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        worst["macro_f1"] = max(worst["macro_f1"], abs(M.macro_f1(cm) - skm.f1_score(y, p, average="macro", zero_division=0)))
        worst["balanced_accuracy"] = max(worst["balanced_accuracy"], abs(M.balanced_accuracy(cm) - skm.balanced_accuracy_score(y, p)))
        with np.errstate(all="ignore"):
            sk_kappa = skm.cohen_kappa_score(y, p)
        if not np.isnan(sk_kappa):
            worst["kappa"] = max(worst["kappa"], abs(M.cohen_kappa(cm) - sk_kappa))
        labels = [str(k) for k in range(n_classes)]
        rep = M.per_class_report(cm, labels)
        sk_p = skm.precision_score(y, p, labels=range(n_classes), average=None, zero_division=0)
        sk_r = skm.recall_score(y, p, labels=range(n_classes), average=None, zero_division=0)
        worst["precision"] = max(worst["precision"], max(abs(rep[lab]["precision"] - sk_p[k]) for k, lab in enumerate(labels)))
        worst["recall"] = max(worst["recall"], max(abs(rep[lab]["recall"] - sk_r[k]) for k, lab in enumerate(labels)))
    for name, err in worst.items():
        check(f"{name} matches sklearn over 200 random cases", err < 1e-12, f"max abs error {err:g}")


def test_majority_baseline() -> None:
    """Always guessing the majority class gives balanced accuracy exactly 1/n_classes and kappa 0."""
    y = np.array([0] * 50 + [1] * 20 + [2] * 10 + [3] * 5 + [4] * 5 + [5] * 4 + [6] * 6)
    cm = M.confusion_matrix(y, np.zeros_like(y), 7)
    check("majority balanced accuracy is exactly 1/7", M.balanced_accuracy(cm) == 1 / 7, f"{M.balanced_accuracy(cm)!r}")
    check("majority kappa is 0", M.cohen_kappa(cm) == 0.0, f"{M.cohen_kappa(cm)!r}")


def test_wilson_interval() -> None:
    """Wilson 95% intervals match published values and stay inside [0, 1]."""
    lo, hi = M.wilson_interval(8, 10)
    check("8/10 -> (0.4902, 0.9433)", abs(lo - 0.4902) < 1e-4 and abs(hi - 0.9433) < 1e-4, f"({lo:.4f}, {hi:.4f})")
    lo, hi = M.wilson_interval(0, 5)
    check("0/5 -> (0, 0.4345)", lo == 0.0 and abs(hi - 0.4345) < 1e-4, f"({lo:.4f}, {hi:.4f})")
    lo, hi = M.wilson_interval(5, 5)
    check("5/5 -> (0.5655, 1)", abs(lo - 0.5655) < 1e-4 and hi == 1.0, f"({lo:.4f}, {hi:.4f})")
    check("n = 0 carries no information", M.wilson_interval(0, 0) == (0.0, 1.0))


def test_classification_report_and_pooling() -> None:
    """classification_report carries the headline numbers; pool_oof refuses a non-partition."""
    y = np.array([0, 0, 1, 1, 2, 2])
    cm = M.confusion_matrix(y, np.array([0, 1, 1, 1, 2, 0]), 3)
    rep = M.classification_report(cm, ["a", "b", "c"])
    check("report carries n, macro-F1, balanced accuracy, kappa",
          rep["n"] == 6 and {"macro_f1", "balanced_accuracy", "kappa", "accuracy"} <= set(rep))
    check("per-class recall interval present", rep["per_class"]["a"]["recall_ci95"] == list(M.wilson_interval(1, 2)))

    oof = M.pool_oof(4, [([0, 2], [5, 6]), ([1, 3], [7, 8])])
    check("pool_oof places each prediction at its clip", oof.tolist() == [5, 7, 6, 8])
    for name, parts in (("a clip predicted twice", [([0, 1], [1, 1]), ([1, 2, 3], [1, 1, 1])]),
                        ("a clip never predicted", [([0, 1], [1, 1]), ([2], [1])])):
        try:
            M.pool_oof(4, parts)
            check(f"pool_oof rejects {name}", False, "did not raise")
        except ValueError:
            check(f"pool_oof rejects {name}", True)


def test_check_degenerate_preserved() -> None:
    """check_degenerate still flags a constant classifier and passes a varied one."""
    y = np.array([0, 1, 2, 3, 0, 1])
    constant = np.zeros((6, 7))
    constant[:, 2] = 1.0
    varied = np.eye(7)[y]
    with contextlib.redirect_stdout(io.StringIO()) as out:
        flagged = M.check_degenerate(constant, y)
        passed = M.check_degenerate(varied, y)
    check("constant classifier is degenerate", flagged is True)
    check("its message names the constant class", "DEGENERATE" in out.getvalue())
    check("a varied classifier is not", passed is False)


def main() -> None:
    for fn in (
        test_outer_folds_partition_the_pool,
        test_groups_never_straddle,
        test_inner_folds_stay_inside_outer_train,
        test_deterministic_given_seed,
        test_thin_class_fails_loudly,
        test_single_group_class_fails_loudly,
        test_absent_vocabulary_class_fails_loudly,
        test_metrics_match_sklearn,
        test_majority_baseline,
        test_wilson_interval,
        test_classification_report_and_pooling,
        test_check_degenerate_preserved,
    ):
        print(f"\n{fn.__doc__.splitlines()[0]}")
        fn()
    finish("splits and metrics")


if __name__ == "__main__":
    main()
