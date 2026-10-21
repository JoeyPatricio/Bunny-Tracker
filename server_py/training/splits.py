"""Grouped, stratified, nested cross-validation folds.

Replaces the single train/val split as the evaluation design. With 112 clips a held-out test set would be ~3 clips per class, too
few to tell a real gain from noise. Repeated grouped k-fold instead gives every
clip exactly one out-of-fold prediction per repeat, so the pooled OOF confusion
matrix has the full dataset as its support.

Nesting: inside each outer fold, the outer-train clips are split again into
inner folds. Early stopping and every hyperparameter choice (window, learning
rate, augmentation policy) happen on inner folds only, so the outer eval clips
are never touched during selection.

group_key lives here as the single definition. dataset.py's split_by_group is
kept as legacy and imports it from this module.

Training venv only (sklearn). metrics.py is the numpy-only half.
"""
import re
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from shared.labels import LABELS


def group_key(filename: str) -> str:
    """Source-group key for a clip — line-for-line port of groupKey() in
    TrainingStudio.jsx. Clips cut from the same source video/session share
    scene, lighting, and rabbit; letting them straddle train/val would measure
    "have I seen this video" instead of generalization.
    """
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2})", filename)
    if m:
        return m.group(1)
    m = re.match(r"^recording-([a-z]+)-src(\d+)", filename, re.IGNORECASE)
    if m:
        return f"{m.group(1)}-src{m.group(2)}"
    m = re.match(r"^recording-([a-z]+)-([a-z]+)", filename, re.IGNORECASE)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return filename


@dataclass(frozen=True)
class Fold:
    """One outer fold of one repeat. All index arrays index the clip list
    passed to make_folds, so inner folds can be used without remapping."""
    repeat: int
    outer: int
    train: np.ndarray                                # outer-train clip indices
    eval: np.ndarray                                 # outer-eval clip indices
    inner: tuple[tuple[np.ndarray, np.ndarray], ...]  # (inner-train, inner-val) pairs, subsets of `train`


def _remedy(label: str) -> str:
    return (
        f"Harvest more '{label}' footage with Clipping Mode on different days, or "
        f"merge '{label}' into a neighbouring class and bump the ethogram version "
        f"(docs/ethogram.md). Do not lower n_splits to make this pass: the folds "
        f"would then be too large to give a stable estimate."
    )


def check_support(labels: list[str], groups: list[str], classes: list[str], n_splits: int,
                  where: str = "the CV pool") -> None:
    """Every class must span at least n_splits distinct groups, or some fold is
    guaranteed to evaluate it on zero clips. That fold would report a macro-F1
    silently computed without the class, which is the failure this exists to
    stop. Raises SystemExit naming the class, its counts, and the remedy."""
    for label in classes:
        clip_count = sum(1 for lab in labels if lab == label)
        group_count = len({g for lab, g in zip(labels, groups, strict=True) if lab == label})
        if clip_count < n_splits:
            raise SystemExit(
                f"Class '{label}' has {clip_count} clip(s) in {where}, fewer than "
                f"n_splits={n_splits}, so at least one fold would evaluate it on zero clips. "
                + _remedy(label)
            )
        if group_count < n_splits:
            raise SystemExit(
                f"Class '{label}' has {clip_count} clip(s) in {where} but they come from only "
                f"{group_count} source group(s) (see group_key), fewer than n_splits={n_splits}. "
                f"Clips from one group can never be split across folds, so at least one fold "
                f"would evaluate '{label}' on zero clips. " + _remedy(label)
            )


def _check_folds_have_support(y: np.ndarray, splits, classes: list[str], where: str) -> None:
    # StratifiedGroupKFold stratifies greedily and does not guarantee every
    # class lands in every fold even when the precondition above holds. Check
    # the folds it actually produced rather than trusting it.
    present = set(y[np.concatenate([e for _, e in splits])].tolist())
    for k, (_, eval_idx) in enumerate(splits):
        missing = present - set(y[eval_idx].tolist())
        if missing:
            names = sorted(classes[c] for c in missing)
            raise SystemExit(
                f"{where}, fold {k}: no eval clips for {names}. The grouping leaves too few "
                f"independent groups for these classes to reach every fold. " + _remedy(names[0])
            )


def make_folds(filenames: list[str], labels: list[str], *, n_splits: int = 5, n_inner: int = 4,
               repeats: int = 5, seed: int = 0, classes: list[str] | None = None) -> list[Fold]:
    """Repeated, nested, grouped, stratified folds. Deterministic given `seed`.

    Repeat r shuffles with a seed derived from (seed, r), so repeats differ
    from each other but a rerun with the same seed reproduces every fold
    exactly. `classes` defaults to the model vocabulary, and every class in it
    must be present with enough groups: a vocabulary class with no clips is a
    zero-support fold too.
    """
    if len(filenames) != len(labels):
        raise ValueError(f"{len(filenames)} filenames but {len(labels)} labels")
    classes = list(LABELS if classes is None else classes)
    unknown = sorted(set(labels) - set(classes))
    if unknown:
        raise ValueError(f"labels {unknown} are not in the class list {classes}")

    groups = [group_key(f) for f in filenames]
    check_support(labels, groups, classes, n_splits)

    y = np.array([classes.index(lab) for lab in labels])
    g = np.array(groups)
    X = np.zeros((len(y), 1))  # StratifiedGroupKFold ignores features
    folds: list[Fold] = []
    for r in range(repeats):
        outer_seed = int(np.random.SeedSequence([seed, r]).generate_state(1)[0])
        outer = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=outer_seed)
                     .split(X, y, g))
        _check_folds_have_support(y, outer, classes, f"repeat {r}, outer")
        for k, (train_idx, eval_idx) in enumerate(outer):
            check_support([labels[i] for i in train_idx], [groups[i] for i in train_idx],
                          classes, n_inner, where=f"the outer-train set of repeat {r} fold {k}")
            inner_seed = int(np.random.SeedSequence([seed, r, k]).generate_state(1)[0])
            inner_rel = list(StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=inner_seed)
                             .split(X[train_idx], y[train_idx], g[train_idx]))
            _check_folds_have_support(y[train_idx], inner_rel, classes,
                                      f"repeat {r}, outer fold {k}, inner")
            inner = tuple((train_idx[a], train_idx[b]) for a, b in inner_rel)
            folds.append(Fold(repeat=r, outer=k, train=train_idx, eval=eval_idx, inner=inner))
    return folds
