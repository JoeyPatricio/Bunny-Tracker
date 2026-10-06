"""Classification metrics, one definition for every script that reports them.

numpy only, deliberately. calibrate_gate.py imports check_degenerate from here
and runs on the RUNTIME venv, which has no sklearn and no torch. The metrics
are checked against sklearn in training/test_splits_metrics.py instead.

Everything works from a confusion matrix (rows = true, cols = predicted), so a
pooled out-of-fold matrix and a single-split matrix go through identical code.
"""
import math

import numpy as np

from shared.labels import LABELS


def confusion_matrix(true_ids, pred_ids, n_classes: int = len(LABELS)) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_ids, pred_ids, strict=True):
        cm[t, p] += 1
    return cm


def per_class_report(cm: np.ndarray, labels: list[str] = LABELS) -> dict:
    """Precision, recall and support per class. The shape train_result.json
    and quant_result.json already store, kept as-is so old results stay
    readable."""
    report = {}
    for k, label in enumerate(labels):
        tp = cm[k, k]
        support = cm[k].sum()
        pred_total = cm[:, k].sum()
        report[label] = {
            "precision": float(tp / pred_total) if pred_total else 0.0,
            "recall": float(tp / support) if support else 0.0,
            "support": int(support),
        }
    return report


def _f1_per_class(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(float)
    pred_total = cm.sum(axis=0)
    support = cm.sum(axis=1)
    precision = np.divide(tp, pred_total, out=np.zeros_like(tp), where=pred_total > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    return np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)


def macro_f1(cm: np.ndarray) -> float:
    """Unweighted mean F1 over classes that occur as a true or predicted label,
    with 0 for undefined precision/recall. Matches sklearn's f1_score(
    average="macro", zero_division=0). On a pooled OOF matrix every class has
    support, so this is the mean over the whole vocabulary."""
    active = (cm.sum(axis=0) + cm.sum(axis=1)) > 0
    if not active.any():
        return 0.0
    return float(_f1_per_class(cm)[active].mean())


def balanced_accuracy(cm: np.ndarray) -> float:
    """Mean recall over classes with support. Always guessing the majority
    class scores exactly 1/n_classes, which is why it is the headline rather
    than plain accuracy on an imbalanced dataset."""
    support = cm.sum(axis=1)
    has = support > 0
    if not has.any():
        return 0.0
    return float((np.diag(cm)[has] / support[has]).mean())


def cohen_kappa(cm: np.ndarray) -> float:
    """Agreement beyond chance. Leads the paper because it is directly
    comparable to the human-human kappa from the Phase 2 blind re-code."""
    n = cm.sum()
    if n == 0:
        return 0.0
    observed = np.trace(cm) / n
    expected = float((cm.sum(axis=0) * cm.sum(axis=1)).sum()) / (n * n)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return float((observed - expected) / (1 - expected))


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. Used on per-class recall,
    where n is a handful of clips and the normal approximation would give
    intervals that run past 0 or 1. n = 0 carries no information: (0, 1)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def classification_report(cm: np.ndarray, labels: list[str] = LABELS) -> dict:
    """Everything cv_result.json needs from one confusion matrix."""
    f1 = _f1_per_class(cm)
    per_class = per_class_report(cm, labels)
    for k, label in enumerate(labels):
        support = int(cm[k].sum())
        lo, hi = wilson_interval(int(cm[k, k]), support)
        per_class[label].update({"f1": float(f1[k]), "recall_ci95": [lo, hi]})
    n = int(cm.sum())
    return {
        "n": n,
        "accuracy": float(np.trace(cm) / n) if n else 0.0,
        "macro_f1": macro_f1(cm),
        "balanced_accuracy": balanced_accuracy(cm),
        "kappa": cohen_kappa(cm),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": list(labels),
    }


def pool_oof(n_clips: int, fold_predictions) -> np.ndarray:
    """Assemble one out-of-fold prediction per clip from (eval_indices,
    predictions) pairs, one pair per outer fold of a single repeat. Raises if
    any clip is predicted twice or never: either means the folds were not a
    partition, and the pooled matrix would quietly double-count or drop clips."""
    oof = np.full(n_clips, -1, dtype=int)
    for eval_idx, preds in fold_predictions:
        eval_idx = np.asarray(eval_idx)
        if (oof[eval_idx] != -1).any():
            raise ValueError("a clip received an out-of-fold prediction from two folds")
        oof[eval_idx] = preds
    if (oof == -1).any():
        raise ValueError(f"{int((oof == -1).sum())} clip(s) received no out-of-fold prediction")
    return oof


def check_degenerate(P: np.ndarray, y: np.ndarray) -> bool:
    """A model that names one class for everything makes every threshold moot.

    This is the check that was missing when a July model kept being served
    against September footage: nothing errored, train_result.json still
    reported 61.5%, and the classifier had quietly become a constant.
    """
    classes = sorted(set(P.argmax(axis=1).tolist()))
    base_rate = max(np.bincount(y, minlength=len(LABELS))) / len(y)
    accuracy = (P.argmax(axis=1) == y).mean()
    print(f"Deployed model: argmax accuracy {accuracy * 100:.1f}%  "
          f"(base rate {base_rate * 100:.1f}%)  "
          f"distinct predicted classes: {[LABELS[c] for c in classes]}")
    if len(classes) == 1:
        print(f"\n  ⚠ DEGENERATE: every clip is predicted '{LABELS[classes[0]]}'. The model carries no\n"
              f"    signal on this footage, so no threshold can separate anything below. Retrain\n"
              f"    against the current clips before reading the tables.\n")
        return True
    if accuracy <= base_rate:
        print("\n  ⚠ At or below base rate — the model is no better than always guessing the\n"
              "    most common class. Treat the tables below as untrustworthy.\n")
    return False
