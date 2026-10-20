"""Fit the temporal head on cached embeddings.

Three modes, one fitting loop (fit_head, shared with training/crossval.py):

    --all-clips   The deployable model. Trains on every labeled model-class
                  clip with the hyperparameters cross-validation selected
                  (models/cv_result.json). There is no held-out data in this
                  mode, so it reports no accuracy at all: the number to quote
                  is cv_result.json's pooled out-of-fold estimate.
    --fold R:K    Debugging aid. Trains on outer fold K of repeat R with the
                  config CV selected for that fold and prints its eval report.
    (no flag)     Legacy single train/val split, early-stopped on val. Its val
                  accuracy is a selection score and is printed as one; it is
                  not written to train_result.json.

train_result.json carries what export_onnx.py needs (window) plus a pointer to
cv_result.json, and deliberately no accuracy field, so a selection score can
no longer be quoted as a result by reading the wrong file.

--arm picks which crossval gru arm the CV modes take their config and cache
from: "gru" (the run's primary aug policy, the default) or a paired
"gru@<policy>" arm such as gru@albu_v1. Augmentation only changes training
variants, so the runtime's full-frame input is unchanged and any such arm can
be deployed. A crop-view arm (gru@crop) is refused: the agent feeds full
frames, so a head trained on crops would run on inputs it never saw.
train_result.json records the arm, its aug policy and view, and the CV run.

Usage (from server_py/, using the training venv):
    .venv-train/bin/python -m training.train --all-clips [--seed N] [--arm gru@albu_v1]
    .venv-train/bin/python -m training.train --fold 0:3
    .venv-train/bin/python -m training.train [--window N] [--seed N]
"""
import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.labels import LABELS
from training.cache import (
    DEFAULT_CACHE_DIR,
    FULL_VIEW,
    legacy_split_indices,
    load_manifest,
    split_entries,
)
from training.dataset import class_weights
from training.metrics import confusion_matrix, per_class_report
from training.model import TemporalHead

CACHE_DIR = DEFAULT_CACHE_DIR
MANIFEST_PATH = CACHE_DIR / "manifest.json"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.0003
SEED = 0


def seed_everything(seed: int) -> torch.Generator:
    """Make a run reproducible, and return the generator for batch shuffling.

    Without this, two runs of train.py on identical cached embeddings give
    different numbers: head weight init, dropout, and batch shuffling all
    draw from unseeded global RNGs. That makes "the retrain is 2 points worse"
    unreadable, because a 2-point gap is inside the run-to-run spread and there
    was no way to tell the two apart.

    fit_classifier calls this at the start of every fit and then re-initialises
    the model's parameters, so a fit's result depends only on its own seed and
    not on how many fits ran before it in the same process (crossval runs
    hundreds). PYTHONHASHSEED is set for completeness; it only binds in a
    subprocess.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN picks convolution algorithms by benchmarking unless told not to,
    # and the winner can differ between runs on the same hardware. No-op on
    # CPU, which is where this currently trains.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def load_entries(manifest: dict, split: str) -> list[dict]:
    """Entries for "train" or "val" (or "val<W>" for another window length) of
    the legacy single split, from the schema-2 cache.

    Legacy: this is the one train/val split every pre-Phase-3 number came from,
    rebuilt deterministically from manifest["clips"] with split_by_group. Its
    val accuracy is a selection score. crossval.py replaces it as the source
    of reported numbers.
    """
    train_idx, val_idx = legacy_split_indices(manifest)
    window = int(split[3:]) if split.startswith("val") and split[3:] else None
    train_entries, val_entries = split_entries(manifest, train_idx, val_idx, window=window,
                                               cache_dir=CACHE_DIR)
    return train_entries if split == "train" else val_entries


def restrict_window(entries: list[dict], window: int) -> list[dict]:
    """Re-derive windows of a given length from each entry's cached embeddings
    (used by the escalation path to sweep window length without re-extracting)."""
    out = []
    for e in entries:
        t = e["embeddings"].shape[0]
        if window >= t:
            windows = [(0, t)]
        else:
            starts = sorted(set(int(s) for s in np.linspace(0, t - window, num=min(3, t - window + 1))))
            windows = [(s, s + window) for s in starts]
        out.append({**e, "windows": windows})
    return out


@torch.no_grad()
def evaluate_head(head: nn.Module, val_entries: list[dict]):
    """One prediction per val clip (single window each) -> accuracy, confusion matrix."""
    head.eval()
    if not val_entries:
        # Guarded here as well as in train_one: the accuracy divide below is
        # silent about why it failed, and "ZeroDivisionError" after a full
        # training run is a miserable way to learn the val split was empty.
        raise SystemExit("Cannot evaluate: the validation split is empty.")
    x, y = eval_tensors(val_entries)
    pred_ids = head(x).argmax(dim=1).tolist()
    true_ids = y.tolist()
    cm = confusion_matrix(true_ids, pred_ids)
    acc = sum(t == p for t, p in zip(true_ids, pred_ids, strict=True)) / len(true_ids)
    return acc, cm


def train_tensors(entries: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Every (window, target) sample of the training entries, materialised.
    Same samples WindowDataset yields, stacked once instead of per batch."""
    xs, ys = [], []
    for e in entries:
        target = LABELS.index(e["label"])
        for start, end in e["windows"]:
            xs.append(e["embeddings"][start:end])
            ys.append(target)
    return torch.from_numpy(np.stack(xs)).float(), torch.tensor(ys, dtype=torch.long)


def eval_tensors(entries: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """One window per eval clip, its first (eval entries carry exactly one)."""
    xs = [e["embeddings"][slice(*e["windows"][0])] for e in entries]
    ys = [LABELS.index(e["label"]) for e in entries]
    return torch.from_numpy(np.stack(xs)).float(), torch.tensor(ys, dtype=torch.long)


def standardization(train_entries: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """emb_mean/emb_std over every frame of the given training entries only.

    This is what gets baked into the exported head, so it must come from the
    clips the head is trained on and nothing else. Under cross-validation that
    means one fold's training clips: computing it over all clips, as train.py
    did before Phase 3, lets the evaluation clips shape the model's input
    scaling. training/test_crossval_guards.py checks it per fold.
    """
    all_frames = np.concatenate([e["embeddings"] for e in train_entries], axis=0)
    return (torch.from_numpy(all_frames.mean(axis=0)).float(),
            torch.from_numpy(all_frames.std(axis=0) + 1e-6).float())


def fit_classifier(model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor,
                   x_val: torch.Tensor | None, y_val: torch.Tensor | None, *,
                   weights: torch.Tensor, lr: float, seed: int,
                   epochs: int | None = None, patience: int = PATIENCE) -> dict:
    """The one training loop, for the GRU head and crossval's MLP arm alike.

    With validation tensors: early stopping on val loss, best epoch restored.
    Without (x_val None): exactly `epochs` epochs and the final state, which
    is how a model is fitted on a full outer-train set or on all clips after
    CV has chosen the epoch count. There is nothing left to stop on there, and
    stopping on the eval clips would be selection on the evaluation data.
    """
    epochs = EPOCHS if epochs is None else epochs
    generator = seed_everything(seed)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    # Val loss is the plain mean cross-entropy per clip, unweighted, which is
    # what the per-clip loop this replaced computed.
    val_criterion = nn.CrossEntropyLoss()
    n = len(x_train)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    epochs_since_best = 0
    history = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=generator)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(x_train[idx]), y_train[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if x_val is None:
            continue

        model.eval()
        with torch.no_grad():
            val_loss = val_criterion(model(x_val), y_val).item()
        history.append(val_loss)

        # Fail on the first NaN rather than at the end. `nan < best_val_loss`
        # is always False, so a NaN run never records a best checkpoint and
        # only reveals itself as a TypeError from load_state_dict(None) after
        # all epochs have burned. The weights are already NaN by this point and
        # will not recover on their own.
        if math.isnan(val_loss):
            raise SystemExit(
                f"Validation loss went NaN at epoch {epoch + 1}; the head has diverged. "
                f"Gradient clipping is already on (max_norm=5.0), so look at the "
                f"learning rate ({lr}) and at whether the embedding std used "
                f"for standardization has near-zero entries."
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                break

    if x_val is None:
        model.eval()
        return {"model": model, "best_epoch": epochs - 1, "best_val_loss": None, "history": history}

    # Backstop for the NaN guard above: reachable only if the epoch loop never
    # ran at all. load_state_dict(None) raises a TypeError that names neither
    # this variable nor the reason.
    if best_state is None:
        raise SystemExit(
            f"No best checkpoint was recorded across {epochs} epochs, so there is "
            f"nothing to restore."
        )
    model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "best_epoch": best_epoch, "best_val_loss": best_val_loss, "history": history}


def fit_head(train_entries: list[dict], val_entries: list[dict] | None = None, *,
             lr: float = LEARNING_RATE, seed: int = SEED, epochs: int | None = None,
             patience: int = PATIENCE) -> dict:
    """Fit a TemporalHead. Standardization buffers and class weights come from
    `train_entries` alone (see standardization()); they are returned too so
    the per-fold leakage check can compare them."""
    emb_mean, emb_std = standardization(train_entries)
    weights = class_weights(train_entries)
    x_train, y_train = train_tensors(train_entries)
    x_val, y_val = eval_tensors(val_entries) if val_entries else (None, None)
    head = TemporalHead(emb_mean=emb_mean, emb_std=emb_std)
    fit = fit_classifier(head, x_train, y_train, x_val, y_val, weights=weights, lr=lr,
                         seed=seed, epochs=epochs, patience=patience)
    return {**fit, "head": fit["model"], "emb_mean": emb_mean, "emb_std": emb_std,
            "class_weights": weights}


def train_one(window: int, unfreeze_note: str = "", seed: int = SEED) -> dict:
    """Legacy single-split fit, early-stopped on the legacy val split."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    train_entries_full = load_entries(manifest, "train")
    val_entries_full = load_entries(manifest, "val")

    train_entries = restrict_window(train_entries_full, window)
    # Eval clips at another window length come from that length's own eval
    # sample (extract_frames.py --eval-windows), never a slice of this one:
    # slicing would change which moments of the clip are evaluated.
    val_entries = val_entries_full if window == manifest["window"] else load_entries(manifest, f"val{window}")

    # Check the splits before spending anything on them. An empty val split
    # otherwise surfaces as a ZeroDivisionError on the val_loss average at the
    # end of epoch 1, which says nothing about the actual problem.
    if not train_entries:
        raise SystemExit(
            f"Training split is empty (window={window}). Regenerate the embedding cache "
            f"with training/extract_frames.py, and check that server/labels.json has "
            f"human-confirmed labels in it."
        )
    if not val_entries:
        raise SystemExit(
            f"Validation split is empty (window={window}). Every clip landed in train, "
            f"which means dataset.py's split_by_group had nothing to hold back. Check "
            f"{MANIFEST_PATH} and that there is more than one source group in "
            f"server/labels.json."
        )

    fit = fit_head(train_entries, val_entries, seed=seed)
    print(f"Class weights: {dict(zip(LABELS, fit['class_weights'].tolist(), strict=True))}")
    print(f"Restored best epoch {fit['best_epoch'] + 1} (val_loss {fit['best_val_loss']:.3f})")
    head = fit["head"]
    val_acc, cm = evaluate_head(head, val_entries)
    return {
        "window": window,
        "seed": seed,
        "unfreeze_note": unfreeze_note,
        "head": head,
        "val_acc": val_acc,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_report(cm),
        "best_epoch": fit["best_epoch"],
        "best_val_loss": fit["best_val_loss"],
    }


# ── CV-driven modes ──────────────────────────────────────────────────────

def load_cv_arm(cache_dir: Path, out_dir: Path, arm: str) -> tuple[dict, dict, dict]:
    """cv_result.json, the {aug_policy, view} cache that `arm` was fitted on,
    and that cache's manifest, checked to hold the labels CV ran on."""
    path = Path(out_dir) / "cv_result.json"
    if not path.is_file():
        raise SystemExit(f"No {path}. Run training/crossval.py first; its selections set the hyperparameters here.")
    cv = json.loads(path.read_text())
    caches = cv.get("arm_cache")
    if caches is None:
        raise SystemExit(f"{path} predates per-arm cache records (arm_cache). Re-run crossval.py.")
    gru_arms = sorted(a for a in caches if a.split("@")[0] == "gru")
    if arm not in gru_arms:
        raise SystemExit(f"{path} has no arm {arm!r}; its gru arms are {gru_arms}.")
    cache = caches[arm]
    if cache["view"] != FULL_VIEW:
        raise SystemExit(
            f"{arm} was fitted on the {cache['view']!r} view. The agent feeds full frames and has no "
            f"matching ROI preprocessing, so this head cannot be trained for deployment or exported. "
            f"Deploy a full-view arm (one of {[a for a in gru_arms if caches[a]['view'] == FULL_VIEW]}); "
            f"crop stays a crossval measurement until the runtime crops identically."
        )
    manifest = load_manifest(cache_dir, cache["aug_policy"], cache["view"])
    if cv.get("labels_sha256") != manifest.get("labels_sha256"):
        raise SystemExit(
            f"{path} was computed on labels {cv.get('labels_sha256', '?')[:12]} but the cache "
            f"holds labels {manifest.get('labels_sha256', '?')[:12]}. Re-run crossval.py on the "
            f"current cache so the selected hyperparameters match the data."
        )
    return cv, cache, manifest


def aggregate_selection(selections: list[dict], default_window: int) -> dict:
    """One config from the per-outer-fold selections, for the all-clips fit.

    window and lr: the most common choice across outer folds (ties go to the
    cache's default window, then the smaller value, so the result is stable).
    epochs: the median of the selected epoch counts, rounded. The median, not
    the mean, because one fold that early-stopped late should not drag every
    other fold's choice along with it. How much the folds disagree is reported
    in cv_result.json and is itself a result.
    """
    def mode(values, prefer=None):
        counts = Counter(values)
        top = max(counts.values())
        tied = sorted(v for v, c in counts.items() if c == top)
        return prefer if prefer in tied else tied[0]

    return {
        "window": int(mode([s["window"] for s in selections], prefer=default_window)),
        "lr": float(mode([s["lr"] for s in selections])),
        "epochs": int(round(float(np.median([s["epochs"] for s in selections])))),
    }


def train_all_clips(cache_dir: Path, out_dir: Path, seed: int = SEED, arm: str = "gru") -> dict:
    cv, cache, manifest = load_cv_arm(cache_dir, out_dir, arm)
    config = aggregate_selection(cv["arms"][arm]["selections"], manifest["window"])
    every = list(range(len(manifest["clips"])))
    entries, _ = split_entries(manifest, every, [], window=config["window"], cache_dir=cache_dir)
    print(f"All-clips fit on {len(manifest['clips'])} clips with {arm}'s CV-selected config {config}")
    fit = fit_head(entries, None, lr=config["lr"], seed=seed, epochs=config["epochs"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(fit["head"].state_dict(), out_dir / "head_state.pt")
    result = {
        "mode": "all-clips",
        "window": config["window"],
        "lr": config["lr"],
        "epochs": config["epochs"],
        "seed": seed,
        "n_clips": len(manifest["clips"]),
        "labels_sha256": manifest["labels_sha256"],
        "ethogram_version": manifest.get("ethogram_version"),
        "cv_result": "cv_result.json",
        "cv_run_id": cv.get("run_id"),
        "cv_arm": arm,
        "aug_policy": cache["aug_policy"],
        "aug_policy_detail": manifest.get("aug_policy_detail"),
        "view": cache["view"],
        "note": "No held-out data in this mode, so no accuracy is reported. Quote "
                "cv_result.json's pooled out-of-fold estimate, never a number from here.",
    }
    (out_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    print(f"Saved {out_dir / 'head_state.pt'} and train_result.json")
    return {**result, "head": fit["head"], "emb_mean": fit["emb_mean"], "emb_std": fit["emb_std"]}


def train_fold(cache_dir: Path, out_dir: Path, repeat: int, outer: int, seed: int = SEED,
               arm: str = "gru") -> dict:
    from training.splits import make_folds

    cv, _, manifest = load_cv_arm(cache_dir, out_dir, arm)
    cfg = cv["config"]
    folds = make_folds([c["filename"] for c in manifest["clips"]], [c["label"] for c in manifest["clips"]],
                       n_splits=cfg["n_splits"], n_inner=cfg["n_inner"], repeats=cfg["repeats"],
                       seed=cfg["seed"])
    fold = next((f for f in folds if f.repeat == repeat and f.outer == outer), None)
    if fold is None:
        raise SystemExit(f"No fold {repeat}:{outer}; cv_result.json has {cfg['repeats']} repeats x {cfg['n_splits']} folds.")
    sel = next(s for s in cv["arms"][arm]["selections"] if s["repeat"] == repeat and s["outer"] == outer)
    train_entries, eval_entries = split_entries(manifest, fold.train, fold.eval, window=sel["window"],
                                                cache_dir=cache_dir)
    fit = fit_head(train_entries, None, lr=sel["lr"], seed=seed, epochs=sel["epochs"])
    acc, cm = evaluate_head(fit["head"], eval_entries)
    print(f"Fold {repeat}:{outer} of {arm} with {sel}: {len(eval_entries)} eval clips")
    print_report(cm)
    print("One fold of one repeat. Quote cv_result.json's pooled estimate, not this.")
    return {"selection": sel, "accuracy": acc, "confusion_matrix": cm.tolist()}


def print_report(cm: np.ndarray) -> None:
    for label, r in per_class_report(cm).items():
        print(f"  {label:16s} precision={r['precision']*100:5.1f}%  recall={r['recall']*100:5.1f}%  support={r['support']}")
    print("Confusion matrix (rows=true, cols=pred):")
    print("  " + "  ".join(f"{label[:4]:>6s}" for label in LABELS))
    for i, row in enumerate(np.asarray(cm).tolist()):
        print(f"  {LABELS[i][:4]:4s} " + "  ".join(f"{v:6d}" for v in row))


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all-clips", action="store_true",
                      help="final deployable fit on every clip, with CV-selected hyperparameters")
    mode.add_argument("--fold", metavar="R:K", help="fit one outer fold with its CV-selected config")
    parser.add_argument("--window", type=int, default=8, help="legacy mode only")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="RNG seed for weight init, dropout, and batch order")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR,
                        help="where cv_result.json is read and head_state.pt/train_result.json written")
    parser.add_argument("--arm", default="gru",
                        help="crossval gru arm whose config and cache the CV modes use, e.g. gru@albu_v1")
    args = parser.parse_args()

    if args.all_clips:
        train_all_clips(args.cache_dir, args.out_dir, seed=args.seed, arm=args.arm)
        return
    if args.fold:
        r, k = (int(x) for x in args.fold.split(":"))
        train_fold(args.cache_dir, args.out_dir, r, k, seed=args.seed, arm=args.arm)
        return

    result = train_one(args.window, seed=args.seed)
    print(f"\nLegacy split, window={result['window']}, seed={result['seed']}: "
          f"val accuracy {result['val_acc']*100:.1f}% (a SELECTION score, early-stopped on these "
          f"same clips; not written to train_result.json and not a result)")
    print_report(np.array(result["confusion_matrix"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result["head"].state_dict(), args.out_dir / "head_state.pt")
    (args.out_dir / "train_result.json").write_text(json.dumps({
        "mode": "legacy-split", "window": result["window"], "seed": result["seed"],
        "best_epoch": result["best_epoch"], "cv_result": "cv_result.json",
        "note": "Legacy single split. Its val accuracy is a selection score and is "
                "deliberately not recorded. Quote cv_result.json.",
    }, indent=2))
    print(f"\nSaved head weights to {args.out_dir / 'head_state.pt'}")


if __name__ == "__main__":
    main()
