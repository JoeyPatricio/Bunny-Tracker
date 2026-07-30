"""CLI entrypoint: load cached embeddings -> fit the temporal head -> evaluate.

Usage (from server_py/, using the training venv):
    .venv-train/Scripts/python.exe -m training.train [--window N] [--unfreeze]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.labels import LABELS
from training.dataset import WindowDataset, class_weights
from training.model import TemporalHead

CACHE_DIR = Path(__file__).resolve().parent.parent / "models" / "cache"
MANIFEST_PATH = CACHE_DIR / "manifest.json"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.0003


def load_entries(manifest: dict, split: str) -> list[dict]:
    entries = []
    for e in manifest[split]:
        emb = np.load(CACHE_DIR / e["embeddings"])
        entries.append({"embeddings": emb, "label": e["label"], "windows": [tuple(w) for w in e["windows"]]})
    return entries


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


def confusion_matrix(true_ids: list[int], pred_ids: list[int]) -> np.ndarray:
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for t, p in zip(true_ids, pred_ids):
        cm[t, p] += 1
    return cm


def per_class_report(cm: np.ndarray) -> dict:
    report = {}
    for k, label in enumerate(LABELS):
        tp = cm[k, k]
        support = cm[k].sum()
        pred_total = cm[:, k].sum()
        precision = tp / pred_total if pred_total else 0.0
        recall = tp / support if support else 0.0
        report[label] = {"precision": precision, "recall": recall, "support": int(support)}
    return report


@torch.no_grad()
def evaluate_head(head: nn.Module, val_entries: list[dict]):
    """One prediction per val clip (single window each) -> accuracy, confusion matrix."""
    head.eval()
    if not val_entries:
        # Guarded here as well as in train_one: the accuracy divide below is
        # silent about why it failed, and "ZeroDivisionError" after a full
        # training run is a miserable way to learn the val split was empty.
        raise SystemExit("Cannot evaluate: the validation split is empty.")
    true_ids, pred_ids = [], []
    for e in val_entries:
        target = LABELS.index(e["label"])
        start, end = e["windows"][0]
        window = torch.from_numpy(e["embeddings"][start:end]).float().unsqueeze(0)  # [1,N,D]
        logits = head(window)
        pred = logits.argmax(dim=1).item()
        true_ids.append(target)
        pred_ids.append(pred)
    cm = confusion_matrix(true_ids, pred_ids)
    acc = sum(t == p for t, p in zip(true_ids, pred_ids)) / len(true_ids)
    return acc, cm


def train_one(window: int, unfreeze_note: str = "") -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    train_entries_full = load_entries(manifest, "train")
    val_entries_full = load_entries(manifest, "val")

    train_entries = restrict_window(train_entries_full, window)
    if window == manifest["window"]:
        val_entries = val_entries_full
    elif f"val{window}" in manifest:
        val_entries = load_entries(manifest, f"val{window}")
    else:
        val_entries = restrict_window(val_entries_full, window)

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

    train_ds = WindowDataset(train_entries)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    weights = class_weights(train_entries_full)
    print(f"Class weights: {dict(zip(LABELS, weights.tolist()))}")

    # Standardize embeddings using train-set stats, baked into the head (see
    # TemporalHead's emb_mean/emb_std buffers) so it exports straight into ONNX.
    all_train_embeddings = np.concatenate([e["embeddings"] for e in train_entries_full], axis=0)
    emb_mean = torch.from_numpy(all_train_embeddings.mean(axis=0)).float()
    emb_std = torch.from_numpy(all_train_embeddings.std(axis=0) + 1e-6).float()

    head = TemporalHead(emb_mean=emb_mean, emb_std=emb_std)
    optimizer = torch.optim.Adam(head.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    epochs_since_best = 0

    for epoch in range(EPOCHS):
        head.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
            optimizer.step()

        head.eval()
        with torch.no_grad():
            val_losses = []
            for e in val_entries:
                target = LABELS.index(e["label"])
                start, end = e["windows"][0]
                window_t = torch.from_numpy(e["embeddings"][start:end]).float().unsqueeze(0)
                logits = head(window_t)
                val_losses.append(criterion(logits, torch.tensor([target])).item())
            val_loss = sum(val_losses) / len(val_losses)

        # Fail on the first NaN rather than at the end. `nan < best_val_loss`
        # is always False, so a NaN run never records a best checkpoint and
        # only reveals itself as a TypeError from load_state_dict(None) after
        # all EPOCHS have burned. The weights are already NaN by this point and
        # will not recover on their own.
        if math.isnan(val_loss):
            raise SystemExit(
                f"Validation loss went NaN at epoch {epoch + 1}; the head has diverged. "
                f"Gradient clipping is already on (max_norm=5.0), so look at "
                f"LEARNING_RATE={LEARNING_RATE} and at whether the embedding std used "
                f"for standardization has near-zero entries."
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= PATIENCE:
                print(f"  Early stop at epoch {epoch + 1} (val_loss stuck {PATIENCE} epochs)")
                break

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{EPOCHS}  val_loss={val_loss:.3f}")

    # Backstop for the NaN guard above: reachable only if the epoch loop never
    # ran at all. load_state_dict(None) raises a TypeError that names neither
    # this variable nor the reason.
    if best_state is None:
        raise SystemExit(
            f"No best checkpoint was recorded across {EPOCHS} epochs, so there is "
            f"nothing to restore."
        )

    head.load_state_dict(best_state)
    print(f"Restored best epoch {best_epoch + 1} (val_loss {best_val_loss:.3f})")

    val_acc, cm = evaluate_head(head, val_entries)
    report = per_class_report(cm)

    return {
        "window": window,
        "unfreeze_note": unfreeze_note,
        "head": head,
        "val_acc": val_acc,
        "confusion_matrix": cm.tolist(),
        "per_class": report,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=8)
    args = parser.parse_args()

    result = train_one(args.window)
    print(f"\nVal accuracy (window={result['window']}): {result['val_acc']*100:.1f}%")
    print("Per-class precision/recall:")
    for label, r in result["per_class"].items():
        print(f"  {label:10s} precision={r['precision']*100:5.1f}%  recall={r['recall']*100:5.1f}%  support={r['support']}")
    print("Confusion matrix (rows=true, cols=pred):")
    print("  " + "  ".join(f"{l[:4]:>6s}" for l in LABELS))
    for i, row in enumerate(result["confusion_matrix"]):
        print(f"  {LABELS[i][:4]:4s} " + "  ".join(f"{v:6d}" for v in row))

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(result["head"].state_dict(), MODELS_DIR / "head_state.pt")
    (MODELS_DIR / "train_result.json").write_text(json.dumps({
        "window": result["window"], "val_acc": result["val_acc"],
        "confusion_matrix": result["confusion_matrix"], "per_class": result["per_class"],
    }, indent=2))
    print(f"\nSaved head weights to {MODELS_DIR / 'head_state.pt'}")


if __name__ == "__main__":
    main()
