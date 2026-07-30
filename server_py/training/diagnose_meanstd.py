"""Diagnostic only (not part of the shipped pipeline): trains the OLD
system's proven mean/std-pooling classifier architecture on the NEW
(mobilenetv4) backbone's embeddings, to isolate whether the new backbone is
an improvement independent of the GRU temporal head's data-hunger problem.

If this beats the JS baseline but the GRU (training.train) does not, the
bottleneck is the temporal head's capacity vs. the ~145-clip dataset size,
not the backbone. If this ALSO fails to beat baseline, the new backbone
itself isn't yet a clear win on this little data either.

Usage (from server_py/, training venv):
    .venv-train/Scripts/python.exe -m training.diagnose_meanstd
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.labels import LABELS
from training.dataset import class_weights
from training.train import CACHE_DIR, MANIFEST_PATH, confusion_matrix, load_entries, per_class_report

EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001


def mean_std_features(embeddings: np.ndarray) -> np.ndarray:
    """[T,1280] -> [2560] mean||std, matching the JS trainer's FEATURE_DIM."""
    mean = embeddings.mean(axis=0)
    std = embeddings.std(axis=0)
    return np.concatenate([mean, std])


class MeanStdClassifier(nn.Module):
    def __init__(self, input_dim=2560, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    train_entries = load_entries(manifest, "train")
    val_entries = load_entries(manifest, "val")

    x_train = np.stack([mean_std_features(e["embeddings"]) for e in train_entries])
    y_train = np.array([LABELS.index(e["label"]) for e in train_entries])
    x_val = np.stack([mean_std_features(e["embeddings"]) for e in val_entries])
    y_val = np.array([LABELS.index(e["label"]) for e in val_entries])

    print(f"train samples {len(x_train)} (one per clip-variant, full-clip mean/std), val samples {len(x_val)}")

    weights = class_weights(train_entries)
    model = MeanStdClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)

    x_train_t = torch.from_numpy(x_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    x_val_t = torch.from_numpy(x_val).float()
    y_val_t = torch.from_numpy(y_val).long()

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    epochs_since_best = 0
    n = len(x_train_t)

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(x_train_t[idx])
            loss = criterion(logits, y_train_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= PATIENCE:
                print(f"  Early stop at epoch {epoch + 1}")
                break

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{EPOCHS}  val_loss={val_loss:.3f}")

    model.load_state_dict(best_state)
    print(f"Restored best epoch {best_epoch + 1} (val_loss {best_val_loss:.3f})")

    model.eval()
    with torch.no_grad():
        pred_ids = model(x_val_t).argmax(dim=1).tolist()
    true_ids = y_val.tolist()
    acc = sum(t == p for t, p in zip(true_ids, pred_ids)) / len(true_ids)
    cm = confusion_matrix(true_ids, pred_ids)
    report = per_class_report(cm)

    print(f"\nDiagnostic (mean/std pooling on NEW backbone) val accuracy: {acc*100:.1f}%")
    for label, r in report.items():
        print(f"  {label:10s} precision={r['precision']*100:5.1f}%  recall={r['recall']*100:5.1f}%  support={r['support']}")
    print("Confusion matrix (rows=true, cols=pred):")
    print("  " + "  ".join(f"{l[:4]:>6s}" for l in LABELS))
    for i, row in enumerate(cm.tolist()):
        print(f"  {LABELS[i][:4]:4s} " + "  ".join(f"{v:6d}" for v in row))


if __name__ == "__main__":
    main()
