"""Regression gate for notes/fixes.md installment 5.

Lives here rather than under tests/ because it needs torch, which only exists
in .venv-train (see environment.md's two-environments note). Everything under
tests/ must stay runnable on the runtime venv.

Each check drives train.py's guards with deliberately degenerate inputs and
asserts a clear SystemExit rather than the ZeroDivisionError or TypeError the
same input used to produce. Nothing here trains a real model or touches the
embedding cache.

Usage (from server_py/, TRAINING venv):
    .venv-train/Scripts/python.exe -m training.test_train_guards
"""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.labels import LABELS
from training import train as T

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")
        failures.append(name)


def expect_systemexit(name: str, fn, *, must_mention: str) -> None:
    """The guard must exit cleanly with a message that says what to look at,
    not raise the raw ZeroDivisionError/TypeError it used to."""
    try:
        fn()
    except SystemExit as err:
        msg = str(err)
        check(name, must_mention.lower() in msg.lower(), f"message did not mention {must_mention!r}: {msg!r}")
        return
    except (ZeroDivisionError, TypeError) as err:
        check(name, False, f"raised the original unguarded error: {err!r}")
        return
    except Exception as err:
        check(name, False, f"raised something unexpected: {err!r}")
        return
    check(name, False, "did not raise at all")


def fake_entry(frames: int = 8, dim: int = 1280, label: str | None = None) -> dict:
    return {
        "embeddings": np.zeros((frames, dim), dtype=np.float32),
        "label": label or LABELS[0],
        "windows": [(0, frames)],
    }


def test_empty_val_split() -> None:
    """fixes.md 5.1: an empty val split used to be a ZeroDivisionError."""
    manifest = {"window": 8, "train": [], "val": []}

    def run():
        with patch.object(T, "MANIFEST_PATH") as mp, patch.object(T, "load_entries") as le:
            mp.read_text.return_value = __import__("json").dumps(manifest)
            le.side_effect = lambda m, split: [fake_entry()] if split == "train" else []
            T.train_one(8)

    expect_systemexit("5.1 empty val split exits clearly", run, must_mention="Validation split is empty")


def test_empty_train_split() -> None:
    """The mirror case: nothing to train on."""
    manifest = {"window": 8, "train": [], "val": []}

    def run():
        with patch.object(T, "MANIFEST_PATH") as mp, patch.object(T, "load_entries") as le:
            mp.read_text.return_value = __import__("json").dumps(manifest)
            le.side_effect = lambda m, split: [] if split == "train" else [fake_entry()]
            T.train_one(8)

    expect_systemexit("5.1 empty train split exits clearly", run, must_mention="Training split is empty")


def test_evaluate_head_rejects_empty() -> None:
    """evaluate_head's own guard, independent of train_one's."""
    head = T.TemporalHead(emb_mean=torch.zeros(1280), emb_std=torch.ones(1280))
    expect_systemexit(
        "5.1 evaluate_head rejects an empty val set",
        lambda: T.evaluate_head(head, []),
        must_mention="validation split is empty",
    )


def test_nan_val_loss_fails_fast() -> None:
    """fixes.md 5.1: NaN every epoch left best_state None -> TypeError at the end."""
    manifest = {"window": 8, "train": [], "val": []}
    entries = [fake_entry(label=lbl) for lbl in LABELS]

    class NanLoss(torch.nn.Module):
        """Stands in for CrossEntropyLoss, always NaN, to force the divergence
        path without needing weights that actually blow up."""

        def __init__(self, *a, **kw):
            super().__init__()

        def forward(self, logits, target):
            return logits.sum() * float("nan")

    def run():
        with patch.object(T, "MANIFEST_PATH") as mp, patch.object(T, "load_entries") as le, \
             patch.object(T.nn, "CrossEntropyLoss", NanLoss):
            mp.read_text.return_value = __import__("json").dumps(manifest)
            le.side_effect = lambda m, split: entries
            T.train_one(8)

    expect_systemexit("5.1 NaN val loss exits at the first epoch", run, must_mention="NaN at epoch 1")


def test_healthy_run_still_works() -> None:
    """The guards must not fire on a normal tiny run."""
    manifest = {"window": 4, "train": [], "val": []}
    entries = [fake_entry(frames=4, label=lbl) for lbl in LABELS]
    saved_epochs = T.EPOCHS
    T.EPOCHS = 2
    try:
        with patch.object(T, "MANIFEST_PATH") as mp, patch.object(T, "load_entries") as le:
            mp.read_text.return_value = __import__("json").dumps(manifest)
            le.side_effect = lambda m, split: entries
            result = T.train_one(4)
        check("5.1 a healthy run completes", isinstance(result, dict) and "val_acc" in result)
        check("5.1 accuracy is a real number", 0.0 <= result["val_acc"] <= 1.0, str(result.get("val_acc")))
    except SystemExit as err:
        check("5.1 a healthy run completes", False, f"a guard fired on good input: {err}")
    except Exception as err:
        check("5.1 a healthy run completes", False, f"{err!r}")
    finally:
        T.EPOCHS = saved_epochs


def main() -> None:
    for fn in (
        test_empty_val_split,
        test_empty_train_split,
        test_evaluate_head_rejects_empty,
        test_nan_val_loss_fails_fast,
        test_healthy_run_still_works,
    ):
        print(f"\n{fn.__doc__.splitlines()[0]}")
        fn()

    if failures:
        print(f"\nFAIL: {len(failures)} check(s) failed")
        sys.exit(1)
    print("\nPASS: all training guard checks passed")


if __name__ == "__main__":
    main()
