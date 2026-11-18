"""Regression gate for the training guards.

Each check drives the shared fitting loop with deliberately degenerate inputs and
asserts a clear SystemExit rather than the ZeroDivisionError or TypeError the
same input used to produce. Nothing here trains a real model or touches the
embedding cache.

Usage (from server/, TRAINING venv):
    .venv-train/Scripts/python.exe -m tests.training.test_train_guards
"""
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.labels import LABELS
from tests.checks import check, finish
from training import train as T


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
    """An empty val split used to be a ZeroDivisionError."""
    expect_systemexit("5.1 empty val split exits clearly",
                      lambda: T.fit_head([fake_entry()], []), must_mention="Validation split is empty")


def test_empty_train_split() -> None:
    """The mirror case: nothing to train on."""
    expect_systemexit("5.1 empty train split exits clearly",
                      lambda: T.fit_head([], [fake_entry()]), must_mention="Training split is empty")


def test_evaluate_head_rejects_empty() -> None:
    """Evaluation must reject an empty set before computing accuracy."""
    head = T.TemporalHead(emb_mean=torch.zeros(1280), emb_std=torch.ones(1280))
    expect_systemexit(
        "5.1 evaluate_head rejects an empty val set",
        lambda: T.evaluate_head(head, []),
        must_mention="validation split is empty",
    )


def test_nan_val_loss_fails_fast() -> None:
    """NaN every epoch left best_state None -> TypeError at the end."""
    entries = [fake_entry(label=lbl) for lbl in LABELS]

    class NanLoss(torch.nn.Module):
        """Stands in for CrossEntropyLoss, always NaN, to force the divergence
        path without needing weights that actually blow up."""

        def __init__(self, *a, **kw):
            super().__init__()

        def forward(self, logits, target):
            return logits.sum() * float("nan")

    def run():
        with patch.object(T.nn, "CrossEntropyLoss", NanLoss):
            T.fit_head(entries, entries)

    expect_systemexit("5.1 NaN val loss exits at the first epoch", run, must_mention="NaN at epoch 1")


def test_healthy_run_still_works() -> None:
    """The guards must not fire on a normal tiny run."""
    entries = [fake_entry(frames=4, label=lbl) for lbl in LABELS]
    try:
        result = T.fit_head(entries, entries, epochs=2)
        accuracy, _ = T.evaluate_head(result["head"], entries)
        check("5.1 a healthy run completes", result["best_epoch"] >= 0)
        check("5.1 accuracy is a real number", 0.0 <= accuracy <= 1.0, str(accuracy))
        final_fit = T.fit_head(entries, epochs=2)
        check("5.1 final fit runs without validation", final_fit["best_epoch"] == 1
              and final_fit["best_val_loss"] is None)
    except SystemExit as err:
        check("5.1 a healthy run completes", False, f"a guard fired on good input: {err}")
    except Exception as err:
        check("5.1 a healthy run completes", False, f"{err!r}")


def test_training_requires_mode() -> None:
    """Omitting the mode must stop before writing a nondeployable checkpoint."""
    result = subprocess.run(
        [sys.executable, "-m", "training.train"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=30,
    )
    check("5.1 training requires an explicit mode", result.returncode == 2
          and "--all-clips" in result.stderr and "--fold" in result.stderr, result.stderr)


def main() -> None:
    for fn in (
        test_empty_val_split,
        test_empty_train_split,
        test_evaluate_head_rejects_empty,
        test_nan_val_loss_fails_fast,
        test_healthy_run_still_works,
        test_training_requires_mode,
    ):
        print(f"\n{fn.__doc__.splitlines()[0]}")
        fn()

    finish("training guard")


if __name__ == "__main__":
    main()
