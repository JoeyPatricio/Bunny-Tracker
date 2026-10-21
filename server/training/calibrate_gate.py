"""Pick AGENT_INTEREST_THRESHOLD from out-of-fold probabilities.

Before Phase 3 this ran the deployed ONNX pair over every labeled clip,
including the clips that model was trained on, and picked the threshold from
those in-sample scores. A model is most confident on its own training data,
so that threshold was fitted to an optimistic picture of the model and decides
when you get emailed. It now reads models/cv_oof.npz from training/crossval.py:
the gru arm's probabilities for each clip from folds that never trained on it,
averaged over repeats (every repeat's prediction for a clip is out-of-fold, so
the average is too).

It never loads an ONNX model. That also means it no longer measures int8
quantization; training/evaluate.py's deployment gates own that question.

It imports interest_score rather than reimplementing it, so changing the alert
policy automatically changes what this measures. Runs on the RUNTIME venv
(numpy only, no sklearn, no torch).

The unit here is one clip = one window = one candidate decision. The live gate
additionally requires AGENT_ALERT_STREAK consecutive candidate frames and a
motion floor, so real-world firing is strictly rarer than the recall column.
Read the numbers as an upper bound and a policy comparison, not a forecast.

Usage (from server/, runtime venv):
    .venv/bin/python -m training.calibrate_gate [--oof models/cv_oof.npz] [--arm gru]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.capture import interest_score
from shared.labels import LABELS, RESTING_INDICES
from training.metrics import check_degenerate

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
OOF_PATH = MODELS_DIR / "cv_oof.npz"

# v1 had one resting class; v2 has two, so "resting" is a set of indices and
# P(resting) is their sum rather than a single column.
RESTING_IDX = sorted(RESTING_INDICES)
THRESHOLDS = (10, 20, 25, 30, 40, 50, 60, 70, 80)


def load_oof(path: Path, arm: str = "gru") -> tuple[np.ndarray, np.ndarray]:
    """[clips, classes] out-of-fold probabilities (mean over repeats) and y."""
    if not Path(path).is_file():
        raise SystemExit(
            f"No out-of-fold probabilities at {path}. Run training/crossval.py first: "
            f"the threshold is calibrated on predictions for clips the model never "
            f"trained on, which only cross-validation produces."
        )
    data = np.load(path)
    key = f"{arm}_P"
    if key not in data.files:
        arms = sorted(k[:-2] for k in data.files if k.endswith("_P"))
        raise SystemExit(f"{path} has no '{arm}' arm (has {arms}).")
    P = data[key]
    if P.ndim != 3 or P.shape[2] != len(LABELS):
        raise SystemExit(f"{path} {key} has shape {P.shape}; expected [repeats, clips, {len(LABELS)}].")
    return P.mean(axis=0), data["y"]


def report(name: str, score: np.ndarray, interesting: np.ndarray, resting: np.ndarray) -> None:
    n_int, n_rest = int(interesting.sum()), int(resting.sum())
    print(f"POLICY  {name}")
    print(f"  {'thresh':>6s} {'recall (would alert)':>22s} {'false alarms':>20s} {'J':>6s}")
    best = None
    for t in THRESHOLDS:
        fires = score >= t
        recall = fires[interesting].mean() if n_int else 0.0
        false_alarm = fires[resting].mean() if n_rest else 0.0
        j = recall - false_alarm  # Youden's J: the separation this threshold buys
        if best is None or j > best[1]:
            best = (t, j)
        print(f"  {t:6d} {int(fires[interesting].sum()):8d}/{n_int} ({recall * 100:3.0f}%) "
              f"{int(fires[resting].sum()):12d}/{n_rest} ({false_alarm * 100:3.0f}%) {j:+6.2f}")
    print(f"  -> best separation at threshold {best[0]} (J={best[1]:+.2f})"
          + ("  — J<=0 means no threshold beats a coin flip" if best[1] <= 0 else "") + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, default=OOF_PATH, help="crossval.py's cv_oof.npz")
    parser.add_argument("--arm", default="gru", help="which crossval arm's probabilities to use")
    args = parser.parse_args()

    P, y = load_oof(args.oof, args.arm)
    print(f"Out-of-fold probabilities for {len(y)} clips from {args.oof} ({args.arm} arm)")

    resting = np.isin(y, RESTING_IDX)
    interesting = ~resting
    print(f"\n{int(interesting.sum())} interesting clips, {int(resting.sum())} resting clips\n")

    degenerate = check_degenerate(P, y)

    # The gate this replaced, for reference: argmax non-normal at >= 70%.
    old = (~np.isin(P.argmax(axis=1), RESTING_IDX)) & (P.max(axis=1) * 100 >= 70)
    print("BASELINE  pre-interest gate (argmax not resting AND confidence >= 70)")
    print(f"  caught {int(old[interesting].sum())}/{int(interesting.sum())} interesting  "
          f"({old[interesting].mean() * 100:.0f}% recall)   "
          f"false alarms {int(old[resting].sum())}/{int(resting.sum())}\n")

    deployed = np.array([interest_score(p) for p in P], dtype=float)
    report("agent/capture.py interest_score()  <-- what ships", deployed, interesting, resting)

    # Two reference policies, for comparison against whatever is deployed.
    report("1 - P(resting)  (maximum recall)",
           (1 - P[:, RESTING_IDX].sum(axis=1)) * 100, interesting, resting)
    report("top non-resting (requires commitment to one behavior)",
           np.max(np.delete(P, RESTING_IDX, axis=1), axis=1) * 100, interesting, resting)

    if degenerate:
        sys.exit(1)


if __name__ == "__main__":
    main()
