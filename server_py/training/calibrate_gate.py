"""Pick AGENT_INTEREST_THRESHOLD from data instead of intuition.

Runs the DEPLOYED artifacts (models/backbone_int8.onnx + models/head.onnx -
the exact pair agent/capture.py loads) over every human-labeled clip, scores
each one with agent/capture.py's own interest_score(), and reports what each
threshold would have caught and what it would have woken you up for.

It imports interest_score rather than reimplementing it, so changing the alert
policy automatically changes what this measures. Runs on the RUNTIME venv
(onnxruntime, no torch), which is the point: it exercises the same stack the
agent does, including int8 quantization, which training/evaluate.py's fp32 path
does not.

The unit here is one clip = one window = one candidate decision. The live gate
additionally requires AGENT_ALERT_STREAK consecutive candidate frames and a
motion floor, so real-world firing is strictly rarer than the recall column.
Read the numbers as an upper bound and a policy comparison, not a forecast.

Usage (from server_py/, runtime venv):
    .venv/bin/python -m training.calibrate_gate [--refresh] [--window N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.capture import interest_score
from inference.backbone import Backbone
from inference.predictor import Predictor, softmax
from inference.temporal_head import TemporalHead
from shared.frames import extract_frames, normalize
from shared.labels import LABELS, RESTING_INDICES

ROOT = Path(__file__).resolve().parent.parent.parent  # BunnyTracker/
RECORDINGS_DIR = ROOT / "server" / "recordings"
LABELS_JSON = ROOT / "server" / "labels.json"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CACHE_PATH = MODELS_DIR / "cache" / "gate_probs.npz"

# v1 had one resting class; v2 has two, so "resting" is a set of indices and
# P(resting) is their sum rather than a single column.
RESTING_IDX = sorted(RESTING_INDICES)
THRESHOLDS = (10, 20, 25, 30, 40, 50, 60, 70, 80)


def compute_probs(window: int) -> tuple[np.ndarray, np.ndarray]:
    """One softmax window per labeled clip, through the deployed ONNX pair."""
    labels = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    items = sorted((fn, lab) for fn, lab in labels.items() if lab in LABELS)
    if not items:
        raise SystemExit("No human-confirmed labels in server/labels.json — nothing to calibrate against.")

    backbone = Backbone(str(MODELS_DIR / "backbone_int8.onnx"))
    head = TemporalHead(str(MODELS_DIR / "head.onnx"))

    rows, targets = [], []
    started = time.time()
    for i, (filename, label) in enumerate(items, 1):
        path = RECORDINGS_DIR / filename
        if not path.is_file():
            print(f"  [{i}/{len(items)}] {filename} — labeled but not on disk, skipping")
            continue
        predictor = Predictor(backbone=backbone, head=head, window=window)
        probs = None
        for frame in extract_frames(str(path), n=window):
            probs = predictor.push_frame(normalize(frame))
        rows.append(probs)
        targets.append(LABELS.index(label))
        p_rest = float(probs[RESTING_IDX].sum())
        print(f"  [{i}/{len(items)}] {filename[:44]:44s} {label:16s} P(resting)={p_rest:.2f}", flush=True)

    print(f"  ({time.time() - started:.0f}s)")
    P, y = np.array(rows), np.array(targets)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, P=P, y=y)
    return P, y


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
        print(f"\n  ⚠ At or below base rate — the model is no better than always guessing the\n"
              f"    most common class. Treat the tables below as untrustworthy.\n")
    return False


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
    parser.add_argument("--window", type=int, default=8, help="frames per window; must match the deployed head")
    parser.add_argument("--refresh", action="store_true", help="re-run the model instead of using the cache")
    args = parser.parse_args()

    if CACHE_PATH.is_file() and not args.refresh:
        cached = np.load(CACHE_PATH)
        P, y = cached["P"], cached["y"]
        print(f"Loaded cached probabilities for {len(y)} clips (--refresh to recompute)\n")
    else:
        print("Running the deployed ONNX pair over every labeled clip...")
        P, y = compute_probs(args.window)

    resting = np.isin(y, RESTING_IDX)
    interesting = ~resting
    print(f"\n{int(interesting.sum())} interesting clips, {int(resting.sum())} resting clips\n")

    degenerate = check_degenerate(P, y)

    # The gate this replaced, for reference: argmax non-normal at >= 70%.
    old = (~np.isin(P.argmax(axis=1), RESTING_IDX)) & (P.max(axis=1) * 100 >= 70)
    print(f"BASELINE  pre-interest gate (argmax not resting AND confidence >= 70)")
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
