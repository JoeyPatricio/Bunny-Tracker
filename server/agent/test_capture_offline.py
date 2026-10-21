"""Offline harness for AgentDecisionLoop (Phase 3 step 5): replays the same
motion-parity fixture frames through the full decision loop (real ONNX
backbone + head from Phase 1) and confirms it runs without crashing and
produces sane streak/cooldown/warm-up behavior. Not a formal parity gate —
the plan calls this control flow "no numeric risk."

Usage (from server/, runtime venv):
    .venv/Scripts/python.exe -m agent.test_capture_offline
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.capture import AgentDecisionLoop
from inference.backbone import Backbone
from inference.deployment import read_promoted
from inference.temporal_head import TemporalHead
from shared.labels import LABELS

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def main():
    promoted = read_promoted(MODELS_DIR)
    if promoted is None:
        raise SystemExit("No promoted model. Recode, retrain, evaluate and promote before the offline smoke run.")
    backbone = Backbone(str(MODELS_DIR / "backbone_int8.onnx"))
    head = TemporalHead(str(MODELS_DIR / "head.onnx"))
    if head.num_classes != len(LABELS):
        raise SystemExit("The deployed head does not match the ethogram. Retrain and promote before replay.")
    loop = AgentDecisionLoop(backbone, head, embed_window=promoted["manifest"]["window"])

    frames = np.load(FIXTURES_DIR / "motion_fixture_frames.npy")
    print(f"Replaying {len(frames)} fixture frames through AgentDecisionLoop...")

    actions = []
    for i, frame in enumerate(frames):
        # Space fake timestamps 1.5s apart so cooldown math is meaningful,
        # rather than everything landing at the same instant.
        decision = loop.handle_frame(frame, now=i * 1.5)
        tag = ""
        if decision["action"] != "none":
            tag = f"  <-- {decision['action']}"
            actions.append((i, decision["action"]))
        warm_tag = "" if decision["warm"] else f" (warmup {decision['prediction']['windowFill']}/{loop.embed_window})"
        print(
            f"  frame {i:2d}  motion={decision['motion']['level']:5.1f}"
            f"  cells={decision['motion']['hotCells']:2d}"
            f"{'  LIGHTING' if decision['motion']['lighting'] else ''}"
            f"  -> {decision['prediction']['label']}{warm_tag}{tag}"
        )

    print(f"\nCompleted {len(frames)} frames without error. {len(actions)} capture/alert action(s) fired: {actions}")


if __name__ == "__main__":
    main()
