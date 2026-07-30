"""Confirms inference/predictor.py reproduces evaluate.py's numbers on the
same held-out windows (Phase 1 step 4: the training-eval-path vs
runtime-inference-path parity check).

Runs the exact same ONNX float32 pipeline evaluate.py's run_onnx_pipeline()
uses (backbone_fp32.onnx over freshly-extracted frames, then head.onnx), but
through inference/predictor.py's ring-buffer abstraction instead of a flat
loop, and checks the result matches evaluate.py's onnx_fp32_acc exactly.

Usage (from server_py/, training venv, after evaluate.py has run):
    .venv-train/Scripts/python.exe -m training.verify_predictor_parity
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.backbone import Backbone
from inference.predictor import Predictor
from inference.temporal_head import TemporalHead
from shared.frames import extract_frames, normalize
from shared.labels import LABELS

ROOT = Path(__file__).resolve().parent.parent.parent  # BunnyTracker/
RECORDINGS_DIR = ROOT / "server" / "recordings"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    manifest = json.loads((MODELS_DIR / "cache" / "manifest.json").read_text())
    val_entries = manifest["val"]
    window = manifest["window"]

    backbone = Backbone(str(MODELS_DIR / "backbone_fp32.onnx"))
    head = TemporalHead(str(MODELS_DIR / "head.onnx"))

    true_ids, pred_ids = [], []
    for e in val_entries:
        path = RECORDINGS_DIR / e["filename"]
        frames = extract_frames(str(path), n=window)
        predictor = Predictor(backbone=backbone, head=head, window=window)
        probs = None
        for f in frames:
            probs = predictor.push_frame(normalize(f))
        pred = int(np.argmax(probs))
        true_ids.append(LABELS.index(e["label"]))
        pred_ids.append(pred)

    acc = sum(t == p for t, p in zip(true_ids, pred_ids)) / len(true_ids)

    gate_result = json.loads((MODELS_DIR / "gate_result.json").read_text())
    onnx_fp32_acc = gate_result["onnx_fp32_acc"]
    match = np.isclose(acc, onnx_fp32_acc, atol=1e-9)

    print(f"predictor.py (ring buffer) accuracy on val set: {acc*100:.1f}%")
    print(f"evaluate.py ONNX float32 accuracy (same backbone.onnx + head.onnx): {onnx_fp32_acc*100:.1f}%")
    print(f"{'MATCH' if match else 'MISMATCH'} — predictor.py {'reproduces' if match else 'does NOT reproduce'} evaluate.py's numbers")

    if not match:
        sys.exit(1)


if __name__ == "__main__":
    main()
