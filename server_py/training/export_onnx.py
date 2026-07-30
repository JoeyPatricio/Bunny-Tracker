"""Export the frozen backbone + trained head to ONNX, then int8-quantize the
backbone (the expensive half) with a calibration set of real frames.
Section 3.4 of python-port-plan.md.

Usage (from server_py/, training venv):
    .venv-train/Scripts/python.exe -m training.export_onnx
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.frames import extract_frames, normalize
from training.model import EMBEDDING_DIM, TemporalHead, build_backbone

ROOT = Path(__file__).resolve().parent.parent.parent  # BunnyTracker/
RECORDINGS_DIR = ROOT / "server" / "recordings"
LABELS_JSON = ROOT / "server" / "labels.json"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
HEAD_STATE_PATH = MODELS_DIR / "head_state.pt"

BACKBONE_FP32 = MODELS_DIR / "backbone_fp32.onnx"
BACKBONE_INT8 = MODELS_DIR / "backbone_int8.onnx"
HEAD_ONNX = MODELS_DIR / "head.onnx"
CALIBRATION_CLIPS = 20
CALIBRATION_FRAMES_PER_CLIP = 8


def export_backbone(window: int) -> None:
    backbone = build_backbone(pretrained=True)
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        backbone, dummy, str(BACKBONE_FP32),
        input_names=["pixel_values"], output_names=["embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported {BACKBONE_FP32}")


def export_head(window: int) -> None:
    head = TemporalHead()
    head.load_state_dict(torch.load(HEAD_STATE_PATH, weights_only=True))
    head.eval()
    dummy = torch.randn(1, window, EMBEDDING_DIM)
    torch.onnx.export(
        head, dummy, str(HEAD_ONNX),
        input_names=["embeddings"], output_names=["logits"],
        dynamic_axes={"embeddings": {0: "batch", 1: "window"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported {HEAD_ONNX}")


def pick_calibration_frames(seed: int = 0) -> list[np.ndarray]:
    labels = json.loads(LABELS_JSON.read_text())
    filenames = list(labels.keys())
    random.Random(seed).shuffle(filenames)
    frames = []
    for fn in filenames[:CALIBRATION_CLIPS]:
        path = RECORDINGS_DIR / fn
        try:
            clip_frames = extract_frames(str(path), n=CALIBRATION_FRAMES_PER_CLIP)
        except ValueError:
            continue
        frames.extend(clip_frames)
    print(f"Calibration set: {len(frames)} frames from {CALIBRATION_CLIPS} clips")
    return frames


class BackboneCalibrationReader(CalibrationDataReader):
    def __init__(self, frames: list[np.ndarray]):
        self._data = iter(normalize(f)[np.newaxis, ...].astype(np.float32) for f in frames)

    def get_next(self):
        item = next(self._data, None)
        if item is None:
            return None
        return {"pixel_values": item}


def quantize_backbone() -> None:
    frames = pick_calibration_frames()
    reader = BackboneCalibrationReader(frames)
    quantize_static(
        model_input=str(BACKBONE_FP32),
        model_output=str(BACKBONE_INT8),
        calibration_data_reader=reader,
        weight_type=QuantType.QInt8,
        # Per-tensor weight quantization is a known bad fit for MobileNet-style
        # depthwise convolutions (section 6, risk 2's "large int8 drop" case) —
        # per-channel is the plan's named fallback.
        per_channel=True,
    )
    print(f"Quantized -> {BACKBONE_INT8}")


def main():
    manifest = json.loads((MODELS_DIR / "cache" / "manifest.json").read_text())
    train_result = json.loads((MODELS_DIR / "train_result.json").read_text())
    window = train_result["window"]

    MODELS_DIR.mkdir(exist_ok=True)
    export_backbone(window)
    export_head(window)
    quantize_backbone()
    print("\nDone. Float32 and int8 backbone both kept for the accuracy gate comparison.")


if __name__ == "__main__":
    main()
