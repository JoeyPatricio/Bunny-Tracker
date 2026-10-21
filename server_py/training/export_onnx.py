"""Export the frozen backbone + trained head to ONNX, then int8-quantize the
backbone (the expensive half) with a calibration set of real frames.

Writes to models/candidate/, never to models/ in place. The agent and the
server read models/*.onnx directly, so exporting in place WAS the deploy, and
that is how a model with `"gate_passed": false` went live. Now an export only
produces a candidate plus candidate/manifest.json (sha256 of every artifact,
git SHA, labels, ethogram version); training/promote.py is the only thing that
moves a candidate into models/, and it refuses unless the gate passed.

Usage (from server_py/, training venv):
    .venv-train/bin/python -m training.export_onnx [--out DIR]
"""
import argparse
import json
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.frames import extract_frames, normalize
from shared.labels import ETHOGRAM_VERSION, LABELS
from training.artifacts import MANIFEST_NAME, artifacts_in, sha256_file
from training.model import EMBEDDING_DIM, TemporalHead, build_backbone

ROOT = Path(__file__).resolve().parent.parent.parent  # BunnyTracker/
RECORDINGS_DIR = ROOT / "server" / "recordings"
LABELS_JSON = ROOT / "server" / "labels.json"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
HEAD_STATE_PATH = MODELS_DIR / "head_state.pt"
CANDIDATE_DIR = MODELS_DIR / "candidate"

# Module globals rather than arguments because tests repoint them (see
# training/test_export_guards.py). set_output_dir() moves all three together.
BACKBONE_FP32 = CANDIDATE_DIR / "backbone_fp32.onnx"
BACKBONE_INT8 = CANDIDATE_DIR / "backbone_int8.onnx"
HEAD_ONNX = CANDIDATE_DIR / "head.onnx"
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
        dynamo=False,  # see export_head() for why this is pinned
    )
    print(f"Exported {BACKBONE_FP32}")


def export_head(window: int) -> None:
    head = TemporalHead()
    # Deliberately strict. After an ethogram change the checkpoint on disk has
    # the old class count and this raises a size mismatch naming both shapes.
    # strict=False would "succeed" by silently leaving the final layer at its
    # random init, exporting a model that predicts noise with full confidence.
    # Failing here is the correct outcome until a retrain lands.
    head.load_state_dict(torch.load(HEAD_STATE_PATH, weights_only=True))
    head.eval()
    dummy = torch.randn(1, window, EMBEDDING_DIM)
    # dynamo=False pins the legacy TorchScript exporter. torch 2.14 defaults
    # to dynamo=True, which on this model does two unwanted things: it rejects
    # a dynamic `window` axis outright ("tracing inferred a static shape of 8"),
    # and with a batch-1 example input it specializes the batch axis to a
    # literal 1. That is where the deployed head.onnx's static [1, 5] output
    # came from, and a statically-batched head cannot be evaluated in batches.
    # Verified: legacy gives embeddings ['batch','window',1280] -> logits
    # ['batch', len(LABELS)]. tests/test_export_guards.py asserts it.
    torch.onnx.export(
        head, dummy, str(HEAD_ONNX),
        input_names=["embeddings"], output_names=["logits"],
        dynamic_axes={"embeddings": {0: "batch", 1: "window"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported {HEAD_ONNX}")


def pick_calibration_frames(seed: int = 0) -> list[np.ndarray]:
    labels = json.loads(LABELS_JSON.read_text())
    # Filter on LABELS, the way extract_frames.py and calibrate_gate.py both
    # do. `out_of_view` is a valid label but not a model class, and calibrating
    # int8 activation ranges on footage with no rabbit in it means the quantized
    # backbone's dynamic range is partly set by frames the model will never be
    # asked to classify.
    filenames = [fn for fn, label in labels.items() if label in LABELS]
    if not filenames:
        raise SystemExit(
            f"No clips with a model-class label in {LABELS_JSON}. int8 "
            f"calibration needs real footage; label some clips first."
        )
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


def set_output_dir(out: Path) -> None:
    global BACKBONE_FP32, BACKBONE_INT8, HEAD_ONNX
    BACKBONE_FP32 = out / "backbone_fp32.onnx"
    BACKBONE_INT8 = out / "backbone_int8.onnx"
    HEAD_ONNX = out / "head.onnx"


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def validate_train_result(result: dict) -> None:
    if result.get("view", "full") != "full":
        raise SystemExit("Crop is an offline CV arm. The agent has no matching ROI preprocessing.")
    if result.get("mode") != "all-clips" or not result.get("cv_run_id"):
        raise SystemExit("Export requires a CV-selected all-clips head. Run training.train --all-clips first.")
    if any(key not in result for key in ("cv_arm", "aug_policy", "view", "labels_sha256")):
        raise SystemExit("Training result lacks per-arm provenance. Rerun training.train --all-clips.")
    if result.get("ethogram_version") != ETHOGRAM_VERSION:
        raise SystemExit("The head uses a different ethogram. Recode and retrain before export.")
    if result.get("labels_sha256") != sha256_file(LABELS_JSON):
        raise SystemExit("Labels changed after training. Rebuild the cache, cross-validate and retrain before export.")


def write_manifest(out: Path, window: int | None = None, train_result: dict | None = None) -> dict:
    manifest = {
        "artifacts": {p.name: sha256_file(p) for p in artifacts_in(out)},
        "git_sha": git_sha(),
        "labels": list(LABELS),
        "ethogram_version": ETHOGRAM_VERSION,
        "window": window,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    if train_result is not None:
        manifest.update({key: train_result[key] for key in (
            "cv_arm", "cv_run_id", "labels_sha256", "aug_policy", "view",
        )})
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=CANDIDATE_DIR,
                        help="candidate directory (default models/candidate/)")
    args = parser.parse_args()

    out = args.out.resolve()
    if out == MODELS_DIR.resolve():
        raise SystemExit("Export cannot write into the live models directory. Use models/candidate and promote after evaluation.")
    train_result = json.loads((MODELS_DIR / "train_result.json").read_text())
    validate_train_result(train_result)
    window = train_result["window"]

    out.mkdir(parents=True, exist_ok=True)
    # A stale gate result from an earlier candidate must never be read as
    # applying to this one.
    (out / "gate_result.json").unlink(missing_ok=True)
    set_output_dir(out)
    export_backbone(window)
    export_head(window)
    quantize_backbone()
    manifest = write_manifest(out, window, train_result)
    print(f"\nDone. Candidate in {out} ({len(manifest['artifacts'])} artifacts hashed).")
    print("Nothing is deployed yet: run training.evaluate, then training.promote.")


if __name__ == "__main__":
    main()
