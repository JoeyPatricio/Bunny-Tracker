"""Confirms inference/predictor.py reproduces the batched ONNX pipeline (the
training-eval-path vs runtime-inference-path parity check).

Runs after promote, against what is actually deployed in models/. For every
cached clip's eval sample (frames/<clip>.eval<W>.npy, the frames crossval.py
and evaluate.py's G4 use), it computes class probabilities two ways:

  flat      the int8 backbone over all W frames in one batch, then head.onnx
  runtime   inference/predictor.py's ring buffer, fed one frame at a time,
            exactly as the agent does

and requires every clip's probabilities to match within 1e-5 and the argmax
to agree. This used to compare one accuracy against evaluate.py's
quant_result.json; two pipelines can reach the same accuracy while predicting
different clips, so per-clip probabilities are the stronger check, and it no
longer depends on a file from another step.

Needs a schema-2 cache with an eval sample for the deployed window.

Usage (from server_py/, training venv, after promote):
    .venv-train/bin/python -m training.verify_predictor_parity [--models-dir DIR] [--cache-dir DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.backbone import Backbone
from inference.predictor import Predictor
from inference.temporal_head import TemporalHead
from shared.frames import normalize
from shared.labels import LABELS
from training.cache import DEFAULT_CACHE_DIR, load_manifest

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TOLERANCE = 1e-5


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    manifest = load_manifest(args.cache_dir)
    promoted = args.models_dir / "promoted.json"
    window = None
    if promoted.is_file():
        window = (json.loads(promoted.read_text()).get("manifest") or {}).get("window")
    window = int(window or manifest["window"])

    backbone = Backbone(str(args.models_dir / "backbone_int8.onnx"))
    head = TemporalHead(str(args.models_dir / "head.onnx"))

    worst, disagree, n = 0.0, [], 0
    for clip in manifest["clips"]:
        if clip["label"] not in LABELS:
            continue
        frames = np.load(args.cache_dir / "frames" / clip["eval"][str(window)])
        pixels = np.stack([normalize(f) for f in frames]).astype(np.float32)
        flat_logits = head.session.run(None, {"embeddings": backbone.embed(pixels)[np.newaxis]})[0][0]
        flat = _softmax(flat_logits)

        predictor = Predictor(backbone=backbone, head=head, window=window)
        runtime = None
        for p in pixels:
            runtime = predictor.push_frame(p)
        runtime = np.asarray(runtime)

        worst = max(worst, float(np.abs(flat - runtime).max()))
        if int(flat.argmax()) != int(runtime.argmax()):
            disagree.append(clip["filename"])
        n += 1

    ok = worst <= TOLERANCE and not disagree
    print(f"{n} clips, window {window}: max |flat - predictor| probability {worst:.2e} "
          f"(tolerance {TOLERANCE:g}), argmax disagreements {len(disagree)}")
    for fn in disagree[:10]:
        print(f"  disagree: {fn}")
    print("MATCH: predictor.py reproduces the batched pipeline" if ok
          else "MISMATCH: predictor.py does NOT reproduce the batched pipeline")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
