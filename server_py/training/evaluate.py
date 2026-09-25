"""Accuracy and confusion matrix for the exported fp32 and int8 ONNX pair.

IMPORTANT: the numbers printed here are measured on the validation split, which
also drove early stopping and checkpoint selection. They are selection scores,
NOT held-out accuracy, and must not be reported as accuracy. Grouped
cross-validation replaces this in Phase 3 (docs/upgrade-plan.md).

The JS-baseline accuracy gate this file used to carry has been retired: the
tfjs model predicts five ethogram-v1 classes and cannot predict v2's seven, so
the comparison was meaningless. What remains is a quantization check.

Usage (from server_py/, training venv, after export_onnx.py has run):
    .venv-train/bin/python -m training.evaluate
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.frames import extract_frames, normalize
from shared.labels import LABELS

ROOT = Path(__file__).resolve().parent.parent.parent  # BunnyTracker/
RECORDINGS_DIR = ROOT / "server" / "recordings"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# How far int8 may fall behind its own fp32 export before it counts as the
# depthwise-convolution collapse export_onnx.py's per_channel=True guards
# against.
INT8_TOLERANCE = 0.03


def confusion_matrix(true_ids, pred_ids) -> np.ndarray:
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for t, p in zip(true_ids, pred_ids):
        cm[t, p] += 1
    return cm


def per_class_report(cm: np.ndarray) -> dict:
    report = {}
    for k, label in enumerate(LABELS):
        tp = cm[k, k]
        support = cm[k].sum()
        pred_total = cm[:, k].sum()
        report[label] = {
            "precision": tp / pred_total if pred_total else 0.0,
            "recall": tp / support if support else 0.0,
            "support": int(support),
        }
    return report


def run_onnx_pipeline(val_clips: list[dict], backbone_path: Path, head_path: Path,
                      n_frames: int):
    """n_frames comes from the manifest's `window`, never a literal: hardcoding
    8 here meant any retrain with --window N != 8 was silently evaluated at the
    wrong window length and still printed a confident accuracy.
    verify_predictor_parity.py already reads it from the manifest."""
    backbone_sess = ort.InferenceSession(str(backbone_path), providers=["CPUExecutionProvider"])
    head_sess = ort.InferenceSession(str(head_path), providers=["CPUExecutionProvider"])

    true_ids, pred_ids = [], []
    for entry in val_clips:
        path = RECORDINGS_DIR / entry["filename"]
        frames = extract_frames(str(path), n=n_frames)
        batch = np.stack([normalize(f) for f in frames]).astype(np.float32)  # [N,3,224,224]
        embeddings = backbone_sess.run(None, {"pixel_values": batch})[0]     # [N,1280]
        window = embeddings[np.newaxis, ...].astype(np.float32)             # [1,N,1280]
        logits = head_sess.run(None, {"embeddings": window})[0]             # [1,len(LABELS)]
        pred = int(np.argmax(logits[0]))
        true_ids.append(LABELS.index(entry["label"]))
        pred_ids.append(pred)
    cm = confusion_matrix(true_ids, pred_ids)
    acc = sum(t == p for t, p in zip(true_ids, pred_ids)) / len(true_ids)
    return acc, cm


def print_report(name: str, acc: float, cm: np.ndarray):
    print(f"\n{name}: {acc*100:.1f}% overall accuracy")
    report = per_class_report(cm)
    for label, r in report.items():
        print(f"  {label:10s} precision={r['precision']*100:5.1f}%  recall={r['recall']*100:5.1f}%  support={r['support']}")


def main():
    manifest = json.loads((MODELS_DIR / "cache" / "manifest.json").read_text())
    val_clips = manifest["val"]
    n_frames = manifest["window"]

    train_result = json.loads((MODELS_DIR / "train_result.json").read_text())

    print_report("Python retrain, float32 (torch head eval, train.py)", train_result["val_acc"], np.array(train_result["confusion_matrix"]))

    fp32_acc, fp32_cm = run_onnx_pipeline(val_clips, MODELS_DIR / "backbone_fp32.onnx", MODELS_DIR / "head.onnx", n_frames)
    print_report("Python retrain, ONNX float32 (backbone.onnx fp32 + head.onnx)", fp32_acc, fp32_cm)

    int8_acc, int8_cm = run_onnx_pipeline(val_clips, MODELS_DIR / "backbone_int8.onnx", MODELS_DIR / "head.onnx", n_frames)
    print_report("Python retrain, ONNX int8 (backbone_int8.onnx + head.onnx)", int8_acc, int8_cm)

    # QUANTIZATION CHECK ONLY. The old gate here compared int8 accuracy against
    # models/js_baseline.json, a v1 tfjs artifact over five classes that cannot
    # predict ethogram v2's seven. That comparison is retired, not repaired;
    # see docs/upgrade-plan.md. What survives is the one question these two
    # numbers can still answer honestly: did int8 quantization cost accuracy
    # relative to the fp32 export of the same weights?
    #
    # This is deliberately NOT called gate_passed. A real gate lands in Phase 3
    # (evaluate.py becomes G1-G7 and promote.py enforces it); until then
    # nothing here blocks a deploy, and pretending otherwise is how the
    # previous gate came to report failure while the model shipped anyway.
    delta = fp32_acc - int8_acc
    quant_result = {
        "torch_fp32_acc": train_result["val_acc"],
        "onnx_fp32_acc": fp32_acc,
        "onnx_int8_acc": int8_acc,
        "int8_delta": delta,
        "int8_within_tolerance": delta <= INT8_TOLERANCE,
        "note": "Quantization check only, on the selection split. Not a held-out "
                "estimate and not a deploy gate. See docs/upgrade-plan.md Phase 3.",
    }
    (MODELS_DIR / "quant_result.json").write_text(json.dumps(quant_result, indent=2))

    print(f"\n{'='*60}")
    print("QUANTIZATION CHECK: int8 must not fall far behind its own fp32 export")
    print(f"  ONNX fp32:  {fp32_acc*100:.1f}%")
    print(f"  ONNX int8:  {int8_acc*100:.1f}%")
    print(f"  delta:      {delta*100:+.1f}pp (tolerance {INT8_TOLERANCE*100:.0f}pp)")
    print(f"  {'OK' if quant_result['int8_within_tolerance'] else 'REGRESSION - check per-channel quantization'}")
    print(f"{'='*60}")
    print("\nThese are selection-split numbers, not held-out accuracy.")
    print("Grouped cross-validation lands in Phase 3; see docs/upgrade-plan.md.")


if __name__ == "__main__":
    main()
