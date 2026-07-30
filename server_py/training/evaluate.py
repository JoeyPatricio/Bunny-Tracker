"""Held-out accuracy + confusion matrix for float32 and int8, compared against
the JS baseline. This is the hard accuracy gate (section 3.5): the retrained
model must meet or beat the baseline, especially on the temporal classes
(zoomies, yawn) that mean/std pooling handles worst.

Usage (from server_py/, training venv, after export_onnx.py has run):
    .venv-train/Scripts/python.exe -m training.evaluate
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


def run_onnx_pipeline(val_clips: list[dict], backbone_path: Path, head_path: Path):
    backbone_sess = ort.InferenceSession(str(backbone_path), providers=["CPUExecutionProvider"])
    head_sess = ort.InferenceSession(str(head_path), providers=["CPUExecutionProvider"])

    true_ids, pred_ids = [], []
    for entry in val_clips:
        path = RECORDINGS_DIR / entry["filename"]
        frames = extract_frames(str(path), n=8)
        batch = np.stack([normalize(f) for f in frames]).astype(np.float32)  # [8,3,224,224]
        embeddings = backbone_sess.run(None, {"pixel_values": batch})[0]     # [8,1280]
        window = embeddings[np.newaxis, ...].astype(np.float32)             # [1,8,1280]
        logits = head_sess.run(None, {"embeddings": window})[0]             # [1,5]
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

    baseline_path = MODELS_DIR / "js_baseline.json"
    if not baseline_path.exists():
        print(f"ERROR: {baseline_path} missing. Run scripts/js_baseline.mjs first.")
        sys.exit(1)
    baseline = json.loads(baseline_path.read_text())

    train_result = json.loads((MODELS_DIR / "train_result.json").read_text())

    print_report("JS baseline (current live model)", baseline["accuracy"], np.array(baseline["confusion_matrix"]))
    print_report("Python retrain, float32 (torch head eval, Phase 1 train.py)", train_result["val_acc"], np.array(train_result["confusion_matrix"]))

    fp32_acc, fp32_cm = run_onnx_pipeline(val_clips, MODELS_DIR / "backbone_fp32.onnx", MODELS_DIR / "head.onnx")
    print_report("Python retrain, ONNX float32 (backbone.onnx fp32 + head.onnx)", fp32_acc, fp32_cm)

    int8_acc, int8_cm = run_onnx_pipeline(val_clips, MODELS_DIR / "backbone_int8.onnx", MODELS_DIR / "head.onnx")
    print_report("Python retrain, ONNX int8 (backbone_int8.onnx + head.onnx)", int8_acc, int8_cm)

    gate_result = {
        "js_baseline_acc": baseline["accuracy"],
        "torch_fp32_acc": train_result["val_acc"],
        "onnx_fp32_acc": fp32_acc,
        "onnx_int8_acc": int8_acc,
        "gate_passed": int8_acc >= baseline["accuracy"],
    }
    (MODELS_DIR / "gate_result.json").write_text(json.dumps(gate_result, indent=2))

    print(f"\n{'='*60}")
    print(f"ACCURACY GATE: retrained int8 must meet or beat JS baseline")
    print(f"  JS baseline:  {baseline['accuracy']*100:.1f}%")
    print(f"  Retrain int8: {int8_acc*100:.1f}%")
    print(f"  Gate {'PASSED' if gate_result['gate_passed'] else 'FAILED'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
