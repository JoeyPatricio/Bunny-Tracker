"""Per-step latency benchmark for backbone+head ONNX (section 3.6).

No Raspberry Pi 5 is owned yet (see notes/environment.md), so this runs on
the dev box as the closest available signal, NOT the real Pi gate. Numbers
here are reported as dev-box-only until re-run on actual hardware.

Usage (from server_py/, training or runtime venv, after export_onnx.py):
    .venv-train/Scripts/python.exe -m scripts.bench_pi
"""
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.model import EMBEDDING_DIM

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ITER = 50
WARMUP = 5


def bench_session(name: str, onnx_path: Path, make_input):
    if not onnx_path.exists():
        print(f"  {name}: SKIPPED ({onnx_path.name} not found)")
        return None
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    for _ in range(WARMUP):
        sess.run(None, {input_name: make_input()})

    times = []
    for _ in range(ITER):
        x = make_input()
        t0 = time.perf_counter()
        sess.run(None, {input_name: x})
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    print(f"  {name}: mean={times.mean():.2f}ms  p50={np.median(times):.2f}ms  p95={np.percentile(times, 95):.2f}ms")
    return times


def main():
    window = 8
    manifest_path = MODELS_DIR / "cache" / "manifest.json"
    if manifest_path.exists():
        import json
        window = json.loads(manifest_path.read_text())["window"]

    print(f"DEV-BOX benchmark only — NOT the Raspberry Pi 5 gate (no hardware owned yet).")
    print(f"CPU execution provider, window={window}\n")

    backbone_fp32_times = bench_session(
        "backbone fp32", MODELS_DIR / "backbone_fp32.onnx",
        lambda: np.random.randn(1, 3, 224, 224).astype(np.float32),
    )
    backbone_int8_times = bench_session(
        "backbone int8", MODELS_DIR / "backbone_int8.onnx",
        lambda: np.random.randn(1, 3, 224, 224).astype(np.float32),
    )
    head_times = bench_session(
        "head", MODELS_DIR / "head.onnx",
        lambda: np.random.randn(1, window, EMBEDDING_DIM).astype(np.float32),
    )

    if backbone_int8_times is not None and head_times is not None:
        per_step = np.median(backbone_int8_times) + np.median(head_times)
        print(f"\nEstimated per-frame step (int8 backbone + head): {per_step:.2f}ms")
        print(f"Current agent frame budget: 1500ms (AGENT_FRAME_SECONDS, per notes/environment.md)")
        print(f"Dev-box headroom: {1500/per_step:.1f}x budget (Pi CPU will be slower than this dev box — re-benchmark on real hardware before relying on this number)")


if __name__ == "__main__":
    main()
