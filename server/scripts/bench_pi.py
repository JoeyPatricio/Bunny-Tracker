"""Per-step CPU latency benchmark for the promoted ONNX deployment.

Measures isolated inference on the current host. Camera encoding, streaming,
and concurrent application load need a separate capture smoke test.

Usage (from server/, runtime venv, after model promotion):
    .venv/bin/python -m scripts.bench_pi
"""
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.deployment import read_promoted

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ITER = 50
WARMUP = 5

# EMBEDDING_DIM used to be imported from training/model.py, which imports torch
# — a package requirements.txt deliberately excludes, since the runtime venv is
# what ships to the Pi. That made this script crash on import on the one machine
# it exists to measure. Read the dimension off the deployed head graph instead:
# it needs no torch, and it is right by construction for the artifacts actually
# on disk rather than for whatever the training code last declared.
FALLBACK_EMBEDDING_DIM = 1280


def embedding_dim(head_path: Path) -> int:
    """Trailing input dimension of head.onnx ([batch, window, EMBEDDING_DIM])."""
    if not head_path.exists():
        return FALLBACK_EMBEDDING_DIM
    sess = ort.InferenceSession(str(head_path), providers=["CPUExecutionProvider"])
    dim = sess.get_inputs()[0].shape[-1]
    # Only batch and window are exported dynamic; a symbolic name here would
    # mean the export changed shape, so fall back rather than crash on a str.
    return dim if isinstance(dim, int) else FALLBACK_EMBEDDING_DIM


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
    promoted = read_promoted(MODELS_DIR)
    if promoted is None:
        raise ValueError("No promoted model metadata. Promote a model before benchmarking inference.")
    window = promoted["manifest"]["window"]
    embed_dim = embedding_dim(MODELS_DIR / "head.onnx")

    print(f"CPU inference benchmark on {platform.node()} ({platform.machine()}).")
    print(f"CPU execution provider, window={window}\n")

    bench_session(
        "backbone fp32", MODELS_DIR / "backbone_fp32.onnx",
        lambda: np.random.randn(1, 3, 224, 224).astype(np.float32),
    )
    backbone_int8_times = bench_session(
        "backbone int8", MODELS_DIR / "backbone_int8.onnx",
        lambda: np.random.randn(1, 3, 224, 224).astype(np.float32),
    )
    head_times = bench_session(
        "head", MODELS_DIR / "head.onnx",
        lambda: np.random.randn(1, window, embed_dim).astype(np.float32),
    )

    if backbone_int8_times is not None and head_times is not None:
        per_step = np.median(backbone_int8_times) + np.median(head_times)
        print(f"\nEstimated per-frame step (int8 backbone + head): {per_step:.2f}ms")
        print("Default agent frame budget: 1500ms (AGENT_FRAME_SECONDS)")
        print(f"Estimated headroom: {1500/per_step:.1f}x default budget (isolated inference only)")


if __name__ == "__main__":
    main()
