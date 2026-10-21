"""ONNX Runtime session for the frame embedder (per-frame, streaming).
Runs once per incoming frame; the result is pushed to predictor.py's ring buffer.
"""
import numpy as np
import onnxruntime as ort


class Backbone:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def embed(self, pixel_values: np.ndarray) -> np.ndarray:
        """pixel_values: [3,224,224] or [N,3,224,224] float32, ImageNet-normalized.
        Returns [1280] or [N,1280] float32 embedding(s)."""
        batch = pixel_values if pixel_values.ndim == 4 else pixel_values[np.newaxis, ...]
        out = self.session.run(None, {"pixel_values": batch.astype(np.float32)})[0]
        return out if pixel_values.ndim == 4 else out[0]
