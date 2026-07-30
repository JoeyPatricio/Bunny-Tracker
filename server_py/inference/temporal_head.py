"""ONNX Runtime session for the temporal classifier (runs on the embedding ring buffer)."""
import numpy as np
import onnxruntime as ort


class TemporalHead:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def classify(self, embeddings: np.ndarray) -> np.ndarray:
        """embeddings: [N,1280] float32. Returns [num_classes] logits."""
        window = embeddings[np.newaxis, ...].astype(np.float32)  # [1,N,1280]
        logits = self.session.run(None, {"embeddings": window})[0]
        return logits[0]
