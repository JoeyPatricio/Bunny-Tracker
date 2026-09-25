"""ONNX Runtime session for the temporal classifier (runs on the embedding ring buffer)."""
import numpy as np
import onnxruntime as ort


class TemporalHead:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    @property
    def num_classes(self) -> int:
        """Width of the output axis, i.e. how many classes this head was
        trained for. Read from the graph rather than assumed, so a head from
        before an ethogram change can be caught at load instead of quietly
        indexing the wrong names."""
        shape = self.session.get_outputs()[0].shape
        return int(shape[-1])

    def classify(self, embeddings: np.ndarray) -> np.ndarray:
        """embeddings: [N,1280] float32. Returns [num_classes] logits."""
        window = embeddings[np.newaxis, ...].astype(np.float32)  # [1,N,1280]
        logits = self.session.run(None, {"embeddings": window})[0]
        return logits[0]
