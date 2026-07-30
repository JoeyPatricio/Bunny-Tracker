"""Ring buffer of last N embeddings -> softmax. The runtime classify()
replacement: embed each incoming frame once (expensive), classify the window
each step (cheap). See python-port-plan.md section 3.1.
"""
from collections import deque

import numpy as np

from inference.backbone import Backbone
from inference.temporal_head import TemporalHead


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


class Predictor:
    def __init__(self, backbone: Backbone, head: TemporalHead, window: int):
        self.backbone = backbone
        self.head = head
        self.window = window
        self.buffer: deque[np.ndarray] = deque(maxlen=window)

    def push_frame(self, pixel_values: np.ndarray) -> np.ndarray | None:
        """Embed one frame and push it to the ring buffer. Returns class
        probabilities once the buffer holds `window` embeddings, else None."""
        embedding = self.backbone.embed(pixel_values)
        self.buffer.append(embedding)
        if len(self.buffer) < self.window:
            return None
        return self.predict()

    def predict(self) -> np.ndarray:
        embeddings = np.stack(self.buffer)
        logits = self.head.classify(embeddings)
        return softmax(logits)

    def push_embedding(self, embedding: np.ndarray) -> np.ndarray | None:
        """Same as push_frame but for an already-computed embedding (used by
        evaluate.py/tests to reproduce numbers from cached embeddings)."""
        self.buffer.append(embedding)
        if len(self.buffer) < self.window:
            return None
        return self.predict()
