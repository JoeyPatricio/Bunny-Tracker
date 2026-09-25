"""Two-stage model: frozen timm backbone (per-frame embedder) + a small
learned temporal head (window classifier). See docs/provenance.md, python-port-plan.md section 3.1.

Two separate nn.Modules, not one end-to-end graph, so they can be exported as
two ONNX graphs and the agent can embed each frame once and run the cheap head
per step (section 3.1's "why two graphs" note).
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import timm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.labels import LABELS

BACKBONE_NAME = "mobilenetv4_conv_small"
EMBEDDING_DIM = 1280
# Derived, never hardcoded: this was a literal 5 while the ethogram declared 5
# classes, which is fine right up until the ethogram changes and the head
# silently keeps the old width.
NUM_CLASSES = len(LABELS)


def build_backbone(pretrained: bool = True) -> nn.Module:
    """Frozen frame embedder. Always kept in eval() — see section 3.2 (start frozen)."""
    backbone = timm.create_model(
        BACKBONE_NAME, pretrained=pretrained, num_classes=0, global_pool="avg"
    )
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


class TemporalHead(nn.Module):
    """1-layer GRU over a window of frame embeddings -> per-class logits.

    Input [B, N, EMBEDDING_DIM], output [B, NUM_CLASSES]. Cheap: this is the
    part that runs every step in the agent's hot path, unlike the backbone.
    """

    def __init__(self, input_dim: int = EMBEDDING_DIM, proj_dim: int = 64, hidden_dim: int = 32,
                 num_classes: int = NUM_CLASSES, dropout: float = 0.4,
                 emb_mean: torch.Tensor | None = None, emb_std: torch.Tensor | None = None):
        super().__init__()
        # Baked-in standardization of the (frozen) backbone's embeddings, computed
        # once from the training set. Registered as buffers (not parameters) so
        # they export straight into the ONNX graph and the agent needs no separate
        # normalization step at inference time.
        self.register_buffer("emb_mean", emb_mean if emb_mean is not None else torch.zeros(input_dim))
        self.register_buffer("emb_std", emb_std if emb_std is not None else torch.ones(input_dim))
        # Project 1280-d embeddings down before the GRU: with only ~145 unique
        # training clips, a GRU sized directly to the 1280-d embedding has far
        # more input-to-hidden capacity than the data can support (it memorizes
        # the train set in a single epoch). The projection keeps the temporal
        # head small enough to generalize.
        self.proj = nn.Linear(input_dim, proj_dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(proj_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.emb_mean) / self.emb_std
        x = self.proj_dropout(torch.relu(self.proj(x)))
        _, h_n = self.gru(x)          # h_n: [num_layers, B, hidden_dim]
        # h_n[-1], not h_n.squeeze(0). squeeze(0) happens to be equivalent
        # today (num_layers == 1) and is NOT what caused the deployed
        # head.onnx to export with a static batch axis - that was the ONNX
        # exporter, see export_onnx.py's dynamo note. This is a plain
        # correctness fix: squeeze(0) drops whatever size-1 axis is in front,
        # so it silently starts taking the wrong slice the moment num_layers
        # goes above 1, while h_n[-1] always means "the last layer's state".
        h = h_n[-1]                   # [B, hidden_dim]
        h = self.dropout(h)
        return self.fc(h)             # [B, num_classes]
