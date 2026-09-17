"""Bidirectional GRU over the packet sequence: a second sequence baseline.

A recurrent model reads the flow in order and carries state, so it can in
principle notice a change of behaviour *late* in the window that a pooled
convolution would average away. If the GRU matches the CNN, the sequence signal
is local and the convolution's cheaper inductive bias is the right one; if the
GRU wins, order and long-range context matter and the report should say so.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from antod.models.base import N_CLASSES, SEQ_CHANNELS, FlowClassifier, _init_weights, register


class GRUClassifier(FlowClassifier):
    uses_seq = True
    uses_stats = False

    def __init__(
        self,
        in_channels: int = SEQ_CHANNELS,
        hidden: int = 64,
        layers: int = 1,
        dropout: float = 0.2,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__(n_classes=n_classes)
        self.gru = nn.GRU(
            in_channels,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.embed_dim = hidden * 2
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
        self.head.apply(_init_weights)

    def embed(self, seq: Tensor) -> Tensor:
        """Masked mean of the GRU states over real packets only.

        The sequence is packed to its true length before the GRU so that neither
        direction ever reads a padding slot -- a bidirectional GRU's backward
        pass would otherwise start *in* the padding, and the embedding would
        depend on how long the flow was padded to.
        """
        x = seq.transpose(1, 2)  # (B, L, C)
        mask = seq[:, 3, :]  # (B, L)
        lengths = mask.sum(dim=1).clamp(min=1).to(torch.int64).cpu()

        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))  # (B, L, 2H)

        m = mask.unsqueeze(-1)
        return (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def forward(self, seq: Tensor, stats: Tensor) -> Tensor:  # noqa: ARG002 - stats unused
        return self.head(self.embed(seq))


@register("gru")
def _gru(**kwargs) -> GRUClassifier:
    return GRUClassifier(**kwargs)


__all__ = ["GRUClassifier"]
