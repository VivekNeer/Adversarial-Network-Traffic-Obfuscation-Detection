"""MLP over the 51 flow statistics, and a hybrid that reads both views.

The MLP is the comparison the project needs. If a plain feed-forward network on
hand-engineered aggregates matches the 1D-CNN, then the convolution is not earning
its complexity and the honest conclusion is to say so. The interesting case is
where they differ: aggregates are permutation-invariant, so an MLP can see *that*
a flow is padded but not *where* -- it cannot distinguish a uniformly shaped flow
from ordinary traffic carrying one shaped burst, because both produce the same
means and variances.

The hybrid tests whether those two accounts are complementary. Its two trunks are
kept separate until the final head so each keeps its own normalisation: joining
earlier would force a single BatchNorm over a pooled convolutional embedding and a
standardised statistics vector, and the statistics branch tends to be swamped.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from antod.models.base import (
    N_CLASSES,
    N_FEATURES,
    SEQ_CHANNELS,
    FlowClassifier,
    _init_weights,
    register,
)
from antod.models.cnn1d import SequenceTrunk


def _mlp_trunk(in_dim: int, hidden: tuple[int, ...], dropout: float) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [
            nn.Linear(prev, h),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ]
        prev = h
    return nn.Sequential(*layers), prev


class MLP(FlowClassifier):
    """Feed-forward detector over the flow-statistics view."""

    uses_seq = False
    uses_stats = True

    def __init__(
        self,
        in_features: int = N_FEATURES,
        hidden: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__(n_classes=n_classes)
        self.trunk, self.embed_dim = _mlp_trunk(in_features, hidden, dropout)
        self.head = nn.Linear(self.embed_dim, n_classes)
        self.apply(_init_weights)

    def embed(self, stats: Tensor) -> Tensor:
        return self.trunk(stats)

    def forward(self, seq: Tensor, stats: Tensor) -> Tensor:  # noqa: ARG002 - seq unused
        return self.head(self.trunk(stats))


class HybridCNNMLP(FlowClassifier):
    """Convolutional sequence trunk and statistics trunk joined at the head."""

    uses_seq = True
    uses_stats = True

    def __init__(
        self,
        in_channels: int = SEQ_CHANNELS,
        in_features: int = N_FEATURES,
        cnn_channels: tuple[int, ...] = (64, 128, 128),
        stats_hidden: tuple[int, ...] = (128, 64),
        hidden: int = 128,
        dropout: float = 0.1,
        stats_dropout: float = 0.3,
        head_dropout: float = 0.3,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__(n_classes=n_classes)

        self.seq_trunk = SequenceTrunk(
            in_channels=in_channels,
            channels=cnn_channels,
            dropout=dropout,
        )
        self.stats_trunk, stats_dim = _mlp_trunk(in_features, stats_hidden, stats_dropout)

        joint = self.seq_trunk.embed_dim + stats_dim
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(joint, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(hidden, n_classes),
        )
        self.apply(_init_weights)

    def forward(self, seq: Tensor, stats: Tensor) -> Tensor:
        return self.head(torch.cat([self.seq_trunk(seq), self.stats_trunk(stats)], dim=1))


@register("mlp")
def _mlp(**kwargs) -> MLP:
    return MLP(**kwargs)


@register("mlp_wide")
def _mlp_wide(**kwargs) -> MLP:
    """Wider variant, for the capacity ablation."""
    kwargs.setdefault("hidden", (512, 256, 128))
    return MLP(**kwargs)


@register("hybrid")
def _hybrid(**kwargs) -> HybridCNNMLP:
    return HybridCNNMLP(**kwargs)
