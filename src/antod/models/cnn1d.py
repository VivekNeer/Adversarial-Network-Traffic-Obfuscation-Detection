"""1D-CNN over the packet sequence.

The architecture follows the shape of the problem rather than a paper. A flow is
a short, ordered, multi-channel signal, and the evidence for obfuscation lives at
three different scales:

* **local** -- a single padded packet, a fragment pair, one tunnel record. A
  wide first kernel (7 taps, ~7 packets) sees these directly.
* **intermediate** -- a request/response exchange, one beacon check-in, a burst
  boundary. Reached after the first pooling stage, where each unit covers roughly
  20 packets.
* **global** -- "this entire flow has a constant cadence", which is what
  constant-rate shaping produces. Only the pooled representation can express it.

Hence three convolution stages of increasing abstraction and decreasing length,
then **both** average and max pooling over time. That pair is not padding out the
architecture: average pooling answers "how much of this flow looks padded?" while
max pooling answers "is there anywhere in this flow that looks padded?" -- and the
two questions separate a uniformly shaped flow from one with a suspicious
fragment in the middle of otherwise ordinary traffic.

The model is intentionally small (~150k parameters). With 12k training flows a
heavier network would spend its capacity memorising generator noise, and the
robustness results in §5 would then be measuring overfitting rather than the
effect of the attack.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from antod.models.base import (
    N_CLASSES,
    SEQ_CHANNELS,
    FlowClassifier,
    _init_weights,
    register,
)


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> optional MaxPool -> Dropout.

    BatchNorm before the activation is what makes this trainable at a useful
    learning rate: raw channels arrive on wildly different scales (a normalised
    size in [-1, 1] next to a log-IAT that can sit at -1) and without
    normalisation the first layer's gradients are dominated by one channel.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        pool: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.pool(self.act(self.norm(self.conv(x)))))


class SequenceTrunk(nn.Module):
    """Convolution stages plus dual pooling: a packet sequence to a flat embedding.

    Kept separate from :class:`CNN1D` so the hybrid model can reuse the trunk
    without also instantiating a classification head it never calls. When the two
    were one class, the hybrid carried ~33k dead parameters that were saved into
    every checkpoint and counted in every reported model size.
    """

    def __init__(
        self,
        in_channels: int = SEQ_CHANNELS,
        channels: tuple[int, ...] = (64, 128, 128),
        kernels: tuple[int, ...] = (7, 5, 3),
        pools: tuple[int, ...] = (2, 2, 1),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not len(channels) == len(kernels) == len(pools):
            raise ValueError("channels, kernels and pools must be the same length")

        blocks: list[nn.Module] = []
        prev = in_channels
        for ch, k, p in zip(channels, kernels, pools, strict=True):
            blocks.append(ConvBlock(prev, ch, kernel=k, pool=p, dropout=dropout))
            prev = ch

        self.features = nn.Sequential(*blocks)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.embed_dim = prev * 2

    def forward(self, seq: Tensor) -> Tensor:
        h = self.features(seq)
        return torch.cat([self.avg_pool(h).squeeze(-1), self.max_pool(h).squeeze(-1)], dim=1)


class CNN1D(FlowClassifier):
    """Three-stage 1D convolutional detector over the packet sequence."""

    uses_seq = True
    uses_stats = False

    def __init__(
        self,
        in_channels: int = SEQ_CHANNELS,
        channels: tuple[int, ...] = (64, 128, 128),
        kernels: tuple[int, ...] = (7, 5, 3),
        pools: tuple[int, ...] = (2, 2, 1),
        hidden: int = 128,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__(n_classes=n_classes)
        self.trunk = SequenceTrunk(
            in_channels=in_channels,
            channels=channels,
            kernels=kernels,
            pools=pools,
            dropout=dropout,
        )
        self.embed_dim = self.trunk.embed_dim

        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(self.embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(hidden, n_classes),
        )
        self.apply(_init_weights)

    def embed(self, seq: Tensor) -> Tensor:
        """Pooled representation of a flow, before the classification head."""
        return self.trunk(seq)

    def forward(self, seq: Tensor, stats: Tensor) -> Tensor:  # noqa: ARG002 - stats unused
        return self.head(self.trunk(seq))


@register("cnn1d")
def _cnn1d(**kwargs) -> CNN1D:
    return CNN1D(**kwargs)


@register("cnn1d_small")
def _cnn1d_small(**kwargs) -> CNN1D:
    """Half-width variant, used for the capacity ablation in the report."""
    kwargs.setdefault("channels", (32, 64, 64))
    kwargs.setdefault("hidden", 64)
    return CNN1D(**kwargs)
