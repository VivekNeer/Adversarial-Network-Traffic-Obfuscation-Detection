"""Inference helpers that go beyond the fixed 128-packet window.

The training pipeline encodes only the first :data:`SEQ_LEN` packets of a flow.
That is fine for the balanced experiments but leaves an obvious gap in
deployment: an attacker can front-load a benign-looking prologue and push the
malicious behaviour past the window. ``score_flow_windows`` closes it by sliding
the window across the *whole* flow and combining the per-window predictions.

Two combination rules are offered because they answer different questions:

``mean``
    average the class probabilities -- "what does this flow look like overall?"

``max_malicious``
    take the window with the highest malicious probability -- "is there any
    stretch of this flow that looks hostile?" This is the right rule for
    detection, where one malicious segment is enough.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from antod.data.datasets import StatsScaler
from antod.data.features import SEQ_LEN, sequence_tensor, stats_vector
from antod.data.synth import BENIGN, LABEL_NAMES
from antod.models import FlowClassifier


@dataclass
class WindowedScore:
    n_windows: int
    proba: np.ndarray  # (3,) combined probabilities
    per_window: np.ndarray  # (n_windows, 3)
    label: int

    @property
    def label_name(self) -> str:
        return LABEL_NAMES[self.label]


def windows_of(
    pkts: np.ndarray, window: int = SEQ_LEN, stride: int | None = None
) -> list[np.ndarray]:
    """Slice a packet array into overlapping windows, always including the tail.

    Each window is re-based so its first packet is at time zero, which is what
    the feature extractor expects and what a detector observing that segment in
    isolation would see.
    """
    stride = stride or max(window // 2, 1)
    n = pkts.shape[0]
    if n <= window:
        return [pkts]

    starts = list(range(0, n - window + 1, stride))
    if starts[-1] != n - window:
        starts.append(n - window)  # the last packets must not be dropped

    out = []
    for s in starts:
        w = pkts[s : s + window].copy()
        w[:, 0] -= w[0, 0]
        out.append(w)
    return out


@torch.no_grad()
def score_flow_windows(
    model: FlowClassifier,
    scaler: StatsScaler,
    pkts: np.ndarray,
    window: int = SEQ_LEN,
    stride: int | None = None,
    combine: str = "max_malicious",
    device: torch.device | str = "cpu",
) -> WindowedScore:
    """Classify a flow of any length by sliding a window across it."""
    if combine not in {"mean", "max_malicious"}:
        raise ValueError(f"unknown combine rule {combine!r}")

    pieces = windows_of(pkts, window, stride)
    seq = torch.from_numpy(np.stack([sequence_tensor(w, window) for w in pieces])).to(device)
    stats = torch.from_numpy(scaler.transform(np.stack([stats_vector(w) for w in pieces]))).to(
        device
    )

    model.eval()
    per_window = torch.softmax(model(seq, stats), dim=1).cpu().numpy()

    if combine == "mean":
        proba = per_window.mean(axis=0)
    else:
        malicious = 1.0 - per_window[:, BENIGN]
        proba = per_window[int(malicious.argmax())]

    return WindowedScore(
        n_windows=len(pieces),
        proba=proba,
        per_window=per_window,
        label=int(proba.argmax()),
    )
