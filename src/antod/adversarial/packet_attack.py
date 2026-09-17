"""Packet-space black-box attack: modify the flow itself, re-extract, re-score.

The gradient attacks in :mod:`antod.adversarial.attacks` perturb *features* under
a box that approximates what an attacker can do. This module removes the
approximation. It edits the packet array directly using only the two operations
an attacker genuinely has -- **pad** a packet (grow it) and **delay** a packet
(hold it back) -- then re-runs the real feature extractor and asks the model
again. Every adversarial flow it produces is, by construction, a flow that could
be sent.

It is also **black-box**: it never reads a gradient, only the model's output
probability. That is the realistic setting for an attacker who can probe a
deployed detector but does not have its weights, and it makes the result
directly comparable across architectures (the gradient attacks are not, because
the MLP has no sequence gradient and the CNN has no statistics gradient).

The search is a simple greedy hill-climb: propose a random batch of pads and
delays, keep it if the benign probability goes up, discard it otherwise, until
the query budget is spent or the flow crosses the decision boundary. Simple on
purpose -- the point is to measure whether the model can be walked across the
boundary with physically legal edits, not to find the tightest possible path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from antod.data.datasets import StatsScaler
from antod.data.features import sequence_tensor, stats_vector
from antod.data.profiles import MTU
from antod.data.synth import BENIGN, Flow
from antod.models import FlowClassifier


@dataclass
class PacketAttackConfig:
    max_queries: int = 150
    #: most bytes added to any single packet in one proposal
    pad_bytes: int = 300
    #: most delay added to any single gap in one proposal, seconds
    delay_s: float = 0.05
    #: share of packets touched by one proposal
    frac_packets: float = 0.15
    #: stop as soon as the flow is classified as this
    target: int = BENIGN
    seed: int = 0


@dataclass
class PacketAttackResult:
    original: np.ndarray
    adversarial: np.ndarray
    queries: int
    success: bool
    p_target_before: float
    p_target_after: float
    bytes_added: float
    delay_added: float

    @property
    def overhead(self) -> float:
        """Extra bytes as a fraction of the original volume: the attacker's cost."""
        return float(self.bytes_added / max(self.original[:, 1].sum(), 1.0))


@torch.no_grad()
def _score(
    model: FlowClassifier, scaler: StatsScaler, pkts: np.ndarray, device: torch.device
) -> np.ndarray:
    seq = torch.from_numpy(sequence_tensor(pkts)[None]).to(device)
    stats = torch.from_numpy(scaler.transform(stats_vector(pkts)[None])).to(device)
    return torch.softmax(model(seq, stats), dim=1)[0].cpu().numpy()


def _propose(pkts: np.ndarray, cfg: PacketAttackConfig, rng: np.random.Generator) -> np.ndarray:
    """One random batch of pads and delays. Both are monotone: legal by construction."""
    out = pkts.copy()
    n = out.shape[0]
    k = max(1, int(round(n * cfg.frac_packets)))

    # pad: grow k random packets, never past the MTU
    idx = rng.choice(n, size=k, replace=False)
    out[idx, 1] = np.minimum(out[idx, 1] + rng.integers(1, cfg.pad_bytes + 1, size=k), MTU)

    # delay: hold back k random packets; a delay shifts everything after it too,
    # so it is applied to the gap and re-accumulated -- causality is preserved
    if n > 1:
        gaps = np.diff(out[:, 0], prepend=0.0)
        idx = rng.choice(np.arange(1, n), size=min(k, n - 1), replace=False)
        gaps[idx] += rng.uniform(0.0, cfg.delay_s, size=idx.size)
        out[:, 0] = np.cumsum(gaps)
        out[:, 0] -= out[0, 0]
    return out


def packet_space_attack(
    model: FlowClassifier,
    scaler: StatsScaler,
    pkts: np.ndarray,
    cfg: PacketAttackConfig,
    device: torch.device | str = "cpu",
) -> PacketAttackResult:
    """Greedy hill-climb over pad/delay edits until the flow reads as ``cfg.target``."""
    device = torch.device(device)
    rng = np.random.default_rng(cfg.seed)
    model.eval()

    current = pkts.copy()
    p_before = _score(model, scaler, current, device)
    best = float(p_before[cfg.target])
    queries = 1
    success = int(p_before.argmax()) == cfg.target

    while queries < cfg.max_queries and not success:
        candidate = _propose(current, cfg, rng)
        proba = _score(model, scaler, candidate, device)
        queries += 1
        if proba[cfg.target] > best:
            current, best = candidate, float(proba[cfg.target])
            success = int(proba.argmax()) == cfg.target

    return PacketAttackResult(
        original=pkts,
        adversarial=current,
        queries=queries,
        success=success,
        p_target_before=float(p_before[cfg.target]),
        p_target_after=best,
        bytes_added=float(current[:, 1].sum() - pkts[:, 1].sum()),
        delay_added=float(current[-1, 0] - pkts[-1, 0]),
    )


def evaluate_packet_attack(
    model: FlowClassifier,
    scaler: StatsScaler,
    flows: list[Flow],
    cfg: PacketAttackConfig,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Run the attack on every malicious flow given and summarise.

    ``evasion_before`` is the share already read as benign with no edits;
    ``evasion_after`` is the share the attacker can walk across the boundary
    within the query budget. ``mean_overhead`` is the price: extra bytes as a
    fraction of the original flow, averaged over successful evasions.
    """
    malicious = [f for f in flows if f.label != BENIGN]
    if not malicious:
        raise ValueError("no malicious flows to attack")

    results = [packet_space_attack(model, scaler, f.packets, cfg, device) for f in malicious]
    successes = [r for r in results if r.success]
    walked = [r for r in successes if r.queries > 1]  # not already misclassified

    return {
        "n_flows": len(results),
        "evasion_before": float(np.mean([r.queries == 1 and r.success for r in results])),
        "evasion_after": float(np.mean([r.success for r in results])),
        "mean_queries_to_evade": float(np.mean([r.queries for r in walked])) if walked else 0.0,
        "mean_overhead": float(np.mean([r.overhead for r in walked])) if walked else 0.0,
        "mean_delay_added_s": float(np.mean([r.delay_added for r in walked])) if walked else 0.0,
        "max_queries": cfg.max_queries,
    }
