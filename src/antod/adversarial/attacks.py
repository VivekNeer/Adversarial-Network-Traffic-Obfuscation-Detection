"""FGSM and PGD against both input surfaces, under domain-validity constraints.

The whole point of this module is the constraints, so they come first.

Why an unconstrained attack is the wrong measurement
----------------------------------------------------
Standard FGSM on images perturbs every pixel in either direction inside an
:math:`\\ell_\\infty` ball, and the only validity requirement is that pixels stay
in ``[0, 1]``. Traffic is not like that. An attacker who controls a malicious flow
can do two things to its observable shape:

* **pad** a packet -- make it larger, never smaller. The payload has to fit.
* **delay** a packet -- send it later, never earlier. Causality.

Direction cannot be flipped (that would mean a different party sent the packet),
and packets cannot be conjured into or out of existence at a fixed sequence
position. So the feasible set is a **one-sided box**, not a symmetric ball:

=========== ==============================================================
channel 0   signed size: magnitude may only grow, up to the MTU
channel 1   log inter-arrival time: may only increase
channel 2   direction: frozen
channel 3   validity mask: frozen
=========== ==============================================================

An unconstrained attack is still implemented (``constrained=False``) because the
comparison is informative: it quantifies how much a naive robustness evaluation
overstates the threat by permitting flows that could not exist. Reporting only
the unconstrained number would make the detector look far more fragile than it is.

The statistics surface gets the same treatment through
:class:`~antod.data.datasets.FeatureConstraints`: perturbations are projected into
the box where proportions stay in ``[0, 1]``, counts stay non-negative and
entropies stay under their 32-bin ceiling.

Threat model
------------
``targeted=True`` aims the attack at the benign class, which is what an evader
actually wants -- being mistaken for ordinary traffic, not merely misclassified as
some other kind of attack. The untargeted variant is the standard robustness
curve. Experiments report both, since they answer different questions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from antod.data.datasets import FeatureConstraints, StatsScaler
from antod.data.synth import BENIGN
from antod.models import FlowClassifier

#: Largest inter-arrival time an attacker would plausibly introduce, in the
#: scaled units of sequence channel 1: ``log10(300 s) / 6``.
MAX_DELAY_SCALED = 0.41

_CH_SIZE, _CH_IAT, _CH_DIR, _CH_MASK = 0, 1, 2, 3


@dataclass
class AttackConfig:
    """One attack specification."""

    name: str = "pgd"  # fgsm | pgd | none
    surface: str = "stats"  # stats | seq | both
    eps: float = 0.1
    alpha: float | None = None  # step size; defaults to 2.5 * eps / steps
    steps: int = 10
    random_start: bool = True
    constrained: bool = True
    targeted: bool = False
    target_class: int = BENIGN
    seed: int = 0

    def __post_init__(self) -> None:
        if self.name not in {"fgsm", "pgd", "none"}:
            raise ValueError(f"unknown attack {self.name!r}")
        if self.surface not in {"stats", "seq", "both"}:
            raise ValueError(f"unknown surface {self.surface!r}")
        if self.eps < 0:
            raise ValueError("eps must be non-negative")
        if self.name == "pgd" and self.steps < 1:
            raise ValueError("pgd needs at least one step")

    @property
    def step_size(self) -> float:
        if self.alpha is not None:
            return self.alpha
        if self.name == "fgsm":
            return self.eps
        # the usual PGD heuristic: enough total travel to cross the ball 2.5 times
        return 2.5 * self.eps / max(self.steps, 1)

    def label(self) -> str:
        if self.name == "none":
            return "clean"
        bits = [self.name, self.surface, f"eps={self.eps:g}"]
        if self.name == "pgd":
            bits.append(f"steps={self.steps}")
        bits.append("constrained" if self.constrained else "unconstrained")
        if self.targeted:
            bits.append("targeted")
        return " ".join(bits)


# --------------------------------------------------------------------------- #
# feasible sets
# --------------------------------------------------------------------------- #
def sequence_bounds(seq: Tensor, constrained: bool = True) -> tuple[Tensor, Tensor]:
    """Per-element lower and upper bounds for a perturbed packet sequence.

    With ``constrained=True`` this encodes pad-only and delay-only: the magnitude
    of the signed size may grow towards the MTU but not shrink, and the log-IAT may
    increase but not decrease. Direction and the validity mask are pinned to their
    clean values in both modes -- flipping a direction bit or inventing a packet at
    a fixed sequence position does not correspond to anything an attacker can do.
    """
    lo = torch.full_like(seq, -1.0)
    hi = torch.full_like(seq, 1.0)

    if constrained:
        size = seq[:, _CH_SIZE, :]
        # sign() is 0 at exactly zero; treat those as upstream so the branch is defined
        sign = torch.where(size < 0, -torch.ones_like(size), torch.ones_like(size))
        magnitude = size.abs()
        lo[:, _CH_SIZE, :] = torch.where(sign > 0, magnitude, -torch.ones_like(size))
        hi[:, _CH_SIZE, :] = torch.where(sign > 0, torch.ones_like(size), -magnitude)

        iat = seq[:, _CH_IAT, :]
        lo[:, _CH_IAT, :] = iat
        # delay-only, capped at MAX_DELAY_SCALED -- but never below the clean value,
        # or a flow that already has a long gap would get an empty feasible range
        hi[:, _CH_IAT, :] = torch.maximum(torch.full_like(iat, MAX_DELAY_SCALED), iat)

    # direction and mask never move
    for ch in (_CH_DIR, _CH_MASK):
        lo[:, ch, :] = seq[:, ch, :]
        hi[:, ch, :] = seq[:, ch, :]

    # padding positions carry no packet: freeze every channel there
    valid = seq[:, _CH_MASK : _CH_MASK + 1, :] > 0
    lo = torch.where(valid, lo, seq)
    hi = torch.where(valid, hi, seq)
    return lo, hi


def stats_bounds(
    scaler: StatsScaler, constrained: bool = True, device: torch.device | str = "cpu"
) -> tuple[Tensor, Tensor]:
    """Lower and upper bounds for perturbed flow statistics, in scaled units."""
    if not constrained:
        c = float(scaler.clip)
        n = int(np.asarray(scaler.state_dict()["mean"]).size)
        return (
            torch.full((n,), -c, device=device),
            torch.full((n,), c, device=device),
        )
    return FeatureConstraints().torch_bounds(scaler, device=str(device))


# --------------------------------------------------------------------------- #
# the attacks
# --------------------------------------------------------------------------- #
def _loss_and_labels(cfg: AttackConfig, y: Tensor) -> tuple[Tensor, float]:
    """Target labels and the sign of the gradient step.

    Untargeted attacks ascend the loss of the true label. Targeted attacks
    *descend* the loss of the chosen target, so the step sign flips.
    """
    if cfg.targeted:
        return torch.full_like(y, cfg.target_class), -1.0
    return y, +1.0


def _project(x: Tensor, clean: Tensor, eps: float, lo: Tensor, hi: Tensor) -> Tensor:
    """Project onto the intersection of the eps-ball and the domain box."""
    x = torch.clamp(x, clean - eps, clean + eps)
    return torch.clamp(x, lo, hi)


def run_attack(
    model: FlowClassifier,
    seq: Tensor,
    stats: Tensor,
    y: Tensor,
    cfg: AttackConfig,
    scaler: StatsScaler | None = None,
) -> tuple[Tensor, Tensor]:
    """Craft adversarial versions of ``(seq, stats)``. Returns detached tensors.

    Surfaces the model does not read are returned untouched: perturbing the
    statistics of a pure sequence model would produce a "robust" result that
    reflects only the wiring.
    """
    if cfg.name == "none" or cfg.eps == 0:
        return seq.detach(), stats.detach()

    device = seq.device
    attack_seq = cfg.surface in {"seq", "both"} and model.uses_seq
    attack_stats = cfg.surface in {"stats", "both"} and model.uses_stats
    if not attack_seq and not attack_stats:
        return seq.detach(), stats.detach()

    if attack_stats and scaler is None:
        raise ValueError("attacking the statistics surface requires the fitted scaler")

    criterion = nn.CrossEntropyLoss()
    labels, direction = _loss_and_labels(cfg, y)

    seq_lo, seq_hi = sequence_bounds(seq, cfg.constrained)
    if attack_stats:
        s_lo, s_hi = stats_bounds(scaler, cfg.constrained, device)
        stats_lo = s_lo.unsqueeze(0).expand_as(stats)
        stats_hi = s_hi.unsqueeze(0).expand_as(stats)
    else:
        stats_lo = stats_hi = stats

    adv_seq = seq.clone().detach()
    adv_stats = stats.clone().detach()

    if cfg.name == "pgd" and cfg.random_start:
        generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
        if attack_seq:
            noise = (torch.rand(seq.shape, generator=generator).to(device) * 2 - 1) * cfg.eps
            adv_seq = _project(adv_seq + noise, seq, cfg.eps, seq_lo, seq_hi)
        if attack_stats:
            noise = (torch.rand(stats.shape, generator=generator).to(device) * 2 - 1) * cfg.eps
            adv_stats = _project(adv_stats + noise, stats, cfg.eps, stats_lo, stats_hi)

    n_steps = 1 if cfg.name == "fgsm" else cfg.steps
    step = cfg.step_size

    was_training = model.training
    model.eval()
    for _ in range(n_steps):
        adv_seq = adv_seq.detach().requires_grad_(attack_seq)
        adv_stats = adv_stats.detach().requires_grad_(attack_stats)

        loss = criterion(model(adv_seq, adv_stats), labels)
        grads = torch.autograd.grad(
            loss,
            [t for t, on in ((adv_seq, attack_seq), (adv_stats, attack_stats)) if on],
            allow_unused=True,
        )
        grad_iter = iter(grads)

        with torch.no_grad():
            if attack_seq:
                g = next(grad_iter)
                if g is not None:
                    adv_seq = _project(
                        adv_seq + direction * step * g.sign(), seq, cfg.eps, seq_lo, seq_hi
                    )
            if attack_stats:
                g = next(grad_iter)
                if g is not None:
                    adv_stats = _project(
                        adv_stats + direction * step * g.sign(),
                        stats,
                        cfg.eps,
                        stats_lo,
                        stats_hi,
                    )
    if was_training:
        model.train()

    return adv_seq.detach(), adv_stats.detach()


def fgsm(
    model: FlowClassifier,
    seq: Tensor,
    stats: Tensor,
    y: Tensor,
    eps: float = 0.1,
    surface: str = "stats",
    scaler: StatsScaler | None = None,
    **kwargs,
) -> tuple[Tensor, Tensor]:
    """Single-step gradient-sign attack."""
    cfg = AttackConfig(name="fgsm", surface=surface, eps=eps, **kwargs)
    return run_attack(model, seq, stats, y, cfg, scaler)


def pgd(
    model: FlowClassifier,
    seq: Tensor,
    stats: Tensor,
    y: Tensor,
    eps: float = 0.1,
    steps: int = 10,
    surface: str = "stats",
    scaler: StatsScaler | None = None,
    **kwargs,
) -> tuple[Tensor, Tensor]:
    """Iterative projected gradient attack."""
    cfg = AttackConfig(name="pgd", surface=surface, eps=eps, steps=steps, **kwargs)
    return run_attack(model, seq, stats, y, cfg, scaler)


def make_adversary(cfg: AttackConfig, scaler: StatsScaler):
    """Bind an attack config into the ``Adversary`` callable the trainer expects."""

    def adversary(
        model: FlowClassifier, seq: Tensor, stats: Tensor, y: Tensor
    ) -> tuple[Tensor, Tensor]:
        return run_attack(model, seq, stats, y, cfg, scaler)

    return adversary


def perturbation_norms(
    clean_seq: Tensor, adv_seq: Tensor, clean_stats: Tensor, adv_stats: Tensor
) -> dict[str, float]:
    """How large the perturbation actually was, for reporting alongside the drop.

    A robustness number is meaningless without it: an attack that needs to double
    every packet size is not the same threat as one that needs a 2% nudge, even if
    both defeat the model.
    """
    d_seq = (adv_seq - clean_seq).abs()
    d_stats = (adv_stats - clean_stats).abs()
    return {
        "seq_linf": float(d_seq.max()) if d_seq.numel() else 0.0,
        "seq_l2_mean": float(d_seq.flatten(1).norm(dim=1).mean()) if d_seq.numel() else 0.0,
        "stats_linf": float(d_stats.max()) if d_stats.numel() else 0.0,
        "stats_l2_mean": float(d_stats.norm(dim=1).mean()) if d_stats.numel() else 0.0,
    }
