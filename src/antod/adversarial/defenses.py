"""Defenses: adversarial training, and randomised smoothing at inference.

**Adversarial training** is the standard answer and the one the project's title
promises: generate perturbed flows during training and learn on them. The
implementation is deliberately thin -- it builds an attack, hands it to the same
:class:`~antod.train.Trainer` the baseline uses, and changes nothing else. If the
defended model were trained by a different loop, any difference between the two
could be the loop rather than the defense.

The attack used for training is intentionally *weaker* than the attack used for
evaluation (fewer PGD steps by default). Training against the full evaluation
attack would be circular: the model would be fitted to the exact perturbation it
is later tested on, and the reported robustness would not generalise to an
attacker who did anything slightly different.

**Randomised smoothing** is the cheap test-time alternative: classify several
noisy copies of a flow and take the majority vote. It needs no retraining, which
makes it deployable on a model already in production, and it exposes a real
trade-off -- noise that blunts an adversarial perturbation also blurs the genuine
size and timing fingerprints the detector depends on. Measuring where that
trade-off sits is more useful than assuming it helps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from antod.adversarial.attacks import AttackConfig, make_adversary
from antod.data.datasets import Splits
from antod.models import FlowClassifier
from antod.train import TrainConfig, Trainer, TrainResult


@dataclass
class DefenseConfig:
    """How to harden a model."""

    method: str = "adversarial_training"  # adversarial_training | none
    attack: AttackConfig = field(
        default_factory=lambda: AttackConfig(name="pgd", surface="both", eps=0.1, steps=5)
    )

    def __post_init__(self) -> None:
        if self.method not in {"adversarial_training", "none"}:
            raise ValueError(f"unknown defense {self.method!r}")


def adversarial_training(
    train_cfg: TrainConfig,
    splits: Splits,
    defense: DefenseConfig | None = None,
    model: FlowClassifier | None = None,
) -> TrainResult:
    """Train with a share of each batch replaced by adversarial examples.

    ``train_cfg.adv_ratio`` controls the share. It is below 1.0 by default so the
    model keeps seeing clean traffic: a detector that traded ordinary-traffic
    accuracy for robustness has not obviously been improved, and keeping both
    numbers meaningful requires both kinds of example in the batch.
    """
    defense = defense or DefenseConfig()
    if defense.method == "none":
        return Trainer(train_cfg).fit(splits, model=model)

    adversary = make_adversary(defense.attack, splits.scaler)
    return Trainer(train_cfg, adversary=adversary).fit(splits, model=model)


@torch.no_grad()
def smoothed_predict(
    model: FlowClassifier,
    seq: Tensor,
    stats: Tensor,
    sigma: float = 0.05,
    n_samples: int = 16,
    seed: int = 0,
) -> Tensor:
    """Majority vote over ``n_samples`` Gaussian-noised copies of each flow.

    Noise is added only to the surfaces the model reads, and the sequence's
    direction and validity channels are left alone -- perturbing the mask would
    ask the model about packets that do not exist, which is not a defense but a
    different question.
    """
    model.eval()
    device = seq.device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    votes = torch.zeros(seq.size(0), model.n_classes, device=device)

    for _ in range(max(n_samples, 1)):
        noisy_seq, noisy_stats = seq, stats

        if model.uses_seq and sigma > 0:
            noise = torch.randn(seq.shape, generator=generator).to(device) * sigma
            noise[:, 2:4, :] = 0.0  # direction and mask stay exact
            noise = noise * (seq[:, 3:4, :] > 0)  # and nothing leaks into padding
            noisy_seq = torch.clamp(seq + noise, -1.0, 1.0)

        if model.uses_stats and sigma > 0:
            noise = torch.randn(stats.shape, generator=generator).to(device) * sigma
            noisy_stats = stats + noise

        pred = model(noisy_seq, noisy_stats).argmax(dim=1)
        votes.scatter_add_(1, pred.unsqueeze(1), torch.ones_like(votes[:, :1]))

    return votes.argmax(dim=1)


@dataclass
class SmoothingConfig:
    sigma: float = 0.05
    n_samples: int = 16
    seed: int = 0
