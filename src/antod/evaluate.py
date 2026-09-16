"""Evaluation routines: clean scores, attack sweeps, per-technique probes, transfer.

The routines here exist because a single accuracy number cannot answer the
questions the project actually asks:

``per_technique_recall``
    *Which* evasion technique defeats the detector? An obfuscated-recall of 0.90
    means something different if the missing tenth is spread across all eight
    transforms than if it is entirely protocol mimicry.

``robustness_curve``
    How does the model degrade as the attacker's budget grows? A single epsilon
    reports one point on a curve whose shape is the actual result -- graceful
    decay and a cliff at the same midpoint are different security properties.

``transfer_matrix``
    Does an attacker need the model? White-box numbers assume the adversary has
    the weights. Crafting on one architecture and testing on another measures the
    realistic case, and a defense that only survives white-box attacks while
    collapsing under transfer has not been tested at all.

``constrained_vs_unconstrained``
    How much does ignoring domain validity overstate the threat? This is the
    number that separates a real vulnerability from an artefact of attacking a
    feature vector as if it were an image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from antod.adversarial.attacks import AttackConfig, perturbation_norms, run_attack
from antod.adversarial.defenses import smoothed_predict
from antod.data.datasets import (
    FlowDataset,
    FlowTensorDataset,
    Splits,
    StatsScaler,
)
from antod.data.obfuscation import TRANSFORMS
from antod.data.synth import (
    MALICIOUS_OBFUSCATED,
    SynthConfig,
    generate_technique_probe,
)
from antod.models import FlowClassifier, SklearnBaseline, build_baseline
from antod.utils.common import get_logger
from antod.utils.metrics import Metrics, compute_metrics, recall_by_group

logger = get_logger(__name__)


@dataclass
class AttackOutcome:
    """One attack against one model."""

    label: str
    metrics: Metrics
    norms: dict[str, float]
    evasion_rate: float

    def row(self) -> dict[str, Any]:
        return {
            "attack": self.label,
            "accuracy": self.metrics.accuracy,
            "macro_f1": self.metrics.macro_f1,
            "obfuscated_recall": self.metrics.obfuscated_recall,
            "malicious_recall": self.metrics.malicious_recall,
            "false_positive_rate": self.metrics.false_positive_rate,
            "evasion_rate": self.evasion_rate,
            "stats_linf": self.norms["stats_linf"],
            "seq_linf": self.norms["seq_linf"],
        }


def _tensors(ds: FlowDataset, scaler: StatsScaler, device: torch.device) -> tuple[Tensor, ...]:
    seq, stats, y = FlowTensorDataset(ds, scaler).tensors()
    return seq.to(device), stats.to(device), y.to(device)


@torch.no_grad()
def _predict(model: FlowClassifier, seq: Tensor, stats: Tensor) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    proba = torch.softmax(model(seq, stats), dim=1)
    return proba.argmax(dim=1).cpu().numpy(), proba.cpu().numpy()


def _evasion_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of truly-malicious flows that the model called benign.

    This, not accuracy, is what an attacker optimises. A malicious flow
    misclassified as the *other* malicious class is still caught.
    """
    malicious = y_true != 0
    if not malicious.any():
        return 0.0
    return float((y_pred[malicious] == 0).mean())


# --------------------------------------------------------------------------- #
# clean and attacked evaluation
# --------------------------------------------------------------------------- #
def evaluate_clean(
    model: FlowClassifier, ds: FlowDataset, scaler: StatsScaler, device: torch.device
) -> Metrics:
    seq, stats, y = _tensors(ds, scaler, device)
    y_pred, y_proba = _predict(model, seq, stats)
    return compute_metrics(y.cpu().numpy(), y_pred, y_proba)


def evaluate_attack(
    model: FlowClassifier,
    ds: FlowDataset,
    scaler: StatsScaler,
    cfg: AttackConfig,
    device: torch.device,
    source_model: FlowClassifier | None = None,
) -> AttackOutcome:
    """Attack a model and score the result.

    ``source_model`` makes this a transfer attack: examples are crafted against
    the source and evaluated on ``model``.
    """
    seq, stats, y = _tensors(ds, scaler, device)
    crafter = source_model or model
    adv_seq, adv_stats = run_attack(crafter, seq, stats, y, cfg, scaler)

    y_pred, y_proba = _predict(model, adv_seq, adv_stats)
    y_true = y.cpu().numpy()
    return AttackOutcome(
        label=cfg.label(),
        metrics=compute_metrics(y_true, y_pred, y_proba),
        norms=perturbation_norms(seq, adv_seq, stats, adv_stats),
        evasion_rate=_evasion_rate(y_true, y_pred),
    )


def attack_sweep(
    model: FlowClassifier,
    ds: FlowDataset,
    scaler: StatsScaler,
    configs: list[AttackConfig],
    device: torch.device,
) -> list[AttackOutcome]:
    out = []
    for cfg in configs:
        outcome = evaluate_attack(model, ds, scaler, cfg, device)
        logger.info("%-58s %s", outcome.label, outcome.metrics.summary())
        out.append(outcome)
    return out


def robustness_curve(
    model: FlowClassifier,
    ds: FlowDataset,
    scaler: StatsScaler,
    base: AttackConfig,
    eps_values: list[float],
    device: torch.device,
    metric: str = "accuracy",
) -> tuple[list[float], list[float]]:
    """Trace one metric against attack budget. Returns ``(eps, values)``."""
    from dataclasses import replace

    xs, ys = [], []
    for eps in eps_values:
        cfg = replace(base, eps=eps)
        outcome = evaluate_attack(model, ds, scaler, cfg, device)
        xs.append(eps)
        ys.append(
            outcome.evasion_rate
            if metric == "evasion_rate"
            else float(getattr(outcome.metrics, metric))
        )
    return xs, ys


def constrained_vs_unconstrained(
    model: FlowClassifier,
    ds: FlowDataset,
    scaler: StatsScaler,
    base: AttackConfig,
    eps_values: list[float],
    device: torch.device,
) -> dict[str, tuple[list[float], list[float]]]:
    """Robustness curves with and without the pad-only / delay-only constraints."""
    from dataclasses import replace

    return {
        "domain-constrained": robustness_curve(
            model, ds, scaler, replace(base, constrained=True), eps_values, device
        ),
        "unconstrained": robustness_curve(
            model, ds, scaler, replace(base, constrained=False), eps_values, device
        ),
    }


# --------------------------------------------------------------------------- #
# per-technique detectability
# --------------------------------------------------------------------------- #
def per_technique_recall(
    model: FlowClassifier,
    scaler: StatsScaler,
    device: torch.device,
    n_per_technique: int = 400,
    seed: int = 7,
    synth: SynthConfig | None = None,
) -> dict[str, float]:
    """Recall on malicious flows obfuscated by exactly one named technique.

    Each probe set is generated fresh, holding the transform fixed, so the result
    attributes a miss to a technique rather than to a mixed recipe.
    """
    out: dict[str, float] = {}
    for name in TRANSFORMS:
        flows = generate_technique_probe(name, n_per_technique, seed=seed, cfg=synth)
        ds = FlowDataset.from_flows(flows)
        seq, stats, y = _tensors(ds, scaler, device)
        y_pred, _ = _predict(model, seq, stats)
        out[name] = float((y_pred == MALICIOUS_OBFUSCATED).mean())
        logger.info("technique %-24s recall %.4f", name, out[name])
    return out


def per_recipe_accuracy(
    model: FlowClassifier, ds: FlowDataset, scaler: StatsScaler, device: torch.device
) -> dict[str, float]:
    """Accuracy grouped by the recipe recorded on each flow."""
    seq, stats, y = _tensors(ds, scaler, device)
    y_pred, _ = _predict(model, seq, stats)
    return recall_by_group(y.cpu().numpy(), y_pred, ds.recipe)


def per_profile_accuracy(
    model: FlowClassifier, ds: FlowDataset, scaler: StatsScaler, device: torch.device
) -> dict[str, float]:
    """Accuracy grouped by application profile: which traffic type is hardest?"""
    seq, stats, y = _tensors(ds, scaler, device)
    y_pred, _ = _predict(model, seq, stats)
    return recall_by_group(y.cpu().numpy(), y_pred, ds.profile)


# --------------------------------------------------------------------------- #
# transfer
# --------------------------------------------------------------------------- #
def transfer_matrix(
    models: dict[str, FlowClassifier],
    ds: FlowDataset,
    scaler: StatsScaler,
    cfg: AttackConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Craft on each model, evaluate on each model. Diagonal is the white-box case.

    Off-diagonal entries are the realistic threat: an attacker who has studied
    *some* detector, not necessarily the deployed one.
    """
    rows: list[dict[str, Any]] = []
    for source_name, source in models.items():
        row: dict[str, Any] = {"crafted_on": source_name}
        for target_name, target in models.items():
            outcome = evaluate_attack(target, ds, scaler, cfg, device, source_model=source)
            row[target_name] = outcome.metrics.accuracy
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# smoothing
# --------------------------------------------------------------------------- #
def smoothing_sweep(
    model: FlowClassifier,
    ds: FlowDataset,
    scaler: StatsScaler,
    sigmas: list[float],
    device: torch.device,
    attack: AttackConfig | None = None,
    n_samples: int = 16,
) -> list[dict[str, Any]]:
    """Randomised smoothing at several noise levels, clean and under attack.

    Reported together on purpose: noise that blunts a perturbation also blurs the
    genuine size and timing fingerprints, and the clean column is where that cost
    shows up.
    """
    seq, stats, y = _tensors(ds, scaler, device)
    y_true = y.cpu().numpy()

    adv_seq, adv_stats = (seq, stats)
    if attack is not None:
        adv_seq, adv_stats = run_attack(model, seq, stats, y, attack, scaler)

    rows: list[dict[str, Any]] = []
    for sigma in sigmas:
        clean_pred = smoothed_predict(model, seq, stats, sigma, n_samples).cpu().numpy()
        row: dict[str, Any] = {
            "sigma": sigma,
            "clean_accuracy": float((clean_pred == y_true).mean()),
        }
        if attack is not None:
            adv_pred = smoothed_predict(model, adv_seq, adv_stats, sigma, n_samples).cpu().numpy()
            row["attacked_accuracy"] = float((adv_pred == y_true).mean())
            row["evasion_rate"] = _evasion_rate(y_true, adv_pred)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def fit_baselines(splits: Splits, names: list[str], seed: int = 42) -> dict[str, SklearnBaseline]:
    """Fit the classical models on the same scaled statistics the networks see."""
    x_train = splits.scaler.transform(splits.train.stats)
    out: dict[str, SklearnBaseline] = {}
    for name in names:
        logger.info("fitting baseline %s", name)
        out[name] = build_baseline(name, seed=seed).fit(x_train, splits.train.y)
    return out


def evaluate_baselines(baselines: dict[str, SklearnBaseline], splits: Splits) -> dict[str, Metrics]:
    x_test = splits.scaler.transform(splits.test.stats)
    out: dict[str, Metrics] = {}
    for name, model in baselines.items():
        pred = model.predict(x_test)
        proba = model.predict_proba(x_test)
        out[name] = compute_metrics(splits.test.y, pred, proba)
        logger.info("%-22s %s", name, out[name].summary())
    return out
