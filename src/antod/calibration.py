"""Temperature scaling and expected calibration error.

A detector's probabilities are only useful for setting an alert threshold if
they mean what they say -- if flows scored 0.9 are actually malicious about 90%
of the time. Networks trained with cross-entropy are usually *over*-confident,
and adversarially trained ones drift the other way.

Temperature scaling fits a single scalar ``T`` on the validation split so that
``softmax(logits / T)`` is calibrated. It cannot change the argmax, so accuracy
is untouched; it only makes the confidence honest. ECE (expected calibration
error) is the standard measure: the average gap between confidence and accuracy
across confidence bins, weighted by how many samples fall in each.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from antod.models import FlowClassifier


def expected_calibration_error(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """ECE of the top-class confidence, in ``[0, 1]``. Lower is better."""
    confidence = proba.max(axis=1)
    predicted = proba.argmax(axis=1)
    correct = (predicted == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


@dataclass
class TemperatureScaler:
    """One learned scalar dividing the logits."""

    temperature: float = 1.0

    def fit(self, logits: Tensor, y: Tensor, max_iter: int = 200) -> TemperatureScaler:
        """Minimise validation NLL over ``T`` with L-BFGS (a 1-D convex problem)."""
        log_t = torch.zeros(1, requires_grad=True)
        nll = nn.CrossEntropyLoss()
        optimiser = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

        logits = logits.detach()
        y = y.detach()

        def closure() -> Tensor:
            optimiser.zero_grad()
            loss = nll(logits / log_t.exp(), y)
            loss.backward()
            return loss

        optimiser.step(closure)
        self.temperature = float(log_t.exp().item())
        return self

    def apply(self, logits: Tensor) -> Tensor:
        return torch.softmax(logits / self.temperature, dim=1)


@torch.no_grad()
def collect_logits(
    model: FlowClassifier, seq: Tensor, stats: Tensor, batch_size: int = 512
) -> Tensor:
    model.eval()
    out = [
        model(seq[i : i + batch_size], stats[i : i + batch_size])
        for i in range(0, seq.size(0), batch_size)
    ]
    return torch.cat(out, dim=0)


def calibrate(
    model: FlowClassifier,
    val: tuple[Tensor, Tensor, Tensor],
    test: tuple[Tensor, Tensor, Tensor],
) -> dict[str, float]:
    """Fit T on validation, report ECE on test before and after.

    Returns the temperature and both ECEs. A temperature well above 1 means the
    model was over-confident; well below 1, under-confident.
    """
    v_seq, v_stats, v_y = val
    t_seq, t_stats, t_y = test

    scaler = TemperatureScaler().fit(collect_logits(model, v_seq, v_stats), v_y)
    test_logits = collect_logits(model, t_seq, t_stats)
    y = t_y.cpu().numpy()

    before = torch.softmax(test_logits, dim=1).cpu().numpy()
    after = scaler.apply(test_logits).cpu().numpy()
    return {
        "temperature": scaler.temperature,
        "ece_before": expected_calibration_error(before, y),
        "ece_after": expected_calibration_error(after, y),
        "mean_confidence_before": float(before.max(axis=1).mean()),
        "mean_confidence_after": float(after.max(axis=1).mean()),
        "accuracy": float((after.argmax(axis=1) == y).mean()),
    }
