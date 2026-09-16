"""Metrics, chosen for what a network defender would actually ask.

Accuracy over three balanced classes is the headline number, but on its own it
hides the only failure that matters. A detector that catches every plain attack
and misses every obfuscated one still scores ~67%, and that detector is useless:
the obfuscated flows are precisely the ones a signature engine already lets
through. So :class:`Metrics` also carries, separately:

``obfuscated_recall``
    recall on ``malicious_obfuscated`` alone. This is the number the project
    lives or dies by.

``malicious_recall``
    recall after collapsing both malicious classes together -- "was the attack
    caught at all?", independent of whether its disguise was identified.

``false_positive_rate``
    benign flows flagged as malicious. At realistic traffic volumes this
    dominates whether a detector is deployable, and it is where the obfuscated
    *benign* flows in the dataset do their work: without them a model can buy
    recall by treating every padded flow as hostile, and this is the number that
    would expose it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from antod.data.synth import BENIGN, LABEL_NAMES, MALICIOUS_OBFUSCATED, N_CLASSES


@dataclass
class Metrics:
    """One evaluation of one model on one set of flows."""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    obfuscated_recall: float
    malicious_recall: float
    false_positive_rate: float
    roc_auc_macro: float | None
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)
    n_samples: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        auc = "n/a" if self.roc_auc_macro is None else f"{self.roc_auc_macro:.4f}"
        return (
            f"acc {self.accuracy:.4f}  macro-F1 {self.macro_f1:.4f}  "
            f"obf-recall {self.obfuscated_recall:.4f}  mal-recall {self.malicious_recall:.4f}  "
            f"FPR {self.false_positive_rate:.4f}  AUC {auc}  (n={self.n_samples})"
        )

    def table(self) -> str:
        lines = [f"{'class':<24}{'precision':>11}{'recall':>9}{'f1':>9}{'support':>9}"]
        for name in LABEL_NAMES:
            m = self.per_class[name]
            lines.append(
                f"{name:<24}{m['precision']:>11.4f}{m['recall']:>9.4f}"
                f"{m['f1']:>9.4f}{int(m['support']):>9}"
            )
        return "\n".join(lines)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> Metrics:
    """Score predictions against labels.

    ``y_proba`` is optional; without it the AUC is reported as ``None`` rather
    than silently substituted by a hard-label approximation.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = list(range(N_CLASSES))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        LABEL_NAMES[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": float(support[i]),
        }
        for i in labels
    }

    # collapse the two malicious classes: did we raise an alarm at all?
    true_mal = y_true != BENIGN
    pred_mal = y_pred != BENIGN
    malicious_recall = float(pred_mal[true_mal].mean()) if true_mal.any() else 0.0
    false_positive_rate = float(pred_mal[~true_mal].mean()) if (~true_mal).any() else 0.0

    auc: float | None = None
    if y_proba is not None:
        proba = np.asarray(y_proba, dtype=np.float64)
        # AUC is undefined if a class is absent from y_true, which happens on
        # per-technique probe sets that contain only obfuscated flows.
        if len(np.unique(y_true)) == N_CLASSES:
            auc = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))

    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        obfuscated_recall=per_class[LABEL_NAMES[MALICIOUS_OBFUSCATED]]["recall"],
        malicious_recall=malicious_recall,
        false_positive_rate=false_positive_rate,
        roc_auc_macro=auc,
        per_class=per_class,
        confusion=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        n_samples=int(y_true.size),
    )


def recall_by_group(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """Per-group recall, used for the per-technique and per-profile breakdowns.

    Grouping by obfuscation recipe answers the question the average cannot:
    *which* evasion technique defeats the detector. An overall obfuscated-recall
    of 0.9 is a different result depending on whether the missing 10% is spread
    evenly or is entirely protocol mimicry.
    """
    out: dict[str, float] = {}
    for g in sorted(set(groups.tolist())):
        mask = groups == g
        if not mask.any():
            continue
        out[str(g)] = float((y_pred[mask] == y_true[mask]).mean())
    return out


def clean_vs_attacked(clean: Metrics, attacked: Metrics) -> dict[str, float]:
    """The drop caused by an attack, in the terms that matter."""
    return {
        "accuracy_clean": clean.accuracy,
        "accuracy_attacked": attacked.accuracy,
        "accuracy_drop": clean.accuracy - attacked.accuracy,
        "obfuscated_recall_clean": clean.obfuscated_recall,
        "obfuscated_recall_attacked": attacked.obfuscated_recall,
        "obfuscated_recall_drop": clean.obfuscated_recall - attacked.obfuscated_recall,
        "malicious_recall_clean": clean.malicious_recall,
        "malicious_recall_attacked": attacked.malicious_recall,
        "malicious_recall_drop": clean.malicious_recall - attacked.malicious_recall,
    }
