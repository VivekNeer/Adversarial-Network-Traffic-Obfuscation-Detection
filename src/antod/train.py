"""Training loop, checkpointing and early stopping.

Two choices in here are worth justifying.

**Model selection is on validation macro-F1, not accuracy.** With three balanced
classes those usually agree, but they come apart in the case that matters: a model
that collapses ``malicious_obfuscated`` into ``malicious_plain`` loses little
accuracy and a great deal of macro-F1. Selecting on accuracy would quietly prefer
the collapsed model, which is the one failure mode this project exists to avoid.
``selection_metric`` can be set to ``obfuscated_recall`` to push harder still.

**Adversarial training is injected, not imported.** The trainer takes an
``adversary`` callable rather than reaching into :mod:`antod.adversarial`. That
keeps the dependency pointing one way, and it means the adversarially-trained
model is produced by exactly the same loop as the baseline -- so a difference
between them is the defense, not a different training procedure.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from antod.data.datasets import Splits, StatsScaler, make_loaders
from antod.models import FlowClassifier, build_model
from antod.utils.common import get_logger, pick_device, set_seed
from antod.utils.metrics import Metrics, compute_metrics

logger = get_logger(__name__)

#: ``adversary(model, seq, stats, y) -> (seq_adv, stats_adv)``
Adversary = Callable[[FlowClassifier, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]


@dataclass
class TrainConfig:
    """Everything that defines a training run."""

    model: str = "cnn1d"
    model_kwargs: dict = field(default_factory=dict)

    epochs: int = 40
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    label_smoothing: float = 0.05
    scheduler: str = "cosine"  # cosine | plateau | none

    patience: int = 8
    selection_metric: str = "macro_f1"  # macro_f1 | accuracy | obfuscated_recall

    device: str = "auto"
    seed: int = 42
    num_workers: int = 0

    #: fraction of each batch replaced by adversarial examples during training
    adv_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.scheduler not in {"cosine", "plateau", "none"}:
            raise ValueError(f"unknown scheduler {self.scheduler!r}")
        if self.selection_metric not in {"macro_f1", "accuracy", "obfuscated_recall"}:
            raise ValueError(f"unknown selection_metric {self.selection_metric!r}")
        if not 0.0 <= self.adv_ratio <= 1.0:
            raise ValueError("adv_ratio must be in [0, 1]")


@dataclass
class TrainResult:
    """Outcome of a run: the best model, its validation score, and the curves."""

    model: FlowClassifier
    scaler: StatsScaler
    config: TrainConfig
    history: list[dict[str, float]]
    best_epoch: int
    best_val: Metrics
    train_seconds: float

    def curves(self) -> dict[str, list[float]]:
        """History transposed, ready for plotting."""
        keys = self.history[0].keys() if self.history else []
        return {k: [float(h[k]) for h in self.history] for k in keys}


def _make_scheduler(cfg: TrainConfig, optimiser: torch.optim.Optimizer):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.epochs)
    if cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="max", factor=0.5, patience=3
        )
    return None


@torch.no_grad()
def predict_loader(
    model: FlowClassifier, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a split through the model. Returns ``(y_true, y_pred, y_proba)``."""
    model.eval()
    trues, preds, probas = [], [], []
    for seq, stats, y in loader:
        logits = model(seq.to(device), stats.to(device))
        proba = torch.softmax(logits, dim=1)
        trues.append(y.numpy())
        preds.append(proba.argmax(dim=1).cpu().numpy())
        probas.append(proba.cpu().numpy())
    return np.concatenate(trues), np.concatenate(preds), np.concatenate(probas)


def evaluate_model(model: FlowClassifier, loader: DataLoader, device: torch.device) -> Metrics:
    y_true, y_pred, y_proba = predict_loader(model, loader, device)
    return compute_metrics(y_true, y_pred, y_proba)


class Trainer:
    """Standard supervised loop with early stopping and optional adversarial training."""

    def __init__(
        self,
        cfg: TrainConfig,
        adversary: Adversary | None = None,
    ) -> None:
        self.cfg = cfg
        self.adversary = adversary
        self.device = pick_device(cfg.device)

    def fit(self, splits: Splits, model: FlowClassifier | None = None) -> TrainResult:
        cfg = self.cfg
        set_seed(cfg.seed)

        loaders = make_loaders(splits, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        model = (model or build_model(cfg.model, **cfg.model_kwargs)).to(self.device)
        logger.info("%s on %s", model.describe(), self.device)

        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = _make_scheduler(cfg, optimiser)

        history: list[dict[str, float]] = []
        best_score = -np.inf
        best_state: dict | None = None
        best_epoch = -1
        best_val: Metrics | None = None
        stale = 0
        started = time.time()

        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_acc = self._train_epoch(model, loaders["train"], criterion, optimiser)
            val = evaluate_model(model, loaders["val"], self.device)
            score = getattr(val, cfg.selection_metric)

            if scheduler is not None:
                if cfg.scheduler == "plateau":
                    scheduler.step(score)
                else:
                    scheduler.step()

            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_acc": val.accuracy,
                    "val_macro_f1": val.macro_f1,
                    "val_obf_recall": val.obfuscated_recall,
                    "val_fpr": val.false_positive_rate,
                    "lr": float(optimiser.param_groups[0]["lr"]),
                }
            )

            improved = score > best_score + 1e-5
            if improved:
                best_score, best_epoch, best_val = score, epoch, val
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1

            logger.info(
                "epoch %3d/%d  loss %.4f  train-acc %.4f  val-acc %.4f  "
                "macro-F1 %.4f  obf-recall %.4f%s",
                epoch,
                cfg.epochs,
                train_loss,
                train_acc,
                val.accuracy,
                val.macro_f1,
                val.obfuscated_recall,
                "  *" if improved else "",
            )

            if stale >= cfg.patience:
                logger.info("early stop: no improvement for %d epochs", cfg.patience)
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        assert best_val is not None, "training ran zero epochs"

        return TrainResult(
            model=model,
            scaler=splits.scaler,
            config=cfg,
            history=history,
            best_epoch=best_epoch,
            best_val=best_val,
            train_seconds=time.time() - started,
        )

    def _train_epoch(
        self,
        model: FlowClassifier,
        loader: DataLoader,
        criterion: nn.Module,
        optimiser: torch.optim.Optimizer,
    ) -> tuple[float, float]:
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0

        for seq, stats, y in loader:
            seq = seq.to(self.device)
            stats = stats.to(self.device)
            y = y.to(self.device)

            if self.adversary is not None:
                seq, stats, y = self._mix_adversarial(model, seq, stats, y)

            optimiser.zero_grad(set_to_none=True)
            logits = model(seq, stats)
            loss = criterion(logits, y)
            loss.backward()
            if self.cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
            optimiser.step()

            total_loss += float(loss.detach()) * y.size(0)
            correct += int((logits.detach().argmax(dim=1) == y).sum())
            seen += y.size(0)

        return total_loss / max(seen, 1), correct / max(seen, 1)

    def _mix_adversarial(
        self, model: FlowClassifier, seq: Tensor, stats: Tensor, y: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Replace a share of the batch with adversarial examples.

        A *share* rather than all of it. Training purely on perturbed inputs is
        the standard way to trade clean accuracy for robustness, and a detector
        that has become worse on ordinary traffic to survive an attack has not
        obviously been improved. Keeping clean examples in the batch means the
        robustness number and the clean number can both be read honestly.
        """
        assert self.adversary is not None
        k = int(round(seq.size(0) * self.cfg.adv_ratio))
        if k <= 0:
            return seq, stats, y

        was_training = model.training
        model.eval()  # attack the deterministic model, not a dropout sample
        adv_seq, adv_stats = self.adversary(model, seq[:k], stats[:k], y[:k])
        if was_training:
            model.train()

        return (
            torch.cat([adv_seq.detach(), seq[k:]], dim=0),
            torch.cat([adv_stats.detach(), stats[k:]], dim=0),
            y,
        )


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str | Path, result: TrainResult, extra: dict | None = None) -> Path:
    """Persist weights, the fitted scaler and the config as one file.

    The scaler travels with the weights on purpose: a model applied with the wrong
    feature scaling produces plausible-looking nonsense rather than an error.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": result.config.model,
            "model_kwargs": result.config.model_kwargs,
            "state_dict": result.model.state_dict(),
            "scaler": result.scaler.state_dict(),
            "config": asdict(result.config),
            "history": result.history,
            "best_epoch": result.best_epoch,
            "best_val": result.best_val.to_dict(),
            "extra": extra or {},
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path, device: str = "auto"
) -> tuple[FlowClassifier, StatsScaler, dict]:
    """Rebuild a model and its scaler from a checkpoint."""
    dev = pick_device(device)
    blob = torch.load(Path(path), map_location=dev, weights_only=False)
    model = build_model(blob["model_name"], **blob.get("model_kwargs", {}))
    model.load_state_dict(blob["state_dict"])
    model.to(dev).eval()
    return model, StatsScaler.from_state(blob["scaler"]), blob
