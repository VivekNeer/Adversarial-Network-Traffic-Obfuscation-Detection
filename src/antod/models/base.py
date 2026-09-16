"""Common interface for the detection models, plus a registry.

Every model takes **both** feature views and ignores whichever it does not use:

.. code-block:: python

    logits = model(seq, stats)   # seq: (B, 4, 128), stats: (B, 51)

The uniform signature is deliberate. It lets one trainer, one attack loop and one
evaluation routine drive a pure sequence model, a pure statistics model and a
hybrid without special-casing any of them -- and it means an attack can be aimed
at either input surface simply by choosing which tensor to perturb.

Each model also declares which surfaces it actually reads through
:attr:`FlowClassifier.uses_seq` and :attr:`FlowClassifier.uses_stats`, so the
attack code can refuse to report a meaningless result (a sequence-space attack on
a model that never looks at the sequence is guaranteed to "fail", which would
otherwise read as robustness).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch import Tensor, nn

from antod.data.features import N_FEATURES, SEQ_CHANNELS, SEQ_LEN
from antod.data.synth import N_CLASSES


class FlowClassifier(nn.Module, ABC):
    """A three-class flow classifier reading the sequence view, the statistics view, or both."""

    #: does this model read the packet-sequence input?
    uses_seq: bool = False
    #: does this model read the flow-statistics input?
    uses_stats: bool = False

    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.n_classes = n_classes

    @abstractmethod
    def forward(self, seq: Tensor, stats: Tensor) -> Tensor:
        """Return raw class logits of shape ``(B, n_classes)``."""

    @torch.no_grad()
    def predict(self, seq: Tensor, stats: Tensor) -> Tensor:
        self.eval()
        return self(seq, stats).argmax(dim=1)

    @torch.no_grad()
    def probabilities(self, seq: Tensor, stats: Tensor) -> Tensor:
        self.eval()
        return torch.softmax(self(seq, stats), dim=1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        surfaces = [
            name
            for name, used in (("sequence", self.uses_seq), ("statistics", self.uses_stats))
            if used
        ]
        return (
            f"{type(self).__name__}: {self.n_parameters():,} trainable parameters, "
            f"reads {' + '.join(surfaces)}"
        )


def _init_weights(module: nn.Module) -> None:
    """He initialisation for the ReLU stacks, zero bias, standard BN start."""
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d,)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, Callable[..., FlowClassifier]] = {}


def register(name: str) -> Callable[[Callable[..., FlowClassifier]], Callable[..., FlowClassifier]]:
    def wrap(factory: Callable[..., FlowClassifier]) -> Callable[..., FlowClassifier]:
        if name in _REGISTRY:
            raise ValueError(f"model {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory

    return wrap


def build_model(name: str, **kwargs) -> FlowClassifier:
    """Instantiate a registered model by name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_models() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "N_CLASSES",
    "N_FEATURES",
    "SEQ_CHANNELS",
    "SEQ_LEN",
    "FlowClassifier",
    "_init_weights",
    "available_models",
    "build_model",
    "register",
]
