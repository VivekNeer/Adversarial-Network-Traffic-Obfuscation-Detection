"""Detection models: 1D-CNN, MLP, hybrid and classical baselines.

Importing this package registers every neural architecture, so
``build_model("cnn1d")`` works without the caller knowing which module defines it.
"""

from antod.models.base import FlowClassifier, available_models, build_model, register
from antod.models.baselines import BASELINES, SklearnBaseline, build_baseline
from antod.models.cnn1d import CNN1D, ConvBlock
from antod.models.mlp import MLP, HybridCNNMLP
from antod.models.rnn import GRUClassifier

__all__ = [
    "BASELINES",
    "CNN1D",
    "MLP",
    "ConvBlock",
    "FlowClassifier",
    "GRUClassifier",
    "HybridCNNMLP",
    "SklearnBaseline",
    "available_models",
    "build_baseline",
    "build_model",
    "register",
]
