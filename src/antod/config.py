"""YAML experiment configuration.

Unknown keys are a hard error rather than a warning. A silently ignored
``epocs: 200`` would produce a perfectly plausible run at the default 40 epochs,
and nothing downstream would reveal that the configuration on disk is not the
configuration that ran.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from functools import cache
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

from antod.adversarial.attacks import AttackConfig
from antod.adversarial.defenses import DefenseConfig
from antod.data.synth import SynthConfig
from antod.train import TrainConfig

T = TypeVar("T")


@dataclass
class DatasetSection:
    """Where the flows come from."""

    source: str = "synthetic"  # synthetic | real | cached
    path: str = "data/processed/dataset.npz"

    # synthetic generation
    n_flows: int = 12000
    seed: int = 42
    class_weights: tuple[float, float, float] = (0.34, 0.33, 0.33)
    benign_obfuscation_rate: float = 0.35
    min_steps: int = 1
    max_steps: int = 3
    min_packets: int = 8
    allowed_transforms: list[str] | None = None

    # real captures
    max_flows: int | None = None

    def __post_init__(self) -> None:
        if self.source not in {"synthetic", "real", "cached"}:
            raise ValueError(f"unknown dataset source {self.source!r}")

    def synth_config(self) -> SynthConfig:
        return SynthConfig(
            n_flows=self.n_flows,
            seed=self.seed,
            class_weights=tuple(self.class_weights),
            benign_obfuscation_rate=self.benign_obfuscation_rate,
            min_steps=self.min_steps,
            max_steps=self.max_steps,
            min_packets=self.min_packets,
            allowed_transforms=self.allowed_transforms,
        )


@dataclass
class SplitSection:
    val_frac: float = 0.15
    test_frac: float = 0.20
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0 < self.val_frac + self.test_frac < 1:
            raise ValueError("val_frac + test_frac must leave room for a training split")


@dataclass
class OutputSection:
    dir: str = "experiments/runs/default"
    save_checkpoint: bool = True
    save_figures: bool = True

    @property
    def path(self) -> Path:
        return Path(self.dir)


@dataclass
class ExperimentConfig:
    """One complete experiment: data, splits, model, defense, attacks, outputs."""

    name: str = "default"
    seed: int = 42
    dataset: DatasetSection = field(default_factory=DatasetSection)
    split: SplitSection = field(default_factory=SplitSection)
    train: TrainConfig = field(default_factory=TrainConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    baselines: list[str] = field(default_factory=lambda: ["random_forest", "logistic_regression"])
    attacks: list[AttackConfig] = field(default_factory=list)
    output: OutputSection = field(default_factory=OutputSection)

    def to_dict(self) -> dict:
        return asdict(self)


@cache
def _hints(cls: type) -> dict[str, Any]:
    """Resolved annotations for a dataclass.

    ``field.type`` is only a *string* under ``from __future__ import annotations``,
    so nested sections have to be resolved through ``get_type_hints`` -- otherwise
    the recursion below never fires and a nested mapping is passed straight to the
    constructor as a dict.
    """
    return get_type_hints(cls)


def _build(cls: type[T], data: Any, where: str) -> T:
    """Construct a dataclass from a mapping, rejecting unknown keys."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"{where}: expected a mapping, got {type(data).__name__}")

    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"{where}: unknown option(s) {sorted(unknown)}; valid options are {sorted(known)}"
        )

    hints = _hints(cls)
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        target = hints.get(key)
        if isinstance(value, dict) and isinstance(target, type) and is_dataclass(target):
            kwargs[key] = _build(target, value, f"{where}.{key}")
        else:
            kwargs[key] = value
    return cls(**kwargs)


def config_from_dict(data: dict) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from a plain mapping."""
    data = dict(data or {})

    # attacks is a list of mappings, so it needs its own pass
    attacks_raw = data.pop("attacks", []) or []
    if not isinstance(attacks_raw, list):
        raise TypeError("attacks: expected a list of attack specifications")
    attacks = [_build(AttackConfig, a, f"attacks[{i}]") for i, a in enumerate(attacks_raw)]

    # defense carries a nested attack spec
    defense_raw = data.pop("defense", None)
    defense = DefenseConfig()
    if defense_raw is not None:
        if not isinstance(defense_raw, dict):
            raise TypeError("defense: expected a mapping")
        defense_raw = dict(defense_raw)
        attack_raw = defense_raw.pop("attack", None)
        defense = _build(DefenseConfig, defense_raw, "defense")
        if attack_raw is not None:
            defense.attack = _build(AttackConfig, attack_raw, "defense.attack")

    cfg = _build(ExperimentConfig, data, "config")
    cfg.attacks = attacks
    cfg.defense = defense
    return cfg


def load_config(path: str | Path) -> ExperimentConfig:
    """Read a YAML experiment configuration."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no config at {path}")
    return config_from_dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def save_config(cfg: ExperimentConfig, path: str | Path) -> Path:
    """Write the resolved configuration next to the results it produced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    return path
