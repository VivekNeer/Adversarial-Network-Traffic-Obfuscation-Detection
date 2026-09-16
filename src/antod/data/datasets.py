"""Dataset assembly: flows to arrays, stratified splits, scaling and torch wrappers.

Scaling deserves a note. Flow statistics are violently heavy-tailed -- a DDoS
flood and a DNS lookup differ in ``bytes_per_sec`` by six orders of magnitude --
so a plain standardisation leaves the network fitting a handful of outliers. Every
feature therefore passes through ``arcsinh`` before standardisation: it behaves
like the identity near zero, like a logarithm in the tails, is defined for
negative values (skew and log-IAT features need that), and is invertible.

Invertibility is not incidental. The adversarial attacks in
:mod:`antod.adversarial.attacks` perturb the scaled representation but must project
back into *physically legal* traffic -- a flow cannot have a negative packet count
or a padding fraction above 1 -- and that projection only makes sense if we can get
back to original units. :class:`FeatureConstraints` carries those bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from antod.data.features import (
    FEATURE_NAMES,
    N_FEATURES,
    SEQ_CHANNELS,
    SEQ_LEN,
    sequence_tensor,
    stats_vector,
)
from antod.data.synth import LABEL_NAMES, Flow, SynthConfig, generate_dataset


# --------------------------------------------------------------------------- #
# array container
# --------------------------------------------------------------------------- #
@dataclass
class FlowDataset:
    """Both feature views for a set of flows, plus the provenance of each one."""

    seq: np.ndarray  # (N, C, L) float32
    stats: np.ndarray  # (N, F) float32
    y: np.ndarray  # (N,) int64
    profile: np.ndarray  # (N,) str
    recipe: np.ndarray  # (N,) str

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __post_init__(self) -> None:
        n = len(self)
        for name in ("seq", "stats", "profile", "recipe"):
            got = getattr(self, name).shape[0]
            if got != n:
                raise ValueError(f"{name} has {got} rows but y has {n}")
        if self.stats.shape[1] != N_FEATURES:
            raise ValueError(f"expected {N_FEATURES} features, got {self.stats.shape[1]}")

    @classmethod
    def from_flows(cls, flows: list[Flow], length: int = SEQ_LEN) -> FlowDataset:
        return cls(
            seq=np.stack([sequence_tensor(f.packets, length) for f in flows]).astype(np.float32),
            stats=np.stack([stats_vector(f.packets) for f in flows]).astype(np.float32),
            y=np.array([f.label for f in flows], dtype=np.int64),
            profile=np.array([f.profile for f in flows]),
            recipe=np.array([f.recipe for f in flows]),
        )

    def subset(self, idx: np.ndarray) -> FlowDataset:
        return FlowDataset(
            seq=self.seq[idx],
            stats=self.stats[idx],
            y=self.y[idx],
            profile=self.profile[idx],
            recipe=self.recipe[idx],
        )

    def class_counts(self) -> dict[str, int]:
        return {name: int((self.y == i).sum()) for i, name in enumerate(LABEL_NAMES)}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            seq=self.seq,
            stats=self.stats,
            y=self.y,
            profile=self.profile,
            recipe=self.recipe,
            feature_names=np.array(FEATURE_NAMES),
        )

    @classmethod
    def load(cls, path: str | Path) -> FlowDataset:
        with np.load(Path(path), allow_pickle=False) as z:
            saved = list(z["feature_names"])
            if saved != FEATURE_NAMES:
                raise ValueError(
                    "feature set has changed since this dataset was written; regenerate it"
                )
            return cls(
                seq=z["seq"], stats=z["stats"], y=z["y"], profile=z["profile"], recipe=z["recipe"]
            )


# --------------------------------------------------------------------------- #
# scaling
# --------------------------------------------------------------------------- #
class StatsScaler:
    """``arcsinh`` then standardise, with an exact inverse.

    Fit on the training split only -- fitting on everything leaks test-set scale
    information into the model, which on a heavy-tailed feature like
    ``bytes_per_sec`` is enough to flatter the results measurably.
    """

    def __init__(self, clip: float = 8.0) -> None:
        self.clip = clip
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> StatsScaler:
        z = np.arcsinh(x.astype(np.float64))
        self.mean_ = z.mean(axis=0)
        self.scale_ = np.where(z.std(axis=0) < 1e-8, 1.0, z.std(axis=0))
        return self

    def _check(self) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StatsScaler must be fitted before use")

    def transform(self, x: np.ndarray) -> np.ndarray:
        self._check()
        z = (np.arcsinh(x.astype(np.float64)) - self.mean_) / self.scale_
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        self._check()
        return np.sinh(z.astype(np.float64) * self.scale_ + self.mean_).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def state_dict(self) -> dict:
        self._check()
        return {"mean": self.mean_, "scale": self.scale_, "clip": self.clip}

    @classmethod
    def from_state(cls, state: dict) -> StatsScaler:
        obj = cls(clip=float(state["clip"]))
        obj.mean_ = np.asarray(state["mean"])
        obj.scale_ = np.asarray(state["scale"])
        return obj


# --------------------------------------------------------------------------- #
# domain constraints
# --------------------------------------------------------------------------- #
#: Features that are mathematically confined to [0, 1]: proportions and correlations.
_UNIT_INTERVAL = {
    "byte_up_ratio",
    "pkt_up_ratio",
    "unique_size_ratio",
    "mode_size_frac",
    "mtu_frac",
    "min_pkt_frac",
    "size_grid_64",
    "size_grid_128",
    "size_grid_256",
    "iat_autocorr_peak",
    "iat_up_autocorr_peak",
    "idle_frac",
}

#: Features that may legitimately be negative: shape moments and log-scaled times.
_SIGNED = {
    "size_skew",
    "size_kurt",
    "logiat_skew",
    "logiat_kurt",
    "logiat_mean",
    "logiat_std",
    "logiat_min",
    "logiat_max",
    "logiat_median",
    "logiat_p25",
    "logiat_p75",
}

#: Entropies are computed over 32 bins, so 5 bits is the ceiling.
_ENTROPY = {"size_entropy": 5.0, "iat_entropy": 5.0}


class FeatureConstraints:
    """Per-feature legal ranges in original units, used to project attacks.

    An unconstrained gradient step will happily ask for a flow with 1.4 of its
    packets on a 256-byte boundary, or minus three bursts. Such a point is not
    traffic, and counting it as a successful evasion overstates the attack.
    """

    def __init__(self) -> None:
        lo = np.zeros(N_FEATURES, dtype=np.float64)
        hi = np.full(N_FEATURES, np.inf, dtype=np.float64)

        for i, name in enumerate(FEATURE_NAMES):
            if name in _UNIT_INTERVAL:
                hi[i] = 1.0
            elif name in _ENTROPY:
                hi[i] = _ENTROPY[name]
            elif name in _SIGNED:
                lo[i] = -np.inf

        self.lo = lo
        self.hi = hi

    def project(self, x: np.ndarray) -> np.ndarray:
        """Clamp features in *original* units to their legal ranges."""
        return np.clip(x, self.lo, self.hi).astype(np.float32)

    def project_scaled(self, z: np.ndarray, scaler: StatsScaler) -> np.ndarray:
        """Clamp a *scaled* feature matrix by way of the original-unit bounds."""
        return scaler.transform(self.project(scaler.inverse_transform(z)))

    def torch_bounds(self, scaler: StatsScaler, device: str = "cpu") -> tuple[Tensor, Tensor]:
        """The legal box expressed in scaled coordinates, for in-graph clamping.

        ``arcsinh`` and standardisation are both monotone increasing, so the box
        maps to a box and the bounds transform elementwise. That lets PGD clamp
        inside the autograd graph instead of round-tripping through numpy.
        """
        mean, scale = scaler.state_dict()["mean"], scaler.state_dict()["scale"]
        lo = (np.arcsinh(self.lo) - mean) / scale
        hi = (np.arcsinh(self.hi) - mean) / scale
        lo = np.clip(np.nan_to_num(lo, neginf=-scaler.clip), -scaler.clip, scaler.clip)
        hi = np.clip(np.nan_to_num(hi, posinf=scaler.clip), -scaler.clip, scaler.clip)
        return (
            torch.tensor(lo, dtype=torch.float32, device=device),
            torch.tensor(hi, dtype=torch.float32, device=device),
        )


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
@dataclass
class Splits:
    """Train/validation/test partition with the scaler fitted on train only."""

    train: FlowDataset
    val: FlowDataset
    test: FlowDataset
    scaler: StatsScaler

    def summary(self) -> str:
        rows = [f"{'split':<8}{'n':>7}  " + "  ".join(f"{n:>20}" for n in LABEL_NAMES)]
        for name in ("train", "val", "test"):
            ds: FlowDataset = getattr(self, name)
            counts = ds.class_counts()
            rows.append(
                f"{name:<8}{len(ds):>7}  " + "  ".join(f"{counts[n]:>20}" for n in LABEL_NAMES)
            )
        return "\n".join(rows)


def stratified_split(
    ds: FlowDataset,
    val_frac: float = 0.15,
    test_frac: float = 0.20,
    seed: int = 42,
) -> Splits:
    """Class-stratified three-way split, scaler fitted on the training part."""
    idx = np.arange(len(ds))
    train_idx, hold_idx = train_test_split(
        idx, test_size=val_frac + test_frac, stratify=ds.y, random_state=seed
    )
    rel_test = test_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        hold_idx, test_size=rel_test, stratify=ds.y[hold_idx], random_state=seed
    )

    train = ds.subset(train_idx)
    scaler = StatsScaler().fit(train.stats)
    return Splits(train=train, val=ds.subset(val_idx), test=ds.subset(test_idx), scaler=scaler)


# --------------------------------------------------------------------------- #
# torch plumbing
# --------------------------------------------------------------------------- #


class FlowTensorDataset(Dataset):
    """Torch view over a :class:`FlowDataset` with statistics already scaled."""

    def __init__(self, ds: FlowDataset, scaler: StatsScaler) -> None:
        self.seq = torch.from_numpy(np.ascontiguousarray(ds.seq))
        self.stats = torch.from_numpy(scaler.transform(ds.stats))
        self.y = torch.from_numpy(ds.y)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.seq[i], self.stats[i], self.y[i]

    def tensors(self) -> tuple[Tensor, Tensor, Tensor]:
        """The whole split at once -- the attack code needs it as a single batch."""
        return self.seq, self.stats, self.y


def make_loaders(
    splits: Splits, batch_size: int = 128, num_workers: int = 0
) -> dict[str, DataLoader]:
    """Dataloaders for all three splits; only the training one is shuffled."""
    out: dict[str, DataLoader] = {}
    for name in ("train", "val", "test"):
        ds = FlowTensorDataset(getattr(splits, name), splits.scaler)
        out[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            drop_last=False,
        )
    return out


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def build_dataset(cfg: SynthConfig, length: int = SEQ_LEN) -> FlowDataset:
    """Generate flows and extract both feature views."""
    return FlowDataset.from_flows(generate_dataset(cfg), length=length)


__all__ = [
    "SEQ_CHANNELS",
    "SEQ_LEN",
    "FeatureConstraints",
    "FlowDataset",
    "FlowTensorDataset",
    "Splits",
    "StatsScaler",
    "build_dataset",
    "make_loaders",
    "stratified_split",
]
