"""Flow-level dataset synthesis: profiles plus obfuscation, with exact labels.

The generator draws an application profile, optionally applies an obfuscation
recipe, and emits a :class:`Flow`. Because obfuscation is something *we* apply
rather than something we infer, the label is exact and the recipe is recorded --
which is what makes the per-technique detectability analysis in
:mod:`antod.evaluate` possible at all.

Class definition
----------------

======================== ===== =====================================================
``benign``               ``0`` ordinary traffic, obfuscated or not
``malicious_plain``      ``1`` malicious traffic making no attempt to hide
``malicious_obfuscated`` ``2`` malicious traffic reshaped to evade DPI
======================== ===== =====================================================

Note that class ``0`` contains obfuscated flows on purpose (see the discussion in
:mod:`antod.data.obfuscation`): a corporate VPN user is benign and obfuscated at
the same time, and a model that has never seen such a flow will flag every one of
them. ``benign_obfuscation_rate`` controls how much of that traffic is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from antod.data.obfuscation import Recipe, sample_recipe, single_recipe
from antod.data.profiles import BENIGN_PROFILES, MALICIOUS_PROFILES

BENIGN = 0
MALICIOUS_PLAIN = 1
MALICIOUS_OBFUSCATED = 2

LABEL_NAMES = ["benign", "malicious_plain", "malicious_obfuscated"]
N_CLASSES = 3


@dataclass
class Flow:
    """One bidirectional flow: its packets, its label and how it was produced."""

    packets: np.ndarray  # (n, 3) -> [timestamp, size, direction]
    label: int
    profile: str
    recipe: str = "none"
    obfuscated: bool = False

    @property
    def n_packets(self) -> int:
        return int(self.packets.shape[0])

    @property
    def duration(self) -> float:
        return float(self.packets[-1, 0] - self.packets[0, 0])

    @property
    def total_bytes(self) -> float:
        return float(self.packets[:, 1].sum())


@dataclass
class SynthConfig:
    """Knobs for dataset synthesis."""

    n_flows: int = 12000
    seed: int = 42

    #: proportion of [benign, malicious_plain, malicious_obfuscated]
    class_weights: tuple[float, float, float] = (0.34, 0.33, 0.33)

    #: fraction of *benign* flows that are also obfuscated (VPN users, padded TLS)
    benign_obfuscation_rate: float = 0.35

    #: recipe complexity for obfuscated flows
    min_steps: int = 1
    max_steps: int = 3

    #: drop flows shorter than this, they carry too little signal to be meaningful
    min_packets: int = 8

    #: restrict the transform pool, e.g. for an ablation. ``None`` means all of them.
    allowed_transforms: list[str] | None = None

    profile_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        total = sum(self.class_weights)
        if not np.isclose(total, 1.0):
            raise ValueError(f"class_weights must sum to 1, got {total}")


def _pick(rng: np.random.Generator, names: list[str], weights: dict[str, float]) -> str:
    """Weighted profile choice, falling back to uniform when no weights are given."""
    if not weights:
        return str(rng.choice(names))
    w = np.array([weights.get(n, 1.0) for n in names], dtype=np.float64)
    w = w / w.sum()
    return str(rng.choice(names, p=w))


def generate_flow(
    rng: np.random.Generator,
    label: int,
    cfg: SynthConfig,
    force_recipe: Recipe | None = None,
) -> Flow:
    """Generate a single flow of the requested class."""
    if label == BENIGN:
        profile = _pick(rng, list(BENIGN_PROFILES), cfg.profile_weights)
        pkts = BENIGN_PROFILES[profile](rng)
        obfuscate = force_recipe is not None or rng.random() < cfg.benign_obfuscation_rate
    else:
        profile = _pick(rng, list(MALICIOUS_PROFILES), cfg.profile_weights)
        pkts = MALICIOUS_PROFILES[profile](rng)
        obfuscate = label == MALICIOUS_OBFUSCATED

    recipe_name = "none"
    if obfuscate:
        recipe = force_recipe or sample_recipe(
            rng, cfg.min_steps, cfg.max_steps, cfg.allowed_transforms
        )
        pkts = recipe.apply(pkts, rng)
        recipe_name = recipe.describe()

    return Flow(
        packets=pkts,
        label=label,
        profile=profile,
        recipe=recipe_name,
        obfuscated=obfuscate,
    )


def generate_dataset(cfg: SynthConfig) -> list[Flow]:
    """Generate ``cfg.n_flows`` flows with the configured class balance."""
    rng = np.random.default_rng(cfg.seed)

    counts = np.floor(np.array(cfg.class_weights) * cfg.n_flows).astype(int)
    counts[0] += cfg.n_flows - int(counts.sum())  # give the remainder to benign

    flows: list[Flow] = []
    for label, n in enumerate(counts):
        made = 0
        attempts = 0
        while made < n:
            attempts += 1
            if attempts > 50 * max(n, 1):
                raise RuntimeError(
                    f"could not generate {n} flows for class {LABEL_NAMES[label]}; "
                    "min_packets is probably too high"
                )
            flow = generate_flow(rng, label, cfg)
            if flow.n_packets < cfg.min_packets:
                continue
            flows.append(flow)
            made += 1

    rng.shuffle(flows)
    return flows


def generate_technique_probe(
    transform: str,
    n_flows: int,
    seed: int = 7,
    cfg: SynthConfig | None = None,
) -> list[Flow]:
    """Malicious flows obfuscated by exactly one named transform.

    Used to answer "which evasion technique actually defeats the detector?" rather
    than only reporting an average over mixed recipes.
    """
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(seed)
    flows: list[Flow] = []
    guard = 0
    while len(flows) < n_flows:
        guard += 1
        if guard > 50 * max(n_flows, 1):
            raise RuntimeError(f"could not generate probe flows for {transform!r}")
        flow = generate_flow(
            rng, MALICIOUS_OBFUSCATED, cfg, force_recipe=single_recipe(transform, rng)
        )
        if flow.n_packets >= cfg.min_packets:
            flows.append(flow)
    return flows


def summarise(flows: list[Flow]) -> str:
    """Human-readable dataset summary, printed after generation."""
    labels = np.array([f.label for f in flows])
    lines = [f"{len(flows)} flows"]
    for i, name in enumerate(LABEL_NAMES):
        n = int((labels == i).sum())
        lines.append(f"  {name:<22} {n:>6}  ({n / len(flows):.1%})")

    n_obf_benign = sum(1 for f in flows if f.label == BENIGN and f.obfuscated)
    n_benign = int((labels == BENIGN).sum())
    lines.append(f"  of which benign+obfuscated {n_obf_benign} / {n_benign}")

    pkt = np.array([f.n_packets for f in flows])
    lines.append(
        f"  packets per flow: min {pkt.min()}, median {int(np.median(pkt))}, max {pkt.max()}"
    )
    return "\n".join(lines)
