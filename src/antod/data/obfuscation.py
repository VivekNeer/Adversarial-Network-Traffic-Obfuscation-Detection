"""Obfuscation and DPI-evasion transforms applied to packet-level flow records.

Each transform takes an ``(n, 3)`` packet array (see :mod:`antod.data.profiles`)
and returns a transformed array. They model the *observable effect* of the
evasion techniques catalogued in the DPI-evasion literature:

``block_padding``          pad every record to a block boundary, erasing the size fingerprint
``random_padding``         add random padding, widening the size distribution
``timing_jitter``          add random delay to every gap, blurring periodicity
``constant_rate_shaping``  force packets onto a fixed cadence, destroying timing entirely
``fragmentation``          split records below a threshold so bursts look like many small packets
``tunnel_encapsulation``   wrap the flow in a VPN/TLS tunnel: handshake, per-record overhead, latency
``protocol_mimicry``       resample the size and timing distribution from a benign profile
``dummy_injection``        inject chaff packets to break burst structure

Two design decisions matter for the integrity of the experiment.

First, every transform preserves the **payload volume** that actually has to
cross the wire. An attacker can reshape traffic but cannot decide not to send
the bytes, so ``protocol_mimicry`` emits as many mimicked packets as are needed
to carry the original per-direction byte count. This is the domain constraint
that keeps the detection problem well posed rather than hopeless.

Second, these transforms are applied to **both** benign and malicious flows when
building the dataset. Benign traffic really does get tunnelled -- that is what a
corporate VPN is -- so a detector allowed to equate "obfuscated" with "malicious"
would learn a shortcut that collapses the moment it meets a VPN user. Labelling
obfuscated benign flows as benign forces the model to read the behaviour through
the obfuscation instead of merely detecting the obfuscation.

Scope: these functions rewrite feature arrays and synthetic packet records. They
do not touch sockets and are not a traffic-rewriting tool. They exist to produce
labelled training data for a detector and to measure how far that detector can
be pushed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from antod.data.profiles import (
    ACK_HI,
    ACK_LO,
    BENIGN_PROFILES,
    DOWN,
    MIN_PKT,
    MTU,
    UP,
    _pkts,
)

#: Hard ceiling on packets per flow after obfuscation. Constant-rate shaping and
#: fragmentation are both expansive; without a cap a single flow can blow up.
MAX_PACKETS = 4096


# --------------------------------------------------------------------------- #
# size-domain transforms
# --------------------------------------------------------------------------- #
def block_padding(pkts: np.ndarray, rng: np.random.Generator, block: int = 256) -> np.ndarray:
    """Pad every packet up to the next multiple of ``block`` bytes.

    This is the cheapest and most common size-obfuscation: it quantises the size
    distribution onto a handful of levels, which removes per-application size
    fingerprints but leaves the packet *count* and timing untouched.
    """
    out = pkts.copy()
    padded = np.ceil(out[:, 1] / block) * block
    out[:, 1] = np.minimum(padded, MTU)
    return out


def random_padding(pkts: np.ndarray, rng: np.random.Generator, max_pad: int = 400) -> np.ndarray:
    """Add uniform random padding of up to ``max_pad`` bytes to each packet."""
    out = pkts.copy()
    pad = rng.integers(0, max_pad + 1, size=out.shape[0])
    out[:, 1] = np.minimum(out[:, 1] + pad, MTU)
    return out


def fragmentation(pkts: np.ndarray, rng: np.random.Generator, max_size: int = 400) -> np.ndarray:
    """Split every packet larger than ``max_size`` into sub-MTU fragments.

    Fragmentation is a classic IDS/DPI evasion: a signature spanning a record
    boundary is never seen by an engine that does not reassemble. Statistically
    it trades a few large packets for many small ones at constant byte volume.
    """
    times: list[float] = []
    sizes: list[float] = []
    dirs: list[float] = []

    for t, s, d in pkts:
        if s <= max_size:
            times.append(t)
            sizes.append(s)
            dirs.append(d)
            continue
        n_frag = int(np.ceil(s / max_size))
        remaining = s
        for i in range(n_frag):
            piece = min(max_size, remaining)
            remaining -= piece
            times.append(t + i * rng.uniform(5e-5, 5e-4))
            sizes.append(max(piece, MIN_PKT))
            dirs.append(d)

    return _pkts(times, sizes, dirs)


# --------------------------------------------------------------------------- #
# timing-domain transforms
# --------------------------------------------------------------------------- #
def timing_jitter(pkts: np.ndarray, rng: np.random.Generator, scale: float = 0.05) -> np.ndarray:
    """Add a non-negative random delay to every inter-arrival gap.

    ``scale`` is in seconds. Delays are exponential and strictly positive so the
    packet order is preserved -- a real sender can hold a packet back but cannot
    send it earlier than it was produced.
    """
    out = pkts.copy()
    gaps = np.diff(out[:, 0], prepend=out[0, 0])
    gaps = gaps + rng.exponential(scale, size=gaps.shape[0])
    gaps[0] = 0.0
    out[:, 0] = np.cumsum(gaps)
    return out


def constant_rate_shaping(
    pkts: np.ndarray,
    rng: np.random.Generator,
    interval: float | None = None,
    cell_size: int = 512,
) -> np.ndarray:
    """Force the flow onto a fixed cadence, sending a dummy cell when idle.

    This is the strongest timing countermeasure available to an attacker (it is
    what link padding in anonymity systems does) and it erases the inter-arrival
    time channel completely: every gap becomes ``interval``. The cost is dummy
    traffic, so a shaped flow carries a bandwidth overhead the attacker must accept.

    Each direction is shaped independently. If ``interval`` is not given it is
    derived from the flow's own median gap, which keeps the overhead bounded.
    """
    if interval is None:
        gaps = np.diff(pkts[:, 0])
        med = float(np.median(gaps)) if gaps.size else 0.01
        interval = float(np.clip(med, 1e-4, 0.5))

    times: list[float] = []
    sizes: list[float] = []
    dirs: list[float] = []

    budget = MAX_PACKETS // 2
    for direction in (UP, DOWN):
        sel = pkts[pkts[:, 2] == direction]
        if sel.shape[0] == 0:
            continue
        queue = list(sel[:, 1])
        arrivals = list(sel[:, 0])
        t = float(sel[0, 0])
        end = float(sel[-1, 0])
        slots = 0
        qi = 0
        while (qi < len(queue) or t <= end) and slots < budget:
            if qi < len(queue) and arrivals[qi] <= t + 1e-12:
                # a real packet is waiting: send it in this cell, padded to cell_size
                sizes.append(max(min(float(queue[qi]), MTU), cell_size))
                qi += 1
            else:
                sizes.append(float(cell_size))  # dummy cell
            times.append(t)
            dirs.append(direction)
            t += interval
            slots += 1
        # anything still queued goes out back-to-back at the cell rate
        while qi < len(queue) and slots < budget:
            sizes.append(max(min(float(queue[qi]), MTU), cell_size))
            times.append(t)
            dirs.append(direction)
            qi += 1
            t += interval
            slots += 1

    if not times:
        return pkts.copy()
    return _pkts(times, sizes, dirs)


def dummy_injection(pkts: np.ndarray, rng: np.random.Generator, rate: float = 0.3) -> np.ndarray:
    """Inject chaff packets amounting to ``rate`` times the original packet count.

    Chaff is drawn uniformly across the flow lifetime with sizes sampled from the
    flow's own size distribution, so it breaks up burst structure without
    introducing an obviously foreign size mode.
    """
    n_new = int(round(pkts.shape[0] * rate))
    if n_new <= 0:
        return pkts.copy()

    span = float(pkts[-1, 0] - pkts[0, 0]) or 1e-3
    new_t = pkts[0, 0] + rng.uniform(0, span, n_new)
    new_s = rng.choice(pkts[:, 1], size=n_new, replace=True)
    new_d = rng.choice([UP, DOWN], size=n_new)

    times = list(pkts[:, 0]) + list(new_t)
    sizes = list(pkts[:, 1]) + list(new_s)
    dirs = list(pkts[:, 2]) + list(new_d)
    return _pkts(times, sizes, dirs)


# --------------------------------------------------------------------------- #
# encapsulation and mimicry
# --------------------------------------------------------------------------- #
def tunnel_encapsulation(
    pkts: np.ndarray,
    rng: np.random.Generator,
    overhead: int = 48,
    handshake: bool = True,
    added_latency: float = 0.0,
) -> np.ndarray:
    """Wrap the flow in a VPN/TLS tunnel.

    Three observable effects: a handshake prologue of a few characteristic
    packets, a fixed per-record ``overhead`` (WireGuard 32 B, OpenVPN/TLS 40-80 B),
    and re-fragmentation of anything that now exceeds the MTU. Tunnelling is what
    makes port-based and signature-based classification fail outright, since every
    tunnelled flow looks like the same opaque protocol from the outside.
    """
    times: list[float] = []
    sizes: list[float] = []
    dirs: list[float] = []

    t0 = 0.0
    if handshake:
        t = 0.0
        for size, direction in (
            (rng.uniform(180, 320), UP),
            (rng.uniform(900, 1400), DOWN),
            (rng.uniform(80, 160), UP),
            (rng.uniform(80, 200), DOWN),
        ):
            times.append(t)
            sizes.append(size)
            dirs.append(direction)
            t += rng.uniform(0.005, 0.05)
        t0 = t + rng.uniform(0.001, 0.02)

    for t, s, d in pkts:
        wrapped = s + overhead
        at = t0 + t + added_latency * rng.uniform(0.8, 1.2)
        if wrapped <= MTU:
            times.append(at)
            sizes.append(wrapped)
            dirs.append(d)
        else:  # spills into a second record
            times.append(at)
            sizes.append(MTU)
            dirs.append(d)
            times.append(at + rng.uniform(5e-5, 3e-4))
            sizes.append(max(wrapped - MTU, MIN_PKT))
            dirs.append(d)

    return _pkts(times, sizes, dirs)


def protocol_mimicry(
    pkts: np.ndarray,
    rng: np.random.Generator,
    template: str | None = None,
) -> np.ndarray:
    """Reshape the flow to match a benign profile while preserving byte volume.

    The attacker draws packet sizes and inter-arrival times from a benign template
    (say, video streaming) and sends as many packets as are needed to move the
    original per-direction byte count. Size and timing marginals therefore match
    benign traffic almost exactly; what survives is the *volume and asymmetry* of
    the underlying activity, which is the only signal left for the detector.

    This is the hardest transform in the suite and is expected to cost the most
    recall, which is precisely why it is worth measuring separately.
    """
    if template is None:
        template = str(rng.choice(list(BENIGN_PROFILES)))
    tmpl = BENIGN_PROFILES[template](rng)

    times: list[float] = []
    sizes: list[float] = []
    dirs: list[float] = []

    for direction in (UP, DOWN):
        orig = pkts[pkts[:, 2] == direction]
        if orig.shape[0] == 0:
            continue
        target_bytes = float(orig[:, 1].sum())

        pool = tmpl[tmpl[:, 2] == direction]
        size_pool = pool[:, 1] if pool.shape[0] >= 5 else tmpl[:, 1]
        gap_pool = np.diff(pool[:, 0]) if pool.shape[0] >= 5 else np.diff(tmpl[:, 0])
        if gap_pool.size == 0:
            gap_pool = np.array([1e-3])

        # how many mimicked packets to carry target_bytes
        mean_size = max(float(size_pool.mean()), MIN_PKT)
        n = int(np.ceil(target_bytes / mean_size))
        n = int(np.clip(n, 1, MAX_PACKETS // 2))

        drawn = rng.choice(size_pool, size=n, replace=True).astype(np.float64)
        # rescale so the volume constraint holds exactly, then re-clip to legal sizes
        total = float(drawn.sum())
        if total > 0:
            drawn = np.clip(drawn * (target_bytes / total), MIN_PKT, MTU)

        gaps = rng.choice(gap_pool, size=n, replace=True).astype(np.float64)
        gaps = np.abs(gaps)
        gaps[0] = 0.0
        t = np.cumsum(gaps) + float(orig[0, 0])

        times += list(t)
        sizes += list(drawn)
        dirs += [direction] * n

    if not times:
        return pkts.copy()
    return _pkts(times, sizes, dirs)


# --------------------------------------------------------------------------- #
# recipes
# --------------------------------------------------------------------------- #
Transform = Callable[..., np.ndarray]

TRANSFORMS: dict[str, Transform] = {
    "block_padding": block_padding,
    "random_padding": random_padding,
    "fragmentation": fragmentation,
    "timing_jitter": timing_jitter,
    "constant_rate_shaping": constant_rate_shaping,
    "dummy_injection": dummy_injection,
    "tunnel_encapsulation": tunnel_encapsulation,
    "protocol_mimicry": protocol_mimicry,
}


def _sample_params(name: str, rng: np.random.Generator) -> dict:
    """Draw a plausible intensity for one transform."""
    if name == "block_padding":
        return {"block": int(rng.choice([64, 128, 256, 512]))}
    if name == "random_padding":
        return {"max_pad": int(rng.integers(64, 600))}
    if name == "fragmentation":
        return {"max_size": int(rng.choice([200, 300, 400, 576]))}
    if name == "timing_jitter":
        return {"scale": float(rng.choice([0.005, 0.02, 0.05, 0.2]))}
    if name == "constant_rate_shaping":
        return {
            "interval": float(rng.choice([0.005, 0.02, 0.05, 0.1])),
            "cell_size": int(rng.choice([256, 512, 1024])),
        }
    if name == "dummy_injection":
        return {"rate": float(rng.uniform(0.1, 0.8))}
    if name == "tunnel_encapsulation":
        return {
            "overhead": int(rng.choice([32, 48, 64, 80])),
            "handshake": bool(rng.random() < 0.8),
            "added_latency": float(rng.choice([0.0, 0.005, 0.02])),
        }
    if name == "protocol_mimicry":
        return {"template": str(rng.choice(list(BENIGN_PROFILES)))}
    return {}


# Transform order is not arbitrary: mimicry must reshape the original flow before
# anything else touches it, and tunnelling is applied last because in reality the
# tunnel sees whatever the application already produced.
_ORDER = [
    "protocol_mimicry",
    "fragmentation",
    "block_padding",
    "random_padding",
    "dummy_injection",
    "timing_jitter",
    "constant_rate_shaping",
    "tunnel_encapsulation",
]


@dataclass
class Recipe:
    """A named, ordered set of obfuscation transforms with sampled intensities."""

    steps: list[tuple[str, dict]] = field(default_factory=list)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.steps)

    def apply(self, pkts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = pkts
        for name, params in self.steps:
            out = TRANSFORMS[name](out, rng, **params)
            if out.shape[0] > MAX_PACKETS:
                out = out[:MAX_PACKETS]
        return out

    def describe(self) -> str:
        return "+".join(self.names) if self.steps else "none"


def sample_recipe(
    rng: np.random.Generator,
    min_steps: int = 1,
    max_steps: int = 3,
    allowed: list[str] | None = None,
) -> Recipe:
    """Pick a random obfuscation recipe of ``min_steps``..``max_steps`` transforms."""
    pool = allowed if allowed is not None else list(TRANSFORMS)
    k = int(rng.integers(min_steps, max_steps + 1))
    k = min(k, len(pool))
    chosen = list(rng.choice(pool, size=k, replace=False))
    chosen.sort(key=_ORDER.index)
    return Recipe([(name, _sample_params(name, rng)) for name in chosen])


def single_recipe(name: str, rng: np.random.Generator) -> Recipe:
    """A one-transform recipe, used for the per-technique detectability breakdown."""
    return Recipe([(name, _sample_params(name, rng))])


__all__ = [
    "ACK_HI",
    "ACK_LO",
    "MAX_PACKETS",
    "TRANSFORMS",
    "Recipe",
    "block_padding",
    "constant_rate_shaping",
    "dummy_injection",
    "fragmentation",
    "protocol_mimicry",
    "random_padding",
    "sample_recipe",
    "single_recipe",
    "timing_jitter",
    "tunnel_encapsulation",
]
