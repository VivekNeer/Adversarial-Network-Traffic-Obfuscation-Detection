"""Two views of a flow: a packet sequence for the CNN, flow statistics for the MLP.

Sequence view
-------------
``sequence_tensor`` returns a ``(C, L)`` array -- the layout a 1D convolution
expects -- holding the first ``L`` packets of the flow across four channels:

===== ===============================================================
ch 0  signed packet size, ``size/MTU`` carrying the sign of direction
ch 1  log-scaled inter-arrival time
ch 2  direction, ``+1`` up / ``-1`` down
ch 3  validity mask, ``1`` for a real packet and ``0`` for padding
===== ===============================================================

Signing the size in channel 0 means one filter can learn an asymmetry pattern
("large down, small up" = a download; the reverse = exfiltration) instead of
needing to correlate two channels. The mask channel exists because zero-padding
a short flow is otherwise indistinguishable from a real packet of size zero at
zero delay.

Statistics view
---------------
``flow_statistics`` returns 51 aggregate features. Most are the usual
volume/size/timing moments used by flow classifiers, but three groups are there
specifically to expose obfuscation:

* **size-grid occupancy** (``size_grid_64/128/256``) -- the fraction of packets
  sitting exactly on a power-of-two boundary. Block padding drives this to 1.0,
  and nothing in natural traffic does that.
* **size diversity** (``unique_size_ratio``, ``mode_size_frac``) -- padding
  collapses a continuous size distribution onto a few levels.
* **timing regularity** (``iat_cv``, ``iat_autocorr_peak``) -- a C2 beacon has a
  low coefficient of variation because it is a metronome; constant-rate shaping
  pushes it to ~0 and takes the autocorrelation peak with it.

The statistics are deliberately human-readable so that a tree ensemble trained on
them can be asked *which* of these actually carries the signal, rather than the
deep model being the only account we can give of the result.
"""

from __future__ import annotations

import numpy as np

from antod.data.profiles import MIN_PKT, MTU

#: packets kept in the sequence view; ~90% of generated flows are shorter than this
SEQ_LEN = 128
SEQ_CHANNELS = 4

_EPS = 1e-9
_IAT_FLOOR = 1e-6


# --------------------------------------------------------------------------- #
# sequence view
# --------------------------------------------------------------------------- #
def sequence_tensor(pkts: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    """Encode the first ``length`` packets of a flow as a ``(SEQ_CHANNELS, length)`` array."""
    n = min(pkts.shape[0], length)
    out = np.zeros((SEQ_CHANNELS, length), dtype=np.float32)

    size = pkts[:n, 1]
    direction = pkts[:n, 2]
    iat = np.diff(pkts[:n, 0], prepend=pkts[0, 0])

    out[0, :n] = (size / MTU) * direction
    # inter-arrival times span microseconds to minutes; log10 makes that linear.
    # The /6 puts the useful range roughly in [-1, 1] without a fitted scaler.
    out[1, :n] = np.log10(np.maximum(iat, _IAT_FLOOR)) / 6.0
    out[2, :n] = direction
    out[3, :n] = 1.0
    return out


# --------------------------------------------------------------------------- #
# statistics view
# --------------------------------------------------------------------------- #
def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def _moments(x: np.ndarray, prefix: str) -> dict[str, float]:
    """Mean, spread, extremes, quartiles, skew and kurtosis of one series."""
    if x.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_p25": 0.0,
            f"{prefix}_p75": 0.0,
            f"{prefix}_skew": 0.0,
            f"{prefix}_kurt": 0.0,
        }
    mean = float(x.mean())
    std = float(x.std())
    centred = x - mean
    return {
        f"{prefix}_mean": _safe(mean),
        f"{prefix}_std": _safe(std),
        f"{prefix}_min": _safe(x.min()),
        f"{prefix}_max": _safe(x.max()),
        f"{prefix}_median": _safe(np.median(x)),
        f"{prefix}_p25": _safe(np.percentile(x, 25)),
        f"{prefix}_p75": _safe(np.percentile(x, 75)),
        f"{prefix}_skew": _safe((centred**3).mean() / (std**3 + _EPS)),
        f"{prefix}_kurt": _safe((centred**4).mean() / (std**4 + _EPS) - 3.0),
    }


def _entropy(x: np.ndarray, bins: int, lo: float, hi: float) -> float:
    """Shannon entropy in bits of a fixed-support histogram of ``x``."""
    if x.size == 0:
        return 0.0
    hist, _ = np.histogram(x, bins=bins, range=(lo, hi))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return _safe(-(p * np.log2(p)).sum())


def _autocorr_peak(x: np.ndarray) -> float:
    """Strongest self-similarity of a series at any non-zero lag, in ``[0, 1]``.

    A periodic beacon repeats its inter-arrival pattern, so the autocorrelation
    of its IAT series has a clear peak. Jittered or naturally bursty traffic does not.
    """
    if x.size < 8:
        return 0.0
    y = x - x.mean()
    denom = float((y * y).sum())
    if denom <= _EPS:
        return 1.0  # a perfectly constant series is maximally self-similar
    full = np.correlate(y, y, mode="full")[y.size - 1 :] / denom
    lags = full[1 : max(2, y.size // 2)]
    return _safe(np.abs(lags).max()) if lags.size else 0.0


def _grid_fraction(sizes: np.ndarray, block: int) -> float:
    """Fraction of *unsaturated* packets whose size is an exact multiple of ``block``.

    The direct fingerprint of block padding: natural traffic lands on a 256-byte
    boundary about 1/256 of the time, padded traffic lands there always.

    Packets sitting at the MTU are excluded from the ratio. A padder cannot grow a
    1300-byte record to the next 256-byte boundary without exceeding the MTU, so
    such a record is clamped at 1500 -- which is not a multiple of 64, 128 or 256.
    Counting those in the denominator would dilute the fingerprint in exactly the
    bulk-transfer flows where padding is most worth detecting, so a saturated
    packet is treated as carrying no evidence about the block size either way.
    """
    unsaturated = sizes[sizes < MTU]
    if unsaturated.size == 0:
        return 0.0
    return _safe(np.mean(np.mod(unsaturated, block) == 0))


def _burst_stats(times: np.ndarray, idle: float) -> dict[str, float]:
    """Split the flow at gaps longer than ``idle`` and describe the resulting bursts."""
    if times.size < 2:
        return {"n_bursts": 1.0, "burst_pkts_mean": float(times.size), "idle_frac": 0.0}
    gaps = np.diff(times)
    breaks = np.flatnonzero(gaps > idle)
    n_bursts = float(breaks.size + 1)
    span = float(times[-1] - times[0])
    idle_time = float(gaps[gaps > idle].sum())
    return {
        "n_bursts": n_bursts,
        "burst_pkts_mean": _safe(times.size / n_bursts),
        "idle_frac": _safe(idle_time / span) if span > 0 else 0.0,
    }


def flow_statistics(pkts: np.ndarray) -> dict[str, float]:
    """Compute the 51 aggregate features described in the module docstring."""
    t = pkts[:, 0]
    size = pkts[:, 1]
    direction = pkts[:, 2]

    up = size[direction > 0]
    down = size[direction < 0]
    t_up = t[direction > 0]

    n = float(size.size)
    duration = float(t[-1] - t[0])
    total = float(size.sum())

    iat = np.diff(t)
    iat_up = np.diff(t_up) if t_up.size >= 2 else np.array([])

    feats: dict[str, float] = {
        # ---- volume and shape of the conversation ----
        "n_packets": n,
        "log_n_packets": _safe(np.log1p(n)),
        "duration": _safe(duration),
        "log_duration": _safe(np.log1p(duration)),
        "total_bytes": _safe(total),
        "log_total_bytes": _safe(np.log1p(total)),
        "bytes_up": _safe(up.sum()),
        "bytes_down": _safe(down.sum()),
        "byte_up_ratio": _safe(up.sum() / (total + _EPS)),
        "pkt_up_ratio": _safe(up.size / n),
        "pkts_per_sec": _safe(n / (duration + _EPS)),
        "bytes_per_sec": _safe(total / (duration + _EPS)),
        "mean_pkt_size": _safe(total / n),
        # ---- size distribution ----
        "size_entropy": _entropy(size, bins=32, lo=MIN_PKT, hi=MTU),
        "unique_size_ratio": _safe(np.unique(size).size / n),
        "mode_size_frac": _safe(np.bincount(size.astype(np.int64)).max() / n),
        "mtu_frac": _safe(np.mean(size >= MTU - 1)),
        "min_pkt_frac": _safe(np.mean(size <= MIN_PKT + 20)),
        "size_grid_64": _grid_fraction(size, 64),
        "size_grid_128": _grid_fraction(size, 128),
        "size_grid_256": _grid_fraction(size, 256),
        "size_up_mean": _safe(up.mean()) if up.size else 0.0,
        "size_up_std": _safe(up.std()) if up.size else 0.0,
        "size_down_mean": _safe(down.mean()) if down.size else 0.0,
        "size_down_std": _safe(down.std()) if down.size else 0.0,
        # ---- timing distribution ----
        "iat_entropy": _entropy(np.log10(np.maximum(iat, _IAT_FLOOR)), bins=32, lo=-6, hi=2),
        "iat_cv": _safe(iat.std() / (iat.mean() + _EPS)) if iat.size else 0.0,
        "iat_autocorr_peak": _autocorr_peak(iat),
        "iat_up_cv": _safe(iat_up.std() / (iat_up.mean() + _EPS)) if iat_up.size else 0.0,
        "iat_up_autocorr_peak": _autocorr_peak(iat_up),
    }

    feats.update(_moments(size, "size"))
    feats.update(_moments(np.log10(np.maximum(iat, _IAT_FLOOR)), "logiat"))
    feats.update(_burst_stats(t, idle=0.5))

    return feats


#: Canonical feature order. Fixed so that a saved scaler, a saved model and a
#: feature-importance table all refer to the same column.
FEATURE_NAMES: list[str] = list(flow_statistics(np.array([[0.0, 100.0, 1.0], [0.01, 200.0, -1.0]])))
N_FEATURES = len(FEATURE_NAMES)


def stats_vector(pkts: np.ndarray) -> np.ndarray:
    """Flow statistics as a vector in :data:`FEATURE_NAMES` order."""
    feats = flow_statistics(pkts)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float32)


def extract(pkts: np.ndarray, length: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Both views of one flow: ``(sequence, statistics)``."""
    return sequence_tensor(pkts, length), stats_vector(pkts)
