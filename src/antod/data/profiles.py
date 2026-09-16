"""Packet-level generators for benign and malicious application profiles.

Every generator returns an ``(n, 3)`` float array of packet records:

===== ==========================================================================
col 0 timestamp in seconds, relative to the first packet of the flow
col 1 packet size in bytes, including headers
col 2 direction, ``+1`` for client to server (up), ``-1`` for server to client
===== ==========================================================================

Parameters reproduce the *qualitative* signatures reported in the traffic
classification literature: heavy-tailed think times with MTU-filling downstream
bursts for web browsing, fixed-cadence small packets for VoIP, low-jitter
periodic beacons for command-and-control, and so on. They are not a claim of
packet-for-packet fidelity to any particular capture. The point is that the
classes are separable by flow *shape* rather than by payload bytes, which is
exactly the regime a signature-based DPI engine cannot reach.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

MTU = 1500
MIN_PKT = 40
ACK_LO, ACK_HI = 40, 72

UP = 1.0
DOWN = -1.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pkts(times: list[float], sizes: list[float], dirs: list[float]) -> np.ndarray:
    """Assemble, time-sort and normalise a packet list into an (n, 3) array."""
    t = np.asarray(times, dtype=np.float64)
    s = np.clip(np.rint(np.asarray(sizes, dtype=np.float64)), MIN_PKT, MTU)
    d = np.asarray(dirs, dtype=np.float64)

    order = np.argsort(t, kind="stable")
    t, s, d = t[order], s[order], d[order]
    t = t - t[0]
    return np.stack([t, s, d], axis=1)


def _think_time(rng: np.random.Generator, scale: float = 0.6, alpha: float = 1.7) -> float:
    """Heavy-tailed idle gap. Pareto tails are the standard model for think time."""
    return float(scale * (rng.pareto(alpha) + 1.0))


def _bulk_transfer(
    rng: np.random.Generator,
    t0: float,
    nbytes: float,
    rate_bps: float,
    direction: float = DOWN,
    ack_every: int = 2,
) -> tuple[list[float], list[float], list[float], float]:
    """Emit a bulk transfer of ``nbytes`` in ``direction`` with ACKs coming back.

    Packets fill the MTU until the final remainder. Inter-packet gaps follow the
    serialisation delay at ``rate_bps`` with multiplicative jitter, which produces
    the tight-burst / long-gap texture that distinguishes real transfers from a
    uniform packet train.
    """
    times: list[float] = []
    sizes: list[float] = []
    dirs: list[float] = []

    remaining = max(int(nbytes), MIN_PKT)
    t = t0
    emitted = 0
    while remaining > 0:
        payload = min(remaining, MTU)
        remaining -= payload
        t += ((payload * 8) / rate_bps) * rng.uniform(0.5, 1.8)
        times.append(t)
        sizes.append(float(payload))
        dirs.append(direction)
        emitted += 1
        if emitted % ack_every == 0:
            times.append(t + rng.uniform(1e-4, 3e-3))
            sizes.append(rng.uniform(ACK_LO, ACK_HI))
            dirs.append(-direction)
    return times, sizes, dirs, t


# --------------------------------------------------------------------------- #
# benign profiles
# --------------------------------------------------------------------------- #
def web_browsing(rng: np.random.Generator) -> np.ndarray:
    """Page loads: small requests up, bursty MTU-filling responses down, long think times."""
    times: list[float] = []
    sizes: list[float] = []
    dirs: list[float] = []
    rate = rng.uniform(2e6, 4e7)
    t = 0.0

    for _ in range(int(rng.integers(3, 15))):
        for _ in range(int(rng.integers(1, 3))):  # request, occasionally split
            times.append(t)
            sizes.append(rng.normal(480, 130))
            dirs.append(UP)
            t += rng.uniform(2e-4, 2e-3)
        t += rng.uniform(0.01, 0.15)  # RTT plus server processing
        bt, bs, bd, t = _bulk_transfer(rng, t, rng.lognormal(9.2, 1.3), rate)
        times += bt
        sizes += bs
        dirs += bd
        t += _think_time(rng, 0.5, 1.7)

    return _pkts(times, sizes, dirs)


def video_streaming(rng: np.random.Generator) -> np.ndarray:
    """Adaptive-bitrate playback: periodic chunk requests, very regular large bursts."""
    times: list[float] = []
    sizes: list[float] = []
    dirs: list[float] = []
    rate = rng.uniform(8e6, 8e7)
    period = rng.uniform(2.0, 5.0)
    chunk = rng.uniform(1.5e5, 4.0e5)
    t = 0.0

    for _ in range(int(rng.integers(3, 8))):
        times.append(t)
        sizes.append(rng.normal(620, 80))
        dirs.append(UP)
        t += rng.uniform(0.01, 0.06)
        bt, bs, bd, t_end = _bulk_transfer(rng, t, chunk * rng.uniform(0.85, 1.15), rate)
        times += bt
        sizes += bs
        dirs += bd
        # the player then sleeps to the next chunk boundary: near-constant cadence
        t = max(t_end, t + period * rng.uniform(0.97, 1.03))

    return _pkts(times, sizes, dirs)


def voip(rng: np.random.Generator) -> np.ndarray:
    """RTP-style call: constant small packets both ways at a fixed codec interval."""
    times: list[float] = []
    sizes: list[float] = []
    dirs: list[float] = []
    interval = float(rng.choice([0.02, 0.03]))
    duration = rng.uniform(3.0, 15.0)
    base_up = rng.uniform(160, 220)
    base_down = rng.uniform(160, 220)

    t = 0.0
    while t < duration:
        times.append(t + rng.normal(0, 8e-4))
        sizes.append(rng.normal(base_up, 6))
        dirs.append(UP)
        times.append(t + interval / 2 + rng.normal(0, 8e-4))
        sizes.append(rng.normal(base_down, 6))
        dirs.append(DOWN)
        t += interval

    return _pkts(times, sizes, dirs)


def file_download(rng: np.random.Generator) -> np.ndarray:
    """One sustained bulk download behind a short request prologue."""
    times = [0.0]
    sizes = [rng.normal(520, 100)]
    dirs = [UP]
    bt, bs, bd, _ = _bulk_transfer(
        rng,
        rng.uniform(0.01, 0.08),
        rng.uniform(2.0e5, 1.5e6),
        rng.uniform(1e7, 1.2e8),
        ack_every=3,
    )
    return _pkts(times + bt, sizes + bs, dirs + bd)


def dns_query(rng: np.random.Generator) -> np.ndarray:
    """A handful of tiny request/response exchanges."""
    times: list[float] = []
    sizes: list[float] = []
    dirs: list[float] = []
    t = 0.0
    for _ in range(int(rng.integers(1, 4))):
        times.append(t)
        sizes.append(rng.uniform(70, 110))
        dirs.append(UP)
        t += rng.uniform(0.005, 0.09)
        times.append(t)
        sizes.append(rng.uniform(90, 300))
        dirs.append(DOWN)
        t += rng.uniform(0.05, 1.5)
    return _pkts(times, sizes, dirs)


def ssh_interactive(rng: np.random.Generator) -> np.ndarray:
    """A person at a terminal: tiny keystroke-paced packets with echo coming back."""
    times: list[float] = []
    sizes: list[float] = []
    dirs: list[float] = []
    t = 0.0

    for _ in range(int(rng.integers(4, 12))):  # commands
        for _ in range(int(rng.integers(3, 25))):  # keystrokes
            times.append(t)
            sizes.append(rng.normal(92, 12))
            dirs.append(UP)
            t += rng.uniform(1e-3, 6e-3)
            times.append(t)
            sizes.append(rng.normal(92, 14))
            dirs.append(DOWN)
            t += abs(rng.lognormal(-1.9, 0.7))  # typing gap, ~150 ms median
        bt, bs, bd, t = _bulk_transfer(  # command output
            rng, t + rng.uniform(0.02, 0.2), rng.lognormal(7.5, 1.2), rng.uniform(5e6, 5e7)
        )
        times += bt
        sizes += bs
        dirs += bd
        t += _think_time(rng, 0.9, 1.5)

    return _pkts(times, sizes, dirs)


# --------------------------------------------------------------------------- #
# malicious profiles
# --------------------------------------------------------------------------- #
def c2_beacon(rng: np.random.Generator) -> np.ndarray:
    """Command-and-control check-in: near-constant period, near-constant request size.

    The give-away is not the content but the metronome: low jitter on the
    inter-arrival times and very low variance on the request size.
    """
    times: list[float] = []
    sizes: list[float] = []
    dirs: list[float] = []
    period = rng.uniform(5.0, 60.0)
    jitter = rng.uniform(0.0, 0.08)  # as a fraction of the period
    req = rng.uniform(220, 400)
    t = 0.0

    for _ in range(int(rng.integers(4, 20))):
        times.append(t)
        sizes.append(rng.normal(req, 10))
        dirs.append(UP)
        t += rng.uniform(0.02, 0.25)
        for _ in range(int(rng.integers(1, 4))):  # tasking payload back
            times.append(t)
            sizes.append(rng.normal(rng.uniform(300, 900), 40))
            dirs.append(DOWN)
            t += rng.uniform(1e-3, 0.02)
        t += period * (1.0 + rng.uniform(-jitter, jitter))

    return _pkts(times, sizes, dirs)


def data_exfiltration(rng: np.random.Generator) -> np.ndarray:
    """Sustained *upstream* bulk transfer: a download run backwards."""
    times = [0.0]
    sizes = [rng.normal(300, 60)]
    dirs = [UP]
    bt, bs, bd, _ = _bulk_transfer(
        rng,
        rng.uniform(0.01, 0.1),
        rng.uniform(1.0e5, 2.0e6),
        rng.uniform(2e6, 3e7),
        direction=UP,
        ack_every=3,
    )
    return _pkts(times + bt, sizes + bs, dirs + bd)


def port_scan(rng: np.random.Generator) -> np.ndarray:
    """SYN sweep: a wall of minimum-size packets with almost no inter-arrival time."""
    times: list[float] = []
    sizes: list[float] = []
    dirs: list[float] = []
    t = 0.0
    for _ in range(int(rng.integers(40, 200))):
        times.append(t)
        sizes.append(rng.uniform(54, 74))
        dirs.append(UP)
        t += abs(rng.normal(1.5e-3, 1.0e-3)) + 1e-5
        if rng.random() < 0.25:  # RST or SYN-ACK from the few live ports
            times.append(t)
            sizes.append(rng.uniform(54, 66))
            dirs.append(DOWN)
    return _pkts(times, sizes, dirs)


def brute_force(rng: np.random.Generator) -> np.ndarray:
    """Credential stuffing: one short exchange repeated with machine regularity."""
    times: list[float] = []
    sizes: list[float] = []
    dirs: list[float] = []
    req = rng.uniform(380, 700)
    resp = rng.uniform(200, 500)
    gap = rng.uniform(0.15, 1.5)
    t = 0.0

    for _ in range(int(rng.integers(10, 60))):
        times.append(t)
        sizes.append(rng.normal(req, 8))
        dirs.append(UP)
        t += rng.uniform(0.01, 0.08)
        times.append(t)
        sizes.append(rng.normal(resp, 8))
        dirs.append(DOWN)
        t += gap * rng.uniform(0.95, 1.05)

    return _pkts(times, sizes, dirs)


def ddos_flood(rng: np.random.Generator) -> np.ndarray:
    """Volumetric flood: uniform small packets at line rate, effectively one-way."""
    n = int(rng.integers(300, 900))
    base = rng.uniform(60, 140)
    t = np.cumsum(np.abs(rng.normal(4e-5, 2e-5, n)) + 1e-6)
    s = rng.normal(base, 4, n)
    d = np.where(rng.random(n) < 0.03, DOWN, UP)
    return _pkts(list(t), list(s), list(d))


def reverse_shell(rng: np.random.Generator) -> np.ndarray:
    """An attacker at a terminal. Deliberately close to ``ssh_interactive``.

    Keeping a malicious profile that overlaps a benign one stops the task from
    being solvable by a threshold rule, which is where a deep model earns its keep.
    """
    times: list[float] = []
    sizes: list[float] = []
    dirs: list[float] = []
    t = 0.0

    for _ in range(int(rng.integers(3, 10))):
        # the operator pastes a command rather than typing it: fewer, larger up packets
        times.append(t)
        sizes.append(rng.normal(rng.uniform(120, 420), 25))
        dirs.append(UP)
        t += rng.uniform(0.01, 0.06)
        bt, bs, bd, t = _bulk_transfer(rng, t, rng.lognormal(7.0, 1.4), rng.uniform(1e6, 2e7))
        times += bt
        sizes += bs
        dirs += bd
        t += abs(rng.lognormal(0.6, 0.9))  # operator reading the output

    return _pkts(times, sizes, dirs)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
Generator = Callable[[np.random.Generator], np.ndarray]

BENIGN_PROFILES: dict[str, Generator] = {
    "web_browsing": web_browsing,
    "video_streaming": video_streaming,
    "voip": voip,
    "file_download": file_download,
    "dns_query": dns_query,
    "ssh_interactive": ssh_interactive,
}

MALICIOUS_PROFILES: dict[str, Generator] = {
    "c2_beacon": c2_beacon,
    "data_exfiltration": data_exfiltration,
    "port_scan": port_scan,
    "brute_force": brute_force,
    "ddos_flood": ddos_flood,
    "reverse_shell": reverse_shell,
}

ALL_PROFILES: dict[str, Generator] = {**BENIGN_PROFILES, **MALICIOUS_PROFILES}
