"""Ingest real captures, so the pipeline is not tied to synthetic traffic.

The unit of input here is a **per-packet CSV**: one row per packet, carrying a
flow identifier, a timestamp, a length and enough addressing to work out
direction. That is exactly what ``tshark`` exports from a PCAP, and it is the
only real-data format that preserves what this project actually models -- the
sequence of packet sizes and inter-arrival times.

A note on the public flow-level datasets, because it is a reasonable question to
ask. CIC-IDS2017, UNSW-NB15 and CSE-CIC-IDS2018 ship as *flow-level* CSVs:
already-aggregated statistics such as mean packet length and flow IAT standard
deviation. Those aggregates cannot reconstruct the per-packet sequence the
1D-CNN consumes, and zero-filling the difference would quietly inject a
distribution shift into the middle of the experiment. The right way to use those
corpora with this code is therefore to start from the PCAPs they are derived from
and export per-packet rows -- see ``docs/DATA.md`` for the exact command. Their
label taxonomy is mapped in :data:`ATTACK_LABELS` so that step is mechanical.

Once real flows are loaded, :func:`build_obfuscated_classes` applies the same
obfuscation recipes used on synthetic traffic, which yields the three-class
problem from real benign and real attack captures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from antod.data.obfuscation import sample_recipe
from antod.data.profiles import DOWN, UP, _pkts
from antod.data.synth import (
    BENIGN,
    MALICIOUS_OBFUSCATED,
    MALICIOUS_PLAIN,
    Flow,
    SynthConfig,
)

#: Attack names used across the CIC/UNSW families, mapped to our binary intent.
#: Anything not listed is treated as benign, which is the conservative direction:
#: an unmapped attack becomes a false negative in training rather than poisoning
#: the benign class with an unknown label.
ATTACK_LABELS: dict[str, int] = {
    "benign": BENIGN,
    "normal": BENIGN,
    "bot": MALICIOUS_PLAIN,
    "ddos": MALICIOUS_PLAIN,
    "dos": MALICIOUS_PLAIN,
    "dos goldeneye": MALICIOUS_PLAIN,
    "dos hulk": MALICIOUS_PLAIN,
    "dos slowhttptest": MALICIOUS_PLAIN,
    "dos slowloris": MALICIOUS_PLAIN,
    "ftp-patator": MALICIOUS_PLAIN,
    "ssh-patator": MALICIOUS_PLAIN,
    "heartbleed": MALICIOUS_PLAIN,
    "infiltration": MALICIOUS_PLAIN,
    "portscan": MALICIOUS_PLAIN,
    "web attack": MALICIOUS_PLAIN,
    "web attack - brute force": MALICIOUS_PLAIN,
    "web attack - sql injection": MALICIOUS_PLAIN,
    "web attack - xss": MALICIOUS_PLAIN,
    "backdoor": MALICIOUS_PLAIN,
    "analysis": MALICIOUS_PLAIN,
    "exploits": MALICIOUS_PLAIN,
    "fuzzers": MALICIOUS_PLAIN,
    "generic": MALICIOUS_PLAIN,
    "reconnaissance": MALICIOUS_PLAIN,
    "shellcode": MALICIOUS_PLAIN,
    "worms": MALICIOUS_PLAIN,
}

#: Column aliases accepted for each required field, lower-cased and stripped.
_ALIASES: dict[str, tuple[str, ...]] = {
    "flow_id": ("flow_id", "tcp.stream", "udp.stream", "stream", "flow", "flowid"),
    "timestamp": ("timestamp", "frame.time_relative", "time", "ts", "frame.time_epoch"),
    "size": ("size", "frame.len", "length", "len", "bytes", "ip.len"),
    "direction": ("direction", "dir"),
    "src": ("src", "ip.src", "source", "src_ip"),
    "dst": ("dst", "ip.dst", "destination", "dst_ip"),
    "label": ("label", "class", "attack", "attack_cat", "category"),
}


def _resolve(columns: list[str], field: str) -> str | None:
    lut = {c.strip().lower(): c for c in columns}
    for alias in _ALIASES[field]:
        if alias in lut:
            return lut[alias]
    return None


def _label_of(raw: object) -> int:
    """Map a capture label to one of our classes."""
    if isinstance(raw, (int, np.integer)):
        return int(raw) if int(raw) in (BENIGN, MALICIOUS_PLAIN, MALICIOUS_OBFUSCATED) else BENIGN
    key = str(raw).strip().lower()
    return ATTACK_LABELS.get(key, BENIGN)


def load_packet_csv(
    path: str | Path,
    min_packets: int = 8,
    max_flows: int | None = None,
) -> list[Flow]:
    """Read a per-packet CSV and group it into :class:`Flow` objects.

    Required columns (any listed alias is accepted): a flow id, a timestamp and a
    packet length. Direction comes from an explicit ``direction`` column if
    present; otherwise it is inferred by treating the source address of each
    flow's first packet as the client, which is the standard convention.
    """
    df = pd.read_csv(path)
    cols = list(df.columns)

    resolved = {f: _resolve(cols, f) for f in _ALIASES}
    for required in ("flow_id", "timestamp", "size"):
        if resolved[required] is None:
            raise ValueError(
                f"per-packet CSV needs a {required!r} column "
                f"(any of {_ALIASES[required]}); got {cols}"
            )

    df = df.rename(columns={v: k for k, v in resolved.items() if v is not None and v != k})
    df = df.dropna(subset=["flow_id", "timestamp", "size"])
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df = df.dropna(subset=["timestamp", "size"]).sort_values(["flow_id", "timestamp"])

    has_dir = "direction" in df.columns
    has_addr = "src" in df.columns and "dst" in df.columns
    if not has_dir and not has_addr:
        raise ValueError(
            "cannot determine packet direction: supply a 'direction' column, "
            "or 'src'/'dst' addresses so the client side can be inferred"
        )

    flows: list[Flow] = []
    for _, group in df.groupby("flow_id", sort=False):
        if len(group) < min_packets:
            continue

        if has_dir:
            d = np.where(pd.to_numeric(group["direction"], errors="coerce").fillna(1) > 0, UP, DOWN)
        else:
            client = group["src"].iloc[0]
            d = np.where(group["src"].to_numpy() == client, UP, DOWN)

        pkts = _pkts(
            list(group["timestamp"].to_numpy(dtype=np.float64)),
            list(group["size"].to_numpy(dtype=np.float64)),
            list(d.astype(np.float64)),
        )

        label = _label_of(group["label"].iloc[0]) if "label" in group.columns else BENIGN
        flows.append(
            Flow(packets=pkts, label=label, profile="capture", recipe="none", obfuscated=False)
        )
        if max_flows is not None and len(flows) >= max_flows:
            break

    if not flows:
        raise ValueError(f"no flow in {path} reached min_packets={min_packets}")
    return flows


def build_obfuscated_classes(
    flows: list[Flow],
    cfg: SynthConfig | None = None,
    seed: int = 17,
) -> list[Flow]:
    """Turn labelled real flows into the three-class problem.

    Malicious flows are split: a share stays plain (class 1) and the rest is
    obfuscated (class 2). Benign flows are left as class 0, with
    ``cfg.benign_obfuscation_rate`` of them obfuscated anyway, for the same reason
    the synthetic generator does it -- so the model cannot equate obfuscation with
    hostility.
    """
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(seed)

    # among malicious flows, the share that should end up obfuscated
    w_plain, w_obf = cfg.class_weights[1], cfg.class_weights[2]
    obf_share = w_obf / max(w_plain + w_obf, 1e-9)

    out: list[Flow] = []
    for flow in flows:
        malicious = flow.label != BENIGN
        obfuscate = (
            rng.random() < obf_share if malicious else rng.random() < cfg.benign_obfuscation_rate
        )

        if not obfuscate:
            out.append(
                Flow(
                    packets=flow.packets,
                    label=MALICIOUS_PLAIN if malicious else BENIGN,
                    profile=flow.profile,
                    recipe="none",
                    obfuscated=False,
                )
            )
            continue

        recipe = sample_recipe(rng, cfg.min_steps, cfg.max_steps, cfg.allowed_transforms)
        out.append(
            Flow(
                packets=recipe.apply(flow.packets, rng),
                label=MALICIOUS_OBFUSCATED if malicious else BENIGN,
                profile=flow.profile,
                recipe=recipe.describe(),
                obfuscated=True,
            )
        )

    rng.shuffle(out)
    return out


def load_real_dataset(
    path: str | Path,
    cfg: SynthConfig | None = None,
    min_packets: int = 8,
    max_flows: int | None = None,
    seed: int = 17,
) -> list[Flow]:
    """Full real-data path: per-packet CSV in, three labelled classes out."""
    flows = load_packet_csv(path, min_packets=min_packets, max_flows=max_flows)
    return build_obfuscated_classes(flows, cfg=cfg, seed=seed)
