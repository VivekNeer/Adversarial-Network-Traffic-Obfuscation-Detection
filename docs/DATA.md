# Data

ANTOD can be driven from synthetic traffic (the default) or from real captures.

## 1. Synthetic traffic (default)

`antod.data.synth` draws an application profile, optionally applies an
obfuscation recipe, and emits a labelled flow. Twelve profiles are available:

| Benign | Malicious |
| --- | --- |
| `web_browsing` | `c2_beacon` |
| `video_streaming` | `data_exfiltration` |
| `voip` | `port_scan` |
| `file_download` | `brute_force` |
| `dns_query` | `ddos_flood` |
| `ssh_interactive` | `reverse_shell` |

`reverse_shell` and `ssh_interactive` overlap on purpose. Without a pair of
classes that a size/rate threshold cannot separate, a mini project can report
99% accuracy while demonstrating nothing.

Everything is seeded: `SynthConfig(seed=42)` reproduces a dataset exactly,
including which recipe landed on which flow.

### Why synthetic is the default, and what it costs

Being explicit about this, since it is the main limitation of the study.

The advantage is that **obfuscation is applied rather than inferred**. In any
real corpus, nobody has annotated which flows were padded, tunnelled or shaped,
so the `malicious_obfuscated` class would have to be guessed — and a model
evaluated against guessed labels measures agreement with the guess. Here the
recipe is recorded per flow, which is what makes the per-technique breakdown in
§5 of the report possible at all.

The cost is external validity. Parameters come from published measurement
studies, not from a capture of any particular network, so absolute accuracy on
this dataset is an upper bound on what the same model would achieve on live
traffic. The honest reading of the headline numbers is *relative*: which
architecture wins, which evasion technique hurts most, and how much adversarial
training buys back.

## 2. Real captures

The loader consumes a **per-packet CSV** — one row per packet. This is the only
format that preserves what the 1D-CNN actually reads: the ordered sequence of
packet sizes and inter-arrival times.

### Exporting from a PCAP

```bash
tshark -r capture.pcap -T fields -E separator=, -E header=y -e tcp.stream -e frame.time_relative -e frame.len -e ip.src -e ip.dst > packets.csv
```

Recognised column names (any alias works):

| Field | Aliases |
| --- | --- |
| flow id | `flow_id`, `tcp.stream`, `udp.stream`, `stream`, `flow` |
| timestamp | `timestamp`, `frame.time_relative`, `frame.time_epoch`, `time`, `ts` |
| size | `size`, `frame.len`, `length`, `len`, `bytes`, `ip.len` |
| direction | `direction`, `dir` — or inferred from `src`/`dst` |
| label | `label`, `class`, `attack`, `attack_cat`, `category` |

Direction is taken from an explicit column when present; otherwise the source
address of each flow's first packet is treated as the client.

Then point the config at it:

```yaml
dataset:
  source: real
  path: data/raw/packets.csv
  max_flows: 20000
```

`build_obfuscated_classes` applies the same obfuscation recipes to the real
flows, so the three-class problem is constructed identically whether the base
traffic is synthetic or captured.

### On CIC-IDS2017, UNSW-NB15 and CSE-CIC-IDS2018

These ship as **flow-level** CSVs: pre-aggregated statistics such as mean packet
length and flow IAT standard deviation. They cannot reconstruct the per-packet
sequence, and zero-filling the gap would inject a distribution shift into the
middle of the experiment — the model would learn to separate "rows with real
sequence data" from "rows without".

So the supported route is to start from the PCAPs those corpora are derived from
and export per-packet rows with the `tshark` command above. Their attack
taxonomy is already mapped in `antod.data.real_loader.ATTACK_LABELS`
(`DoS Hulk`, `PortScan`, `FTP-Patator`, `Infiltration`, `Exploits`, `Worms`, …),
so labelling is mechanical once the packets are exported. Unmapped labels fall
back to benign, which costs recall rather than contaminating the benign class
with an unknown attack.

## 3. Feature views

Both paths produce the same two views per flow.

**Sequence view** — `(4, 128)`, for the 1D-CNN:

| Channel | Contents |
| --- | --- |
| 0 | signed packet size, `size/MTU` carrying the direction sign |
| 1 | `log10` inter-arrival time, scaled |
| 2 | direction, `+1` / `-1` |
| 3 | validity mask, `1` real / `0` padding |

**Statistics view** — 51 features, for the MLP and the classical baselines.
Volume, size and timing moments, plus three groups aimed at obfuscation:
`size_grid_{64,128,256}` (block-padding fingerprint), `unique_size_ratio` and
`mode_size_frac` (size-diversity collapse), `iat_cv` and `iat_autocorr_peak`
(timing regularity — how a C2 metronome is caught, and how constant-rate shaping
hides it).

`size_grid_*` excludes packets sitting at the MTU. A padder cannot grow a
1300-byte record to the next 256-byte boundary without exceeding the MTU, so such
a record gets clamped to 1500, which is not a multiple of 64, 128 or 256.
Counting clamped packets would dilute the fingerprint precisely in the
bulk-transfer flows where padding matters most.

## 4. Directory layout

```
data/raw/         captures and exported per-packet CSVs (not versioned)
data/processed/   generated .npz datasets (not versioned)
data/interim/     scratch (not versioned)
```

Datasets are regenerable from a seed, so they are deliberately kept out of git.
