# Adversarial Network Traffic Obfuscation Detection (ANTOD)

Deep Learning Mini Project (CS722T2C) — V Semester, B.E. Computer Science & Engineering
Sahyadri College of Engineering & Management, Mangaluru (VTU, Belagavi)

## Problem

Modern malware does not hide *what* it sends so much as *how* it sends it. By padding packets,
jittering inter-arrival times, fragmenting records, tunnelling over permitted ports and mimicking
the statistical shape of benign TLS or HTTP, an attacker can push malicious traffic past
signature-based Deep Packet Inspection (DPI) without changing the payload semantics at all.

ANTOD attacks this from the defensive side. It learns the *shape* of a flow — the sequence of
packet sizes and inter-arrival times, plus aggregate flow statistics — and classifies it as:

| Class | Meaning |
| --- | --- |
| `benign` | Ordinary application traffic |
| `malicious_plain` | Malicious traffic making no attempt to hide |
| `malicious_obfuscated` | Malicious traffic transformed to evade DPI |

The third class is the interesting one: it is precisely the traffic a signature engine misses.

Because a detector that only works on *unaware* attackers is of little use, ANTOD is then
stress-tested with adversarial machine learning. We craft feature-space FGSM and PGD perturbations
under domain-validity constraints (a packet cannot have negative size, time cannot run backwards),
measure how far accuracy falls, and harden the model with adversarial training.

## Repository layout

```
src/antod/
  data/          traffic simulator, obfuscation transforms, feature extraction, datasets
  models/        1D-CNN, MLP, hybrid model, classical baselines
  adversarial/   FGSM / PGD attacks, adversarial training defenses
  utils/         seeding, metrics, plotting, logging
configs/         YAML experiment configurations
experiments/     committed metrics, tables and figures
docs/            project report and book chapter
tests/           unit tests for the data and model pipeline
scripts/         end-to-end reproduction scripts
```

## Quick start

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
antod generate --config configs/dataset.yaml     # build the dataset
antod train    --config configs/cnn1d.yaml       # train the 1D-CNN
antod attack   --config configs/attack.yaml      # adversarial robustness sweep
antod evaluate --config configs/evaluate.yaml    # tables + figures
```

Or reproduce everything in one shot:

```bash
python scripts/run_all.py
```

## Data

The default dataset is produced by a packet-level traffic simulator (`antod.data.synth`) whose
benign and malicious flow distributions are parameterised from published measurements, with
obfuscation applied by explicit transforms (`antod.data.obfuscation`). This makes labels exact —
we know which flows were obfuscated and how — and makes the whole project reproducible from a
single seed without redistributing capture data.

Real captures are supported too: drop CIC-IDS2017 or UNSW-NB15 CSVs into `data/raw/` and point
`configs/dataset.yaml` at them (`source: real`). See `docs/DATA.md`.

## Team

| Name | USN |
| --- | --- |
| Vivek Neeralagi (Team Lead) | 4SF23CS246 |
| Muhammad Shahbik | 4SF23CS113 |
| Aravind P Sagar | 4SF23CS030 |
| J Aditya | 4SF24CS408 |

Guide: Mrs. Varsha M, Assistant Professor, Department of CSE

## Scope note

The obfuscation transforms in this repository operate on *extracted flow features and synthetic
packet records*, not on live sockets. They exist to generate labelled training data for a detector
and to measure its robustness. This is a defensive research tool; it is not an evasion utility.

## License

MIT — see [LICENSE](LICENSE).
