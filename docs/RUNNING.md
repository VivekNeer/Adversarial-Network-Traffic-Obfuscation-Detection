# Running guide

Tested on Windows 11 with CPU-only PyTorch. Works the same on Linux/macOS.

## 1. Setup

Requires Python 3.10+. The easiest route is [`uv`](https://docs.astral.sh/uv/):

```bash
uv python install 3.11
uv venv --python 3.11
uv pip install -e ".[dev]"
```

Or with plain pip:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows without activating the venv, prefix commands with
`.venv\Scripts\python.exe -m` (e.g. `.venv\Scripts\python.exe -m antod.cli ...`).
The examples below assume the venv is active.

To reproduce the reported numbers exactly, install the pinned versions instead:

```bash
uv pip install -r requirements-lock.txt -e ".[dev]"
```

Check everything works:

```bash
pytest
```

## 2. Quick smoke run (≈3 min, CPU)

```bash
python scripts/run_all.py --quick
```

Generates 2 000 flows, trains all four models for 4 epochs, attacks and
evaluates them. Output lands in `experiments/runs/quick_*/`.

## 3. Full run (≈45–90 min on CPU)

```bash
python scripts/run_all.py
```

Or step by step:

```bash
antod generate -c configs/dataset.yaml        # 20k flows -> data/processed/dataset.npz
antod train    -c configs/cnn1d.yaml          # train + baselines + figures + checkpoint
antod attack   -c configs/cnn1d.yaml          # FGSM/PGD sweep, robustness curves, smoothing
antod evaluate -c configs/cnn1d.yaml          # per-technique recall, per-profile, transfer
```

Repeat `train`/`attack`/`evaluate` for `mlp.yaml`, `hybrid.yaml`, `gru.yaml`,
`hybrid_advtrain.yaml`. `antod all -c <config>` chains the last three.

Seed variance (mean ± std over three seeds, needed before claiming one model
beats another):

```bash
python scripts/seed_variance.py configs/hybrid.yaml --seeds 1 2 3
```

The transfer matrix in `evaluate` only appears once at least two checkpoints
exist under `experiments/results/`. It runs on a stratified 1,000-flow subsample
of the test split (`output.transfer_flows`); attack sweeps use every test flow
unless `output.attack_flows` is set (the GRU config sets 500, because PGD
through the recurrence costs minutes per run on CPU). Clean metrics always use
the whole test split.

## 4. Where the outputs go

Each config writes to its `output.dir` (default `experiments/results/<name>/`):

```
metrics.json            clean test metrics, val metrics, baselines, feature importances,
                        calibration (temperature, ECE), precision at 90/99/99.9% benign
predictions.csv         one row per test flow: profile, recipe, true, predicted, probabilities
attack_results.json     attack sweep rows, robustness/evasion curves, smoothing sweep,
                        packet-space black-box attack summary
evaluation.json         per-technique recall, per-profile accuracy, transfer matrix
checkpoint.pt           weights + fitted scaler + config
config.yaml             the resolved config that produced these results
figures/                confusion_matrix, training_curves, model_comparison,
                        feature_importance, robustness_constrained, evasion_rate,
                        per_technique_recall, per_profile_accuracy  (.png + .csv)
tables/                 model_comparison, attack_sweep, smoothing, packet_attack,
                        per_technique_recall, per_profile_accuracy, transfer_matrix  (.md)
```

The `.md` tables paste straight into the report; the `.csv` beside each figure
holds the exact numbers plotted.

## 5. Command-line overrides

Any config can be tweaked without editing it:

```bash
antod train -c configs/cnn1d.yaml --epochs 10 --n-flows 5000 --out experiments/runs/try1
antod train -c configs/cnn1d.yaml --model cnn1d_small --seed 7
antod train -c configs/hybrid.yaml --device cpu
```

Flags: `--name`, `--out`, `--epochs`, `--model`, `--n-flows`, `--seed`, `--device`.

Available models: `cnn1d`, `cnn1d_small`, `mlp`, `mlp_wide`, `hybrid`, `gru`.
Available baselines: `random_forest`, `rbf_svm`, `logistic_regression`.

## 6. Using your own captures

Export per-packet rows from a PCAP:

```bash
tshark -r capture.pcap -T fields -E separator=, -E header=y -e tcp.stream -e frame.time_relative -e frame.len -e ip.src -e ip.dst > data/raw/packets.csv
```

Then in a config:

```yaml
dataset:
  source: real
  path: data/raw/packets.csv
  max_flows: 20000
```

Add a `label` column (`benign`, `portscan`, `dos hulk`, …) if you have one.
Details in [DATA.md](DATA.md).

To score a capture with an already-trained model (no labels needed):

```bash
antod predict -c configs/hybrid.yaml --input data/raw/packets.csv --predictions-out scored.csv
```

For flows longer than 128 packets, `antod.inference.score_flow_windows` slides
the window across the whole flow and combines the per-window verdicts.

## 7. Using the code directly

```python
from antod.data.synth import SynthConfig
from antod.data.datasets import build_dataset, stratified_split
from antod.train import TrainConfig, Trainer
from antod.evaluate import evaluate_clean, per_technique_recall
from antod.adversarial import AttackConfig, run_attack
import torch

splits = stratified_split(build_dataset(SynthConfig(n_flows=3000, seed=1)))
result = Trainer(TrainConfig(model="hybrid", epochs=15)).fit(splits)

dev = torch.device("cpu")
print(evaluate_clean(result.model, splits.test, splits.scaler, dev).summary())
print(per_technique_recall(result.model, splits.scaler, dev, n_per_technique=200))
```

## 8. Development

```bash
ruff check src tests      # lint
ruff format src tests     # format
pytest -q                 # tests (~1 min)
```

## 9. Troubleshooting

- **`Python was not found`** on Windows — that is the Microsoft Store stub.
  Use `uv python install 3.11` or install Python from python.org.
- **`feature set has changed since this dataset was written`** — regenerate
  with `antod generate`; `FEATURE_NAMES` changed.
- **`no checkpoint at ...`** — run `antod train` for that config first.
- **Slow SVM** — drop `rbf_svm` from `baselines:` in the config; it is the only
  baseline that scales badly with dataset size.
- **CUDA** — `device: auto` picks a GPU if `torch.cuda.is_available()`; the
  default install is CPU-only PyTorch. Install a CUDA build of torch to use one.
