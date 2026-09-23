# Running guide

Tested on Windows 11 with CPU-only PyTorch. Works the same on Linux/macOS.

## 1. Setup & Environment

Requires Python 3.10+. The project environment is already configured in `.venv/`.

### Windows PowerShell Quick Start

ANTOD and its dependencies (PyTorch, scikit-learn, etc.) are installed in the local virtual environment. In PowerShell:

```powershell
# Activate the virtual environment
& .\.venv\Scripts\Activate.ps1

# Verify the antod CLI is active
antod --help
```

> **Tip:** If you prefer running without activating the virtual environment in your shell, you can directly invoke the executable:
> ```powershell
> .\.venv\Scripts\antod.exe --help
> ```
> Global `python` has also been mapped to Python 3.11 in `~/.local/bin/python.exe`, avoiding the Windows Store execution stub (`Python was not found`).

---

## 2. Interactive Web Dashboard (GUI)

An interactive, high-aesthetic web dashboard is included to explore pre-computed results, inspect test flows, and run live inference without re-running experiments.

Launch the GUI dashboard:

```powershell
# Option 1: Via the antod CLI
antod gui

# Option 2: Directly via python script
python scripts/gui.py
```

The server launches at **`http://localhost:8000`** and opens automatically in your browser.

### Key Dashboard Tabs:
1. **Overview & KPIs**: Test accuracy (`96.88%`), obfuscated recall (`94.24%`), malicious recall (`98.60%`), base-rate precision, and threat class cards.
2. **Model Leaderboard**: Side-by-side comparison across all 5 models (`Hybrid`, `CNN-1D`, `MLP`, `GRU`, `Hybrid+AdvTrain`) and classical baselines (`Random Forest`, `RBF-SVM`, `Logistic Regression`), plus confusion matrices and training convergence curves.
3. **Adversarial Robustness**: White-box attack sweeps (FGSM, PGD), physical domain constraints vs unconstrained curves, randomized smoothing, and black-box packet-space attacks.
4. **Techniques & Profiles**: Detection sensitivity by evasion transform (padding, jitter, shaping, fragmentation, tunneling) and application profile accuracy.
5. **Flow Inspector**: Filter and inspect all 4,001 test flows from `predictions.csv` with true vs predicted labels and class probability distributions.
6. **Real-Time Flow Scorer**: Live PyTorch inference using `checkpoint.pt` on customized or synthesized flow profiles.
7. **antod CLI Studio**: Interactive command builder with live background execution logs.

---

## 3. ANTOD CLI Reference

The CLI entry point is `antod`, which supports 8 subcommands:

```
usage: antod [-h] [--config CONFIG] [--name NAME] [--out OUT] [--epochs EPOCHS] [--model MODEL]
             [--n-flows N_FLOWS] [--seed SEED] [--device DEVICE] [--input INPUT]
             [--predictions-out PREDICTIONS_OUT] [--port PORT] [--host HOST] [--no-browser]
             {all,attack,calibrate,evaluate,generate,gui,predict,train}
```

### Commands

| Subcommand | Purpose | Key Outputs |
| :--- | :--- | :--- |
| `antod gui` | Launches the interactive web dashboard on localhost:8000. | Web Dashboard |
| `antod generate` | Generates synthetic flows or loads real capture data and splits them. | `data/processed/dataset.npz` |
| `antod train` | Trains a deep model (`cnn1d`, `mlp`, `hybrid`, `gru`) + fits baselines (`random_forest`, `svm`, `lr`). Computes calibration and base-rate precision. | `checkpoint.pt`, `metrics.json`, `predictions.csv`, training & confusion plots, baseline comparison |
| `antod attack` | Adversarial robustness evaluation: FGSM/PGD sweeps, domain-constrained vs unconstrained perturbations, randomized smoothing, and packet-space black-box attacks. | `attack_results.json`, `attack_sweep.md`, `robustness_constrained.png`, `evasion_rate.png`, `packet_attack.md` |
| `antod evaluate` | Detailed post-hoc evaluation: per-technique recall, per-profile accuracy, and multi-model adversarial transferability matrix. | `evaluation.json`, `per_technique_recall.md`, `per_profile_accuracy.md`, `transfer_matrix.md` |
| `antod calibrate` | Re-runs temperature scaling calibration and base-rate precision on an existing checkpoint without retraining. | Refreshed `metrics.json`, `predictions.csv` |
| `antod predict` | Classifies an unlabeled per-packet CSV from a real capture using a trained model. | Scored CSV with predictions and class probabilities (`predictions.csv`) |
| `antod all` | Chains `train` ➔ `attack` ➔ `evaluate` in a single run. | All experiment figures, tables, and metrics |

---

## 4. Existing Pre-Run Results

**All major model configurations have already been run and evaluated!** You do not need to re-run the time-consuming training or attack sweeps to view results.

The generated figures, tables, metrics, and predictions are committed and organized under `experiments/results/`:

```
experiments/results/
├── cnn1d/                # 1D Convolutional Neural Network (packet sequence view)
├── mlp/                  # Multi-Layer Perceptron (51 statistical flow features)
├── hybrid/               # Hybrid Model (CNN sequence + MLP statistics - best clean performer)
├── gru/                  # Gated Recurrent Unit (sequential baseline)
└── hybrid_advtrain/      # Hybrid Model with Adversarial Training defense (hardened)
```

### Inside each result folder (`experiments/results/<model>/`):

1. **Markdown Tables (`tables/`)** — Formatted tables ready to copy into documents:
   - `model_comparison.md`: Deep model vs Random Forest, RBF-SVM, and Logistic Regression.
   - `attack_sweep.md`: Accuracy, F1, and evasion rates under FGSM and PGD attacks.
   - `packet_attack.md`: Black-box problem-space packet attack results (evasion & overhead).
   - `smoothing.md`: Randomized smoothing defense results across noise standard deviations ($\sigma$).
   - `per_technique_recall.md`: Detection recall across specific evasion techniques (padding, timing jitter, fragmentation, tunneling, etc.).
   - `per_profile_accuracy.md`: Accuracy across application profiles (browsing, video, voip, bulk).
   - `transfer_matrix.md`: Adversarial transferability across different architectures.

2. **Visual Figures (`figures/`)** — Publication-quality charts (with matching `.csv` raw data):
   - `training_curves.png`: Epoch-by-epoch train/validation loss and accuracy.
   - `confusion_matrix.png`: Normalized confusion matrix (benign vs plain vs obfuscated).
   - `model_comparison.png`: Bar comparison across models and baselines.
   - `feature_importance.png`: Top informative flow features from Random Forest.
   - `robustness_constrained.png`: Accuracy curves under domain-valid vs unconstrained attacks.
   - `evasion_rate.png`: Adversarial evasion curve as a function of perturbation budget $\epsilon$.
   - `per_technique_recall.png`: Obfuscation technique recall breakdown.
   - `per_profile_accuracy.png`: Accuracy across traffic application profiles.

3. **Predictions & Checkpoints**:
   - `predictions.csv`: Per-flow predictions on the test set with true label, predicted label, recipe, and softmax probabilities.
   - `metrics.json` & `attack_results.json`: Machine-readable results and calibration stats.
   - `checkpoint.pt`: Saved PyTorch model weights and fitted feature scaler.

---

## 5. Command-Line Overrides & Examples

Any YAML configuration can be modified dynamically via command-line flags without touching the file:

```bash
# Override training epochs and output directory
antod train -c configs/cnn1d.yaml --epochs 10 --out experiments/runs/custom_run

# Override the model architecture and random seed
antod train -c configs/cnn1d.yaml --model cnn1d_small --seed 42

# Override the dataset size for a fast test
antod train -c configs/hybrid.yaml --n-flows 2000

# Force CPU or CUDA device
antod train -c configs/hybrid.yaml --device cpu
```

### Scoring External Captures (`antod predict`)

To score raw packets extracted from a PCAP file using an existing trained checkpoint:

```bash
antod predict -c configs/hybrid.yaml --input data/raw/packets.csv --predictions-out data/scored_packets.csv
```

---

## 6. Development & Verification

```bash
ruff check src tests      # linting
ruff format src tests     # code formatting
pytest -q                 # run unit test suite (~1 min)
```

## 7. Troubleshooting

- **`Python was not found`** on Windows:
  Windows attempted to run the Microsoft Store stub. This has been resolved by placing `python.exe` in `C:\Users\vivek\.local\bin`.
- **`antod : The term 'antod' is not recognized`**:
  Activate the virtual environment first (`& .\.venv\Scripts\Activate.ps1`) or call `.\.venv\Scripts\antod.exe`.
- **`no checkpoint at ...`**:
  Run `antod train` for that configuration first, or check that `--config` points to an experiment with an existing checkpoint under its output path.

