# Codebase walkthrough

Everything lives under `src/antod/`, installed as the `antod` package. Data flows
left to right through the layers below; nothing imports from a layer to its right.

```
data  →  models  →  train  →  adversarial  →  evaluate  →  cli
                       ↑                          ↑
                    utils (metrics, plots, common) ┘
config (YAML → dataclasses, used by cli)
```

## Repository layout

```
configs/                YAML experiment definitions (one per model / defense)
data/                   raw captures, generated .npz datasets  (git-ignored)
docs/                   this documentation + college templates
experiments/            results, tables, figures written by the CLI
scripts/run_all.py      reproduce every experiment in order
src/antod/              the package
tests/                  pytest suite (~200 tests)
pyproject.toml          dependencies, `antod` console script, ruff/pytest config
```

## `src/antod/data/` — traffic, obfuscation, features

| File | What it does |
| --- | --- |
| `profiles.py` | 12 packet-level generators. Each returns an `(n, 3)` array of `[timestamp, size, direction]`. Six benign (`web_browsing`, `video_streaming`, `voip`, `file_download`, `dns_query`, `ssh_interactive`) and six malicious (`c2_beacon`, `data_exfiltration`, `port_scan`, `brute_force`, `ddos_flood`, `reverse_shell`). `reverse_shell` deliberately overlaps `ssh_interactive` so the task isn't solvable by a threshold. |
| `obfuscation.py` | Eight transforms on packet arrays: `block_padding`, `random_padding`, `fragmentation`, `timing_jitter`, `constant_rate_shaping`, `dummy_injection`, `tunnel_encapsulation`, `protocol_mimicry`. `Recipe` composes them in a fixed sensible order (mimicry first, tunnel last); `sample_recipe` draws 1–3 with random intensities; `single_recipe` is for per-technique probes. Every transform preserves byte volume and packet validity. |
| `synth.py` | `Flow` dataclass, `SynthConfig`, `generate_dataset` (balanced three classes, benign flows also obfuscated at `benign_obfuscation_rate`), `generate_technique_probe` (malicious flows with exactly one transform, for the per-technique breakdown). Defines `LABEL_NAMES`, `BENIGN=0`, `MALICIOUS_PLAIN=1`, `MALICIOUS_OBFUSCATED=2`. |
| `features.py` | The two views. `sequence_tensor(pkts) → (4, 128)` for the CNN. `flow_statistics(pkts) → dict` of 51 features and `stats_vector` for the MLP. `FEATURE_NAMES` fixes the column order. `size_grid_*` excludes MTU-sized packets (a padder can't grow a 1300 B packet to 1536 without exceeding the MTU, so clamped packets carry no evidence). |
| `datasets.py` | `FlowDataset` (numpy container with `save`/`load`), `StatsScaler` (arcsinh → standardise, invertible), `FeatureConstraints` (per-feature legal ranges used to project attacks back into physically valid traffic), `stratified_split` (scaler fitted on train only), `FlowTensorDataset` + `make_loaders` for torch. |
| `real_loader.py` | `load_packet_csv` reads per-packet CSVs (tshark export) into `Flow`s; `build_obfuscated_classes` applies the same recipes to real flows; `ATTACK_LABELS` maps CIC/UNSW attack names. See [DATA.md](DATA.md). |

## `src/antod/models/` — the classifiers

| File | What it does |
| --- | --- |
| `base.py` | `FlowClassifier` ABC: every model takes `(seq, stats)` and ignores what it doesn't use, and declares `uses_seq` / `uses_stats` (tested against real gradient flow). `@register` / `build_model(name)` registry. |
| `cnn1d.py` | `ConvBlock` (conv → BN → ReLU → pool → dropout), `SequenceTrunk` (three conv stages + average *and* max pooling over time), `CNN1D` (trunk + head). Registered as `cnn1d` and `cnn1d_small`. |
| `mlp.py` | `MLP` over the 51 statistics (`mlp`, `mlp_wide`) and `HybridCNNMLP` (`hybrid`) which joins a `SequenceTrunk` and a statistics trunk at the head. |
| `rnn.py` | `GRUClassifier` (`gru`): bidirectional GRU packed to each flow's true length so padding never leaks into the state, masked-mean pooled. |
| `baselines.py` | `SklearnBaseline` wrapper with `feature_importance()`; factories `random_forest`, `rbf_svm` (calibrated via `CalibratedClassifierCV`), `logistic_regression`; `build_baseline(name)`. |

## `src/antod/train.py` — training loop

`TrainConfig` (epochs, lr, scheduler, patience, `selection_metric`, `adv_ratio`),
`Trainer.fit(splits) → TrainResult`. Selects on validation **macro-F1** by
default because collapsing class 2 into class 1 barely hurts accuracy but kills
macro-F1. Accepts an optional `adversary` callable so adversarial training uses
the exact same loop. `save_checkpoint` / `load_checkpoint` bundle weights, scaler
and config together.

## `src/antod/adversarial/` — attacks and defenses

| File | What it does |
| --- | --- |
| `attacks.py` | `AttackConfig` (fgsm/pgd, surface seq/stats/both, eps, steps, constrained, targeted). `sequence_bounds` builds the one-sided box: size magnitude may only grow, log-IAT may only increase, direction and mask frozen, padding slots frozen. `stats_bounds` does the same for statistics via `FeatureConstraints`. `run_attack` is the PGD/FGSM loop with projection onto eps-ball ∩ domain box; it refuses to perturb a surface the model doesn't read. `make_adversary` binds a config for the trainer. `perturbation_norms` reports how big the perturbation was. |
| `packet_attack.py` | Packet-space black-box attack: greedy hill-climb over pad/delay edits on the raw packet array, re-extracting features after each proposal and querying only output probabilities. `evaluate_packet_attack` reports evasion before/after, queries and byte overhead. |
| `defenses.py` | `adversarial_training` (builds an adversary, hands it to `Trainer`), `smoothed_predict` (randomised smoothing — majority vote over Gaussian-noised copies; never touches direction/mask/padding), `DefenseConfig`. |

## `src/antod/evaluate.py` — answering the real questions

`evaluate_clean`, `evaluate_attack` (supports `source_model` for transfer),
`attack_sweep`, `robustness_curve` (metric vs eps), `constrained_vs_unconstrained`,
`per_technique_recall` (fresh probe sets, one transform each),
`per_profile_accuracy`, `per_recipe_accuracy`, `transfer_matrix`,
`smoothing_sweep`, `fit_baselines` / `evaluate_baselines`.

## `src/antod/calibration.py` and `inference.py`

`calibration.py`: temperature scaling fitted on validation logits plus expected
calibration error before/after — makes the probabilities honest without changing
the argmax. `inference.py`: `score_flow_windows` slides the 128-packet window
across a flow of any length and combines per-window predictions (`mean` or
`max_malicious`).

## `src/antod/utils/`

| File | What it does |
| --- | --- |
| `metrics.py` | `Metrics` dataclass and `compute_metrics`: accuracy, macro/weighted F1, per-class P/R/F1, confusion matrix, macro AUC, plus `obfuscated_recall`, `malicious_recall` (classes 1+2 collapsed) and `false_positive_rate`. `recall_by_group` for per-technique/profile breakdowns; `precision_at_base_rate` re-weights to 90/99/99.9% benign prevalence. |
| `plots.py` | Matplotlib figures: confusion matrix, training curves (two panels, never a twin axis), robustness curves, ranked horizontal bars, model comparison. Validated 3-hue palette, direct labels, a CSV written beside every PNG. |
| `common.py` | `set_seed`, `get_logger`, `pick_device`, `save_json`/`load_json`, `format_table`, `write_markdown_table`. |

## `src/antod/config.py` and `cli.py`

`config.py` turns YAML into nested dataclasses (`ExperimentConfig` → dataset /
split / train / defense / attacks / output). Unknown keys are a hard error so a
typo can't silently run at defaults.

`cli.py` is the `antod` command: `generate`, `train`, `attack`, `evaluate`,
`all`, `predict` (score an unlabelled per-packet CSV), each driven by `--config`. Outputs go to `output.dir` as `metrics.json`,
`attack_results.json`, `evaluation.json`, `checkpoint.pt`, `config.yaml`,
`figures/*.png(+csv)`, `tables/*.md`.

## `scripts/`

`seed_variance.py` re-runs a config under several seeds and reports mean ± std.

### `run_all.py`

Runs generate → train ×4 → attack ×4 → evaluate ×4 in that order (the transfer
matrix needs every checkpoint first). `--quick` does a 2k-flow / 4-epoch smoke
run into `experiments/runs/quick_*`.

## `tests/`

| File | Covers |
| --- | --- |
| `test_obfuscation.py` | Packet validity of every profile and transform, volume preservation, cadence, recipe determinism, dataset balance and labelling. |
| `test_features.py` | Both views, scaler round-trip, `FeatureConstraints`, splits, loaders, persistence. |
| `test_models.py` | Shapes, gradient routing vs declared surfaces, dead-parameter check, eval determinism, baselines. |
| `test_attacks.py` | Pad-only/delay-only invariants, eps-ball, domain box, unconstrained ≥ constrained, targeted attacks, surface routing, smoothing. |
| `test_inference.py` | Calibration/ECE, base-rate precision, sliding-window inference. |
| `test_packet_attack.py` | Packet-space attack legality and monotonicity, GRU padding invariance. |
| `test_train.py` | Trainer, early stopping, best-weights restore, adversarial training, checkpoints, config parsing, metrics semantics. |

Run with `pytest` (or `python -m pytest`) from the repo root.
