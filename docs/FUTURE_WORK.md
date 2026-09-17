# Future work and known gaps

Honest list of what is missing, what is weak, and what would be worth doing next.

## Not done yet

- [ ] **Full experiment run.** The pipeline is verified end to end with
      `scripts/run_all.py --quick`, but the full 20k-flow / 60-epoch run has not
      been completed and `experiments/results/` is empty. Run
      `python scripts/run_all.py` (≈1 hour on CPU) to produce the numbers,
      figures and tables the report needs.
- [ ] **Report and book chapter.** `docs/templates/` holds the college report
      template and the book-chapter guideline; neither has been filled in yet.
      Both should be written *after* the full run, from the generated
      `tables/*.md` and `figures/*.png`.
- [ ] **Validation on real captures.** Every number so far is on synthetic
      traffic. The real-data loader exists (`antod.data.real_loader`) but has not
      been exercised on an actual PCAP. Exporting a slice of CIC-IDS2017 with
      `tshark` and running `source: real` is the single most valuable next step —
      it turns the results from "upper bound on a simulator" into "measured on
      traffic".
- [ ] **Seed variance.** `scripts/seed_variance.py` is in place; run it for each
      model config (3 seeds) and quote mean ± std in the report.

## Known limitations

- **Synthetic data is the main threat to validity.** Profiles reproduce
  qualitative signatures from the literature, not any specific network.
  Absolute accuracy is an upper bound; only *relative* conclusions (which model,
  which technique, how much adversarial training helps) should be read from it.
- **Fixed sequence length of 128 packets during training.** `antod.inference.score_flow_windows`
  now slides the window across the whole flow at inference (mean or
  max-malicious combination), but training still sees only the first window.
- **Gradient attacks perturb features, not packets.** The statistics-space PGD
  perturbs the 51 aggregates directly, and not every point in the feature box is
  a realisable flow. The packet-space attack (`packet_attack.py`) is the strictly
  correct threat model and is now reported alongside; treat the gradient numbers
  as an upper bound on the threat.
- **Black-box search is greedy.** The packet-space attack accepts any edit that
  raises the benign probability; a smarter search (NES, SimBA) would evade with
  fewer queries and less overhead, so its evasion rate is a lower bound.
- **Adversarial training against a single attack config.** The defended model is
  trained against PGD-5 on both surfaces. Robustness to attacks with a
  different norm, surface or step schedule is not measured.
- **No per-packet payload features at all.** Deliberate (that is what DPI does
  and what obfuscation defeats), but a production system would combine both.
- **Class balance is artificial.** Real traffic is >99% benign. The
  false-positive rate reported here is on a balanced test set; at realistic base
  rates the same FPR produces far more alerts than true positives.
- **`rbf_svm` scales poorly.** Calibrated SVM on 14k training rows takes
  minutes; it is dropped from most configs for that reason.

## Improvements worth making

### Data
- Per-packet CSV export of CIC-IDS2017 / UNSW-NB15 PCAPs and a `real` config.
- More profiles: DNS tunnelling, HTTP/2 multiplexing, QUIC, IoT telemetry.
- More transforms: packet reordering, flow splitting across connections,
  domain fronting, traffic morphing with learned target distributions.
- [x] Base-rate-adjusted precision and alerts-per-10k at 90/99/99.9% benign (`precision_at_base_rate`, in `metrics.json`).
- Training at realistic base rates (e.g. 95/3/2) rather than only re-weighting at evaluation.

### Models
- Attention or a small transformer over the packet sequence, to test whether
  position-aware models beat convolution on long-range cadence.
- [x] Bidirectional GRU (`gru`, packed to true length so padding never leaks in) — `configs/gru.yaml`.
- [x] Sliding-window inference for flows longer than 128 packets (`antod/inference.py`).
- [x] Calibration (temperature scaling + ECE) — `antod/calibration.py`, reported in `metrics.json`.

### Adversarial
- [x] Packet-level attack that edits raw packets (pad/delay only) and re-extracts features — `adversarial/packet_attack.py`, run by `antod attack`. It is also black-box (queries only, no gradients), so it is comparable across architectures.
- Stronger black-box search (NES / SimBA-style) — the current attack is a greedy hill-climb.
- Certified robustness via randomised smoothing bounds (Cohen et al.) rather than
  just the empirical majority vote.
- TRADES / MART loss instead of plain PGD adversarial training, and a sweep over
  `adv_ratio`.
- Detection of adversarial examples themselves (input reconstruction error,
  feature-squeezing disagreement).

### Engineering
- GPU support is wired (`device: auto`) but untested; verify on a CUDA box.
- `num_workers > 0` in dataloaders is untested on Windows.
- [x] Per-flow predictions (`predictions.csv`) written by `antod train`.
- [x] `antod predict --input packets.csv` scores a new per-packet CSV with a checkpoint.
- [x] CI workflow (`.github/workflows/ci.yml`): ruff + pytest + quick smoke run on 3.11/3.12.
- [x] Exact versions pinned in `requirements-lock.txt` (pyproject keeps loose bounds).

### Report
- Literature survey of ≥20 recent papers (report template requirement).
- Map to SDGs (template §5.7) — SDG 9 (industry/infrastructure) and SDG 16
  (institutions) are the obvious candidates.
