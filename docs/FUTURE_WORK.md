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
- [ ] **Seed variance.** Results are from one seed. At minimum three seeds with
      mean ± std are needed before any claim that model A beats model B.

## Known limitations

- **Synthetic data is the main threat to validity.** Profiles reproduce
  qualitative signatures from the literature, not any specific network.
  Absolute accuracy is an upper bound; only *relative* conclusions (which model,
  which technique, how much adversarial training helps) should be read from it.
- **Fixed sequence length of 128 packets.** Longer flows are truncated. An
  attacker who front-loads a benign-looking prologue could push the malicious
  behaviour past the window. A sliding-window or multi-window vote would close
  this.
- **Attacks perturb features, not packets.** The sequence-space attack respects
  pad-only / delay-only, but the statistics-space attack perturbs the 51
  aggregates directly, and not every point in the feature box corresponds to a
  realisable flow (e.g. `n_packets` and `duration` are perturbed independently).
  A packet-level attack that modifies the flow and *re-extracts* features would
  be the strictly correct threat model.
- **Only white-box and cross-architecture transfer.** No black-box query
  attacks (e.g. score-based or boundary attacks), which is the realistic setting
  for an attacker who can probe a deployed detector but never sees its weights.
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
- Realistic base rates (e.g. 95/3/2) with metrics reported at fixed FPR.

### Models
- Attention or a small transformer over the packet sequence, to test whether
  position-aware models beat convolution on long-range cadence.
- A bidirectional GRU/LSTM as a second sequence baseline.
- Sliding-window inference for flows longer than 128 packets.
- Calibration (temperature scaling) so probabilities are usable for alerting
  thresholds.

### Adversarial
- Packet-level attacks that re-extract features after each step.
- Black-box query attacks (NES, SimBA, HopSkipJump) against the deployed model.
- Certified robustness via randomised smoothing bounds (Cohen et al.) rather than
  just the empirical majority vote.
- TRADES / MART loss instead of plain PGD adversarial training, and a sweep over
  `adv_ratio`.
- Detection of adversarial examples themselves (input reconstruction error,
  feature-squeezing disagreement).

### Engineering
- GPU support is wired (`device: auto`) but untested; verify on a CUDA box.
- `num_workers > 0` in dataloaders is untested on Windows.
- Save per-flow predictions alongside metrics so failure cases can be inspected.
- A `predict` CLI command that scores a new per-packet CSV with a checkpoint.
- CI workflow running `ruff` + `pytest` on push.
- Pin dependency versions in `pyproject.toml` once the final run is done, so the
  reported numbers stay reproducible.

### Report
- Literature survey of ≥20 recent papers (report template requirement).
- Map to SDGs (template §5.7) — SDG 9 (industry/infrastructure) and SDG 16
  (institutions) are the obvious candidates.
