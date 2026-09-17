# ANTOD — Adversarial Network Traffic Obfuscation Detection

Deep Learning Mini Project (CS722T2C), V Semester B.E. CSE
Sahyadri College of Engineering & Management, Mangaluru — VTU, Belagavi

| Name | USN |
| --- | --- |
| Vivek Neeralagi (Team Lead) | 4SF23CS246 |
| Muhammad Shahbik | 4SF23CS113 |
| Aravind P Sagar | 4SF23CS030 |
| J Aditya | 4SF24CS408 |

Guide: Mrs. Varsha M, Assistant Professor, Department of CSE

---

## 1. The problem

Deep Packet Inspection (DPI) engines detect malware by matching byte patterns
inside packets. Modern malware defeats this without changing what it sends —
only *how* it sends it:

| Technique | What it does | Why DPI misses it |
| --- | --- | --- |
| Padding | Grows every packet to a fixed block size | Size signatures vanish |
| Timing jitter / constant-rate shaping | Randomises or flattens inter-arrival times | Beacon periodicity vanishes |
| Fragmentation | Splits records into sub-MTU pieces | Signatures span fragment boundaries |
| Tunnelling (VPN / TLS) | Wraps the flow in an encrypted envelope | Payload is unreadable |
| Protocol mimicry | Reshapes sizes and timing to look like video/web | Statistically looks benign |
| Dummy injection | Inserts chaff packets | Burst structure is broken |

Once traffic is encrypted and reshaped, the only thing a defender can still see
is the **shape of the flow**: the sequence of packet sizes, directions and
inter-arrival times, plus aggregate statistics. That is exactly what this project
learns from.

## 2. What we built

A deep learning detector that classifies a network flow into one of three classes:

| Label | Class | Meaning |
| --- | --- | --- |
| 0 | `benign` | Ordinary traffic, whether or not it is padded/tunnelled |
| 1 | `malicious_plain` | Attack traffic making no attempt to hide |
| 2 | `malicious_obfuscated` | Attack traffic reshaped to evade DPI |

Class 2 is the one that matters — it is the traffic a signature engine already
lets through.

Then we stress-test the detector with **adversarial machine learning**: FGSM and
PGD attacks that try to push obfuscated malicious flows across the decision
boundary, and adversarial training that hardens the model against them.

### Models

| Model | Input | Params | Purpose |
| --- | --- | --- | --- |
| **1D-CNN** | packet sequence `(4, 128)` | ~126k | Headline model — reads *where* in the flow obfuscation happens |
| **MLP** | 51 flow statistics | ~56k | Ablation — does the convolution earn its cost? |
| **Bi-GRU** | packet sequence `(4, 128)` | ~50k | Second sequence baseline — does order/long-range context matter? |
| **Hybrid** | both | ~149k | Are the two views complementary? |
| Random forest / RBF-SVM / logistic regression | 51 flow statistics | — | Classical baselines; RF also gives feature importances |

### The two feature views

**Sequence view** (for the CNN): first 128 packets of a flow, four channels —
signed size (`size/MTU × direction`), log inter-arrival time, direction, and a
validity mask so padding isn't mistaken for a real zero-size packet.

**Statistics view** (for the MLP and baselines): 51 aggregates — volume, size and
timing moments, plus features designed to expose obfuscation:
`size_grid_{64,128,256}` (block-padding fingerprint), `unique_size_ratio` and
`mode_size_frac` (size-diversity collapse), `iat_cv` and `iat_autocorr_peak`
(timing regularity — catches C2 beacons, and shows when shaping has erased them).

### The dataset

A packet-level traffic simulator (`antod.data.synth`) generates flows from 12
application profiles — 6 benign (web, video, VoIP, file download, DNS, SSH) and
6 malicious (C2 beacon, exfiltration, port scan, brute force, DDoS, reverse
shell). Eight obfuscation transforms are then applied in sampled recipes.

Why synthetic: obfuscation is **applied, not inferred**, so the label and the
exact recipe are known for every flow. No public corpus annotates which flows
were padded or tunnelled. This is what makes the per-technique breakdown
("which evasion actually defeats the model?") possible. Real captures are also
supported via per-packet CSVs from `tshark` — see [DATA.md](DATA.md).

Two design decisions keep the experiment honest:

1. **Benign traffic is obfuscated too** (35% of it) and stays labelled benign.
   A corporate VPN user is benign and obfuscated at the same time. Without this,
   the model learns "obfuscated ⇒ malicious" and flags every VPN user.
2. **Protocol mimicry preserves byte volume.** An attacker can reshape traffic
   but cannot decide not to send the bytes. Without this constraint the
   detection problem would be ill-posed rather than merely hard.

### The adversarial part

Standard FGSM/PGD perturb every input in either direction. Traffic doesn't work
that way. An attacker controlling a malicious flow can:

- **pad** a packet — make it larger, never smaller (the payload must fit);
- **delay** a packet — send it later, never earlier (causality);

and can change neither its direction nor whether it exists. So our attacks
operate inside a **one-sided box**, not a symmetric ball. We keep the
unconstrained attack too, because the gap between the two is itself a result: it
measures how much a naive robustness evaluation overstates the threat.

Evaluation goes beyond accuracy:

| Measure | Question it answers |
| --- | --- |
| Obfuscated recall | Are we catching the flows DPI misses? |
| Malicious recall | Was the attack caught at all, regardless of disguise? |
| False-positive rate | Would this flag legitimate VPN users? |
| Evasion rate | What share of attacks get through as *benign*? (what an attacker optimises) |
| Per-technique recall | *Which* evasion technique defeats the model? |
| Robustness curve | How does accuracy decay as the attacker's budget grows? |
| Transfer matrix | Does the attacker need our weights? |
| Constrained vs unconstrained | How much does ignoring physics overstate the threat? |

Defenses: adversarial training (PGD examples mixed into half of every batch) and
randomised smoothing at inference (majority vote over noisy copies).

## 3. Objectives

1. Build a reproducible, labelled dataset of benign, plain-malicious and
   obfuscated-malicious flows.
2. Train and compare a 1D-CNN, an MLP, a hybrid and classical baselines.
3. Determine which obfuscation techniques are hardest to detect.
4. Measure robustness under domain-constrained FGSM/PGD and transfer attacks.
5. Harden the model with adversarial training and quantify the trade-off.

## 4. Scope note

The obfuscation transforms operate on synthetic packet records and extracted
feature arrays, not on live sockets. This is a defensive research tool for
producing labelled training data and measuring detector robustness; it is not an
evasion utility.

## 5. Headline results (seed 42, 20k flows)

| Model | Accuracy | Obf. recall | PGD ε=0.1 (constrained) | Packet-space evasion |
| --- | --- | --- | --- | --- |
| Random forest | 97.2% | 94.9% | — | — |
| 1D-CNN | 94.1% | 88.6% | 51.2% | 14.1% |
| Bi-GRU | 92.9% | 88.1% | 48.6% | 19.2% |
| MLP | 96.2% | 93.1% | 58.8% | 9.1% |
| Hybrid | 96.9% | 94.2% | 72.8% | 10.1% |
| Hybrid + adversarial training | 96.5% | 94.2% | 93.8% | 10.6% |

Hardest obfuscation technique: fragmentation (65% recall). Adversarial training
costs 0.4 points of clean accuracy, lifts constrained-PGD accuracy at ε=0.1 from
72.8% to 93.8%, but does not help against the packet-space black-box attack.
At 99% benign prevalence the defended hybrid's FPR translates to ~18% precision
(444 false alerts per 10k flows) — the deployment-relevant number.

## 6. Where things are

- Codebase walkthrough: [CODEBASE.md](CODEBASE.md)
- How to run: [RUNNING.md](RUNNING.md)
- Data formats and the real-capture path: [DATA.md](DATA.md)
- Future work and known gaps: [FUTURE_WORK.md](FUTURE_WORK.md)
- College templates: `docs/templates/`
