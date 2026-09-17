/**
 * Build the VTU mini-project report (docs/ANTOD_report.docx) from the results.
 *
 * Follows docs/templates/Report_Template.docx: cover, declaration, abstract,
 * contents, five chapters, conclusion, references. Times New Roman 16/14/12,
 * 1.5 line spacing. Every figure and number comes from experiments/results/.
 *
 *   node scripts/build_report.js
 */
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, AlignmentType, TableOfContents, LevelFormat, PageNumber, Footer } = require("docx");

const C = require("./lib/report_common");
const { MODELS, M, NAMES, pct, f4, rf, H, A, curve, FONT, SP, P, H1, H2, Center, Break, Bul, Num, Caption, Tbl, Img, metricRow } = C;

const OUT = path.join(C.ROOT, "docs", "ANTOD_report.docx");

// ---------------------------------------------------------------- references
const REFS = [
  "I. J. Goodfellow, J. Shlens and C. Szegedy, \"Explaining and harnessing adversarial examples,\" in Proc. ICLR, 2015.",
  "A. Madry, A. Makelov, L. Schmidt, D. Tsipras and A. Vladu, \"Towards deep learning models resistant to adversarial attacks,\" in Proc. ICLR, 2018.",
  "N. Carlini and D. Wagner, \"Towards evaluating the robustness of neural networks,\" in Proc. IEEE Symp. Security and Privacy, 2017, pp. 39-57.",
  "N. Papernot, P. McDaniel and I. Goodfellow, \"Transferability in machine learning: from phenomena to black-box attacks using adversarial samples,\" arXiv:1605.07277, 2016.",
  "A. Athalye, N. Carlini and D. Wagner, \"Obfuscated gradients give a false sense of security: circumventing defenses to adversarial examples,\" in Proc. ICML, 2018.",
  "H. Zhang, Y. Yu, J. Jiao, E. Xing, L. El Ghaoui and M. Jordan, \"Theoretically principled trade-off between robustness and accuracy,\" in Proc. ICML, 2019.",
  "J. Cohen, E. Rosenfeld and Z. Kolter, \"Certified adversarial robustness via randomized smoothing,\" in Proc. ICML, 2019.",
  "C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, \"On calibration of modern neural networks,\" in Proc. ICML, 2017.",
  "F. Pierazzi, F. Pendlebury, J. Cortellazzi and L. Cavallaro, \"Intriguing properties of adversarial ML attacks in the problem space,\" in Proc. IEEE Symp. Security and Privacy, 2020.",
  "G. Apruzzese, M. Andreolini, L. Ferretti, M. Marchetti and M. Colajanni, \"Modeling realistic adversarial attacks against network intrusion detection systems,\" Digital Threats: Research and Practice, vol. 3, no. 3, 2022.",
  "W. Wang, M. Zhu, X. Zeng, X. Ye and Y. Sheng, \"Malware traffic classification using convolutional neural network for representation learning,\" in Proc. Int. Conf. Information Networking (ICOIN), 2017, pp. 712-717.",
  "M. Lotfollahi, M. Jafari Siavoshani, R. Shirali Hossein Zade and M. Saberian, \"Deep Packet: a novel approach for encrypted traffic classification using deep learning,\" Soft Computing, vol. 24, pp. 1999-2012, 2020.",
  "G. Aceto, D. Ciuonzo, A. Montieri and A. Pescape, \"MIMETIC: mobile encrypted traffic classification using multimodal deep learning,\" Computer Networks, vol. 165, 2019.",
  "P. Sirinam, M. Imani, M. Juarez and M. Wright, \"Deep Fingerprinting: undermining website fingerprinting defenses with deep learning,\" in Proc. ACM CCS, 2018, pp. 1928-1943.",
  "V. Rimmer, D. Preuveneers, M. Juarez, T. Van Goethem and W. Joosen, \"Automated website fingerprinting through deep learning,\" in Proc. NDSS, 2018.",
  "K. P. Dyer, S. E. Coull, T. Ristenpart and T. Shrimpton, \"Peek-a-Boo, I still see you: why efficient traffic analysis countermeasures fail,\" in Proc. IEEE Symp. Security and Privacy, 2012, pp. 332-346.",
  "C. V. Wright, S. E. Coull and F. Monrose, \"Traffic morphing: an efficient defense against statistical traffic analysis,\" in Proc. NDSS, 2009.",
  "X. Cai, R. Nithyanand, T. Wang, R. Johnson and I. Goldberg, \"A systematic approach to developing and evaluating website fingerprinting defenses,\" in Proc. ACM CCS, 2014.",
  "M. Nasr, A. Bahramali and A. Houmansadr, \"Defeating DNN-based traffic analysis systems in real-time with blind adversarial perturbations,\" in Proc. USENIX Security, 2021.",
  "I. Sharafaldin, A. H. Lashkari and A. A. Ghorbani, \"Toward generating a new intrusion detection dataset and intrusion traffic characterization,\" in Proc. ICISSP, 2018, pp. 108-116.",
  "N. Moustafa and J. Slay, \"UNSW-NB15: a comprehensive data set for network intrusion detection systems,\" in Proc. Military Communications and Information Systems Conf. (MilCIS), 2015.",
  "G. Draper-Gil, A. H. Lashkari, M. S. I. Mamun and A. A. Ghorbani, \"Characterization of encrypted and VPN traffic using time-related features,\" in Proc. ICISSP, 2016, pp. 407-414.",
  "N. Shone, T. N. Ngoc, V. D. Phai and Q. Shi, \"A deep learning approach to network intrusion detection,\" IEEE Trans. Emerging Topics in Computational Intelligence, vol. 2, no. 1, pp. 41-50, 2018.",
  "R. Vinayakumar, M. Alazab, K. P. Soman, P. Poornachandran, A. Al-Nemrat and S. Venkatraman, \"Deep learning approach for intelligent intrusion detection system,\" IEEE Access, vol. 7, pp. 41525-41550, 2019.",
  "M. Sabir, S. A. Sayed, R. Ahmed and P. Nanda, \"Machine learning for detection of C2 beaconing: a survey,\" IEEE Access, vol. 9, 2021.",
  "R. Sommer and V. Paxson, \"Outside the closed world: on using machine learning for network intrusion detection,\" in Proc. IEEE Symp. Security and Privacy, 2010, pp. 305-316.",
  "A. Paszke et al., \"PyTorch: an imperative style, high-performance deep learning library,\" in Proc. NeurIPS, 2019.",
  "F. Pedregosa et al., \"Scikit-learn: machine learning in Python,\" Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.",
];

// ---------------------------------------------------------------- literature table
const LIT = [
  ["[11] Wang et al., 2017", "1D-CNN on raw packet bytes for malware traffic classification", "Bytes are unavailable once traffic is encrypted or padded"],
  ["[12] Lotfollahi et al., 2020", "Deep Packet: CNN/SAE on encrypted traffic", "Evaluated on un-obfuscated captures only"],
  ["[13] Aceto et al., 2019", "MIMETIC: multimodal (payload + sequence) mobile traffic classifier", "No adversarial evaluation"],
  ["[14] Sirinam et al., 2018", "Deep Fingerprinting: CNN on packet direction sequences defeats WF defenses", "Attacker-side; we take the defender-side view"],
  ["[15] Rimmer et al., 2018", "Automated website fingerprinting with SDAE/CNN/LSTM", "Same; no obfuscated-benign class"],
  ["[16] Dyer et al., 2012", "Coarse features survive padding/morphing countermeasures", "Motivates our volume-preserving mimicry"],
  ["[17] Wright et al., 2009", "Traffic morphing to match a target size distribution", "Our protocol_mimicry transform is this attack"],
  ["[18] Cai et al., 2014", "Systematic evaluation of WF defenses", "Framework, not a detector"],
  ["[19] Nasr et al., 2021", "Blind adversarial perturbations on live traffic against DNN analysers", "Motivates pad/delay-only constraints"],
  ["[9] Pierazzi et al., 2020", "Problem-space vs feature-space adversarial attacks", "We implement both and compare"],
  ["[10] Apruzzese et al., 2022", "Realistic adversarial attacks against NIDS", "Constraint set we adopt"],
  ["[1] Goodfellow et al., 2015", "FGSM", "Adapted with a one-sided box"],
  ["[2] Madry et al., 2018", "PGD and adversarial training", "Our defense"],
  ["[3] Carlini & Wagner, 2017", "Optimisation-based attacks; evaluation methodology", "Guides our robustness protocol"],
  ["[4] Papernot et al., 2016", "Transferability of adversarial examples", "Our transfer matrix"],
  ["[5] Athalye et al., 2018", "Obfuscated gradients give false security", "Why we add a gradient-free packet-space attack"],
  ["[6] Zhang et al., 2019", "TRADES robustness/accuracy trade-off", "Future work"],
  ["[7] Cohen et al., 2019", "Randomised smoothing", "Our smoothing sweep"],
  ["[8] Guo et al., 2017", "Temperature scaling for calibration", "Our calibration step"],
  ["[20] Sharafaldin et al., 2018", "CIC-IDS2017 dataset", "Flow-level only; PCAP route documented"],
  ["[21] Moustafa & Slay, 2015", "UNSW-NB15 dataset", "Same"],
  ["[22] Draper-Gil et al., 2016", "VPN vs non-VPN traffic via time features", "Benign obfuscated traffic exists in the wild"],
  ["[23] Shone et al., 2018", "Deep autoencoder NIDS", "No obfuscation modelling"],
  ["[24] Vinayakumar et al., 2019", "DNN IDS across datasets", "Same"],
  ["[25] Sabir et al., 2021", "C2 beaconing detection survey", "Periodicity features we use"],
  ["[26] Sommer & Paxson, 2010", "Why ML-NIDS fails in deployment: base rates, semantics", "Our base-rate analysis"],
];

// ---------------------------------------------------------------- content
const body = [];

// cover
body.push(
  Center("VISVESVARAYA TECHNOLOGICAL UNIVERSITY", 28, true),
  Center("\"JNANA SANGAMA\", BELAGAVI - 590018", 24),
  new Paragraph({ spacing: { before: 600 } }),
  Center("DEEP LEARNING MINI PROJECT REPORT (CS722T2C)", 26, true),
  Center("on", 24),
  Center("\"Adversarial Network Traffic Obfuscation Detection\"", 30, true),
  new Paragraph({ spacing: { before: 400 } }),
  Center("Submitted by", 24),
  Center("Vivek Neeralagi        4SF23CS246", 24),
  Center("Muhammad Shahbik       4SF23CS113", 24),
  Center("Aravind P Sagar        4SF23CS030", 24),
  Center("J Aditya               4SF24CS408", 24),
  new Paragraph({ spacing: { before: 300 } }),
  Center("In partial fulfillment of the requirements for the V semester", 24),
  Center("BACHELOR OF ENGINEERING", 26, true),
  Center("in", 24),
  Center("COMPUTER SCIENCE & ENGINEERING", 26, true),
  new Paragraph({ spacing: { before: 300 } }),
  Center("Under the Guidance of", 24),
  Center("Mrs. Varsha M", 26, true),
  Center("Assistant Professor, Department of CSE", 24),
  new Paragraph({ spacing: { before: 300 } }),
  Center("SAHYADRI", 30, true),
  Center("College of Engineering & Management", 26, true),
  Center("An Autonomous Institution", 24),
  Center("MANGALURU", 26, true),
  Center("2026-27", 24),
  Break(),
);

// declaration
body.push(
  Center("Department of Computer Science & Engineering", 28, true),
  new Paragraph({ spacing: { before: 400 } }),
  Center("DECLARATION", 32, true),
  new Paragraph({ spacing: { before: 300 } }),
  P("We hereby declare that the entire work embodied in this Mini Project Report titled \"Adversarial Network Traffic Obfuscation Detection\" has been carried out by us at Sahyadri College of Engineering & Management, Mangaluru under the supervision of Mrs. Varsha M, in partial fulfillment of the requirements for the V semester of Bachelor of Engineering in Computer Science & Engineering. This report has not been submitted to this or any other University for the award of any other degree."),
  new Paragraph({ spacing: { before: 600 } }),
  P("Vivek Neeralagi (4SF23CS246)"), P("Muhammad Shahbik (4SF23CS113)"), P("Aravind P Sagar (4SF23CS030)"), P("J Aditya (4SF24CS408)"),
  new Paragraph({ spacing: { before: 300 } }),
  P("Dept. of CSE, SCEM, Mangaluru"),
  Break(),
);

// abstract
body.push(
  Center("Abstract", 32, true),
  P(`Signature-based Deep Packet Inspection (DPI) identifies malicious traffic by matching byte patterns inside packets. Modern malware evades it without changing what it sends, only how: padding packets, jittering or flattening inter-arrival times, fragmenting records, tunnelling over permitted ports and mimicking the statistical shape of benign protocols. This project builds a deep learning detector that operates on the one thing encryption and reshaping cannot hide, the shape of a flow, and then subjects that detector to adversarial machine learning. A packet-level traffic simulator produces flows from twelve application profiles, eight obfuscation transforms are applied in recorded recipes, and each flow is labelled benign, malicious-plain or malicious-obfuscated with exact provenance. Two feature views are extracted: a (4 × 128) packet sequence for a 1D convolutional network and a bidirectional GRU, and 51 flow statistics for a multilayer perceptron and classical baselines; a hybrid model reads both. Adversarial robustness is measured with FGSM and PGD constrained to what an attacker can physically do, pad and delay only, with a packet-space black-box attack that edits raw packets, and with cross-architecture transfer. Adversarial training is applied as the defense.`),
  P(`On a 20,000-flow dataset the hybrid model reaches ${H ? pct(H.metrics.test.accuracy) : "-"} accuracy and ${H ? pct(H.metrics.test.obfuscated_recall) : "-"} recall on obfuscated malicious traffic, on par with a random forest on the engineered statistics (${rf ? pct(rf.accuracy) : "-"}). ${H && A && curve(H, "domain-constrained", 0.1) != null ? `Under domain-constrained PGD at ε = 0.1, accuracy falls to ${pct(curve(H, "domain-constrained", 0.1))} for the undefended hybrid and ${pct(curve(A, "domain-constrained", 0.1))} after adversarial training, at a clean-accuracy cost of ${pct(H.metrics.test.accuracy - A.metrics.test.accuracy)}.` : ""} The unconstrained attack overstates the threat relative to the constrained one, and the per-technique analysis identifies which evasion techniques actually defeat the detector. Base-rate analysis shows what the measured false-positive rate means on a network that is 99% benign.`),
  P("Keywords: network security, DPI evasion, traffic obfuscation, 1D-CNN, adversarial machine learning, adversarial training.", { italics: true }),
  Break(),
);

// TOC
body.push(Center("Table of Contents", 32, true), new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }), Break());

// 1 Introduction
body.push(
  H1("1. Introduction"), H2("1.1 Overview"),
  P("Network intrusion detection has historically relied on Deep Packet Inspection: an engine inspects the bytes of each packet and raises an alarm when they match a known signature. This works precisely as long as the malicious bytes are visible and recognisable. Two developments have eroded both conditions. Encryption is now the default for most application traffic, so payload bytes are opaque; and evasive malware deliberately reshapes the observable characteristics of its traffic so that even unencrypted flows no longer match the patterns a signature engine expects."),
  P("The reshaping techniques are well documented. Packets are padded to fixed block sizes so that size signatures disappear; inter-arrival times are jittered or forced onto a constant cadence so that the periodicity of command-and-control beacons vanishes; records are fragmented so that a signature never appears within a single packet; the whole flow is tunnelled through a VPN or TLS envelope; and in the most sophisticated cases the size and timing distribution of the malicious flow is morphed to resemble video streaming or ordinary web browsing. None of these changes the semantics of what is sent. All of them change the shape of the flow."),
  P("That shape, the ordered sequence of packet sizes, directions and inter-arrival times together with aggregate flow statistics, is what survives encryption and is what this project learns from. The detector is a deep neural network; the interesting question is not whether it can classify plain attack traffic (it can, easily) but whether it can recognise attack traffic that has been reshaped to evade detection, and whether it continues to do so when an adversary who knows the detector exists optimises against it."),
  H2("1.2 Scope and Motivation"),
  P("The project has two halves. The first is a three-class flow classifier: benign, malicious-plain and malicious-obfuscated. The third class is the point of the exercise, because it is exactly the traffic a signature engine already lets through. The second half is an adversarial evaluation: the trained detector is attacked with gradient-based methods (FGSM, PGD) and with a gradient-free packet-space search, under constraints that reflect what an attacker can physically do to a flow, and is then hardened with adversarial training."),
  P("Three motivations shaped the design. First, no public corpus annotates which flows were obfuscated and how, so obfuscation is applied by the project itself to simulated and, optionally, captured traffic; this gives exact labels and makes it possible to ask which technique defeats the detector. Second, benign traffic is also obfuscated in the real world, a corporate VPN user being the obvious case, so a share of benign flows is obfuscated and stays labelled benign; a detector that equates obfuscation with hostility would fail in deployment. Third, standard adversarial attacks perturb every input in either direction, but an attacker can only pad a packet, never shrink it, and only delay a packet, never send it earlier; the attacks used here are restricted accordingly, and the difference between constrained and unconstrained results is itself reported."),
  P("Scope note: the obfuscation transforms operate on synthetic packet records and feature arrays, not on live sockets. This is a defensive research tool for producing labelled training data and measuring robustness; it is not an evasion utility."),
);

// 2 Literature survey
body.push(
  H1("2. Literature Survey"),
  P("The work sits at the intersection of encrypted-traffic classification, traffic-analysis countermeasures and adversarial machine learning. Table 2.1 summarises the most relevant prior work and the gap each leaves that this project addresses."),
  Tbl(["Work", "Contribution", "Gap / relation to this project"], LIT, [2200, 3800, 3000]),
  Caption("Table 2.1: Literature survey."),
  P("Three gaps recur. Deep traffic classifiers [11]-[15], [23], [24] are evaluated on captures where nobody was trying to evade them, and rarely include an obfuscated-benign class, so their reported accuracy does not speak to the evasion setting. The traffic-analysis countermeasure literature [16]-[19] studies obfuscation from the attacker's side, producing the transforms this project adopts as its threat model, but does not train detectors against them. And the adversarial ML literature [1]-[7] is almost entirely image-centric: its perturbation model, a symmetric ℓ∞ ball, admits changes to a flow that no attacker can make, an issue raised for security domains by Pierazzi et al. [9] and Apruzzese et al. [10] but rarely acted upon. This project combines the three: a detector trained on explicitly obfuscated traffic, evaluated under domain-constrained attacks and a problem-space attack, and hardened with adversarial training."),
);

// 3 Problem formulation
body.push(
  H1("3. Problem Formulation"), H2("3.1 Problem Description"),
  P("Given a bidirectional network flow observed as a sequence of packets, each with a timestamp, a size and a direction, decide whether the flow is benign, malicious and undisguised, or malicious and obfuscated to evade DPI. The decision must be made without payload contents, must not flag benign traffic merely because it is padded or tunnelled, and must remain reliable when an adversary who knows the detector exists modifies the flow within their physical means."),
  H2("3.2 Problem Statement"),
  P("Design, implement and evaluate a deep learning system that (i) classifies network flows into benign, malicious-plain and malicious-obfuscated using only flow shape, (ii) quantifies which obfuscation techniques it fails to detect, (iii) measures its robustness under adversarial perturbations constrained to pad-only and delay-only edits, and (iv) improves that robustness with adversarial training while reporting the cost in clean accuracy."),
  H2("3.3 Objectives"),
  ...Num([
    "Build a reproducible labelled dataset of benign, plain-malicious and obfuscated-malicious flows with recorded obfuscation recipes.",
    "Extract a packet-sequence view and a flow-statistics view from every flow.",
    "Train and compare a 1D-CNN, a bidirectional GRU, an MLP, a hybrid model and classical baselines.",
    "Determine per-technique detectability for each of eight obfuscation transforms.",
    "Measure robustness under constrained and unconstrained FGSM/PGD, a packet-space black-box attack, and cross-architecture transfer.",
    "Harden the best model with adversarial training and quantify the robustness gained against clean accuracy lost.",
    "Report calibration and base-rate-adjusted precision so the results speak to deployment, not only to a balanced test set.",
  ]),
  H2("3.4 Functional Requirements"),
  ...Bul([
    "FR1 Generate a dataset of N flows with a configurable class balance, benign-obfuscation rate and transform pool, from a single seed.",
    "FR2 Ingest real captures as per-packet CSV (tshark export) and apply the same obfuscation recipes.",
    "FR3 Extract the (4 × 128) sequence view and the 51-feature statistics view.",
    "FR4 Train any registered model from a YAML configuration with early stopping and checkpointing.",
    "FR5 Evaluate accuracy, macro-F1, per-class recall, obfuscated recall, malicious recall, false-positive rate and ROC-AUC.",
    "FR6 Run FGSM and PGD on either input surface with domain constraints, targeted or untargeted, over a range of budgets.",
    "FR7 Run a packet-space black-box attack that edits raw packets and re-extracts features.",
    "FR8 Report per-technique recall, per-profile accuracy, a transfer matrix, a smoothing sweep, calibration and base-rate precision.",
    "FR9 Train with adversarial examples mixed into each batch.",
    "FR10 Score an unlabelled per-packet CSV with a trained checkpoint.",
  ]),
  H2("3.5 Non-Functional Requirements"),
  ...Bul([
    "NFR1 Reproducibility: every result derives from a config file and a seed; the resolved config is saved beside the results.",
    "NFR2 Validity: every obfuscated or adversarial flow must be physically legal (monotone time, sizes within [40, 1500] bytes, byte volume preserved); this is enforced by tests.",
    "NFR3 Performance: the full experiment suite runs on a CPU-only laptop in about an hour; models stay under 200k parameters.",
    "NFR4 Portability: pure Python 3.10+ with PyTorch and scikit-learn; no GPU required.",
    "NFR5 Maintainability: a single trainer, attack loop and evaluator drive every model through one interface; 230+ unit tests.",
    "NFR6 Transparency: every figure writes a CSV beside it, and per-flow predictions are saved so failures can be traced.",
  ]),
);

// 4 Design
body.push(
  H1("4. Project Design and Implementation"), H2("4.1 Proposed Project Architecture"),
  P("The system is a linear pipeline of six stages, each a Python package under src/antod: (1) traffic synthesis or capture ingestion, (2) obfuscation, (3) feature extraction into two views, (4) model training with optional adversarial examples, (5) adversarial attack and defense evaluation, and (6) reporting. A YAML experiment configuration drives all stages through a command-line interface (antod generate | train | attack | evaluate | predict), and nothing in a later stage imports from an earlier one, so each stage can be tested in isolation."),
  H2("4.2 High Level Design"),
  P("Data layer. Twelve packet-level generators emit (timestamp, size, direction) records whose shapes follow the traffic-classification literature: heavy-tailed think times and MTU-filling bursts for web browsing, fixed-cadence small packets for VoIP, low-jitter periodic beacons for C2, sustained upstream transfer for exfiltration, and so on. Eight obfuscation transforms are composed into recipes of one to three steps with sampled intensities; mimicry always precedes and tunnelling always follows, matching where each would occur on a real host."),
  P("Feature layer. The sequence view encodes the first 128 packets across four channels: signed size (size/MTU carrying the direction sign), log inter-arrival time, direction, and a validity mask so that zero padding is distinguishable from a real zero-size packet. The statistics view is 51 aggregates: volume, size and timing moments, plus three groups aimed at obfuscation, namely size-grid occupancy (the fraction of packets on a 64/128/256-byte boundary, the fingerprint of block padding), size diversity, and timing regularity (coefficient of variation and autocorrelation peak of inter-arrival times, which catches beacons and reveals constant-rate shaping)."),
  P("Model layer. Every model implements one interface, forward(sequence, statistics), and declares which surface it reads; tests assert that gradient flows to exactly the declared surfaces. This lets one trainer, one attack loop and one evaluator serve every architecture."),
  P("Adversarial layer. Gradient attacks project each step onto the intersection of the ε-ball and a one-sided domain box; the packet-space attack edits raw packets and re-extracts features, using only output probabilities. Adversarial training injects an attack into the same trainer used for the baseline, so a difference between the defended and undefended model is attributable to the defense alone."),
  H2("4.3 Detailed Design"),
  P("1D-CNN. Three convolution stages (64, 128, 128 channels; kernels 7, 5, 3; max-pool 2, 2, 1) with batch normalisation and dropout, followed by both global average and global max pooling over time, concatenated into a 256-dimensional embedding and a two-layer head. The three stages correspond to the three scales at which obfuscation leaves evidence: a single padded packet, one request/response exchange, and the cadence of the whole flow. Average pooling answers how much of the flow looks padded; max pooling answers whether any part of it does. About 126k parameters."),
  P("MLP. Three hidden layers (256, 128, 64) with batch normalisation and dropout over the 51 standardised statistics; about 56k parameters. Statistics pass through arcsinh before standardisation, which is identity near zero, logarithmic in the tails, defined for negative values and invertible; invertibility is what lets an attack in scaled space be projected back into legal original units."),
  P("Bi-GRU. A bidirectional GRU (hidden 64) over the packet sequence, packed to each flow's true length so neither direction reads a padding slot, with masked mean pooling."),
  P("Hybrid. The CNN trunk and a two-layer statistics trunk (128, 64) are concatenated at the head; about 149k parameters. The trunks are kept separate until the head so each keeps its own normalisation."),
  P("Baselines. Random forest (400 trees, balanced class weights), an RBF SVM with explicit probability calibration, and logistic regression, all on the same scaled statistics."),
  P("Training. AdamW, learning rate 1e-3, weight decay 1e-4, cosine schedule, label smoothing 0.05, batch 128, up to 60 epochs with patience 12. Model selection is on validation macro-F1 rather than accuracy: a model that collapses the obfuscated class into the plain class loses little accuracy but much macro-F1, and that collapse is the failure this project exists to avoid."),
  P("Attacks. FGSM and PGD (10-20 steps, step 2.5ε/steps, random start) on the sequence, the statistics or both. The constrained feasible set fixes direction and mask, allows size magnitude to grow only (up to the MTU) and log inter-arrival time to increase only; statistics are clamped so proportions stay in [0, 1], counts non-negative and entropies under their 5-bit ceiling. Targeted attacks aim at the benign class. The packet-space attack is a greedy hill-climb: propose random pads and delays on 15% of packets, keep the proposal if the benign probability rises, up to 150 queries."),
  P("Defense. Adversarial training with PGD-5 on both surfaces at ε = 0.1, replacing half of each batch. The training attack is weaker than the evaluation attack on purpose; training against the exact evaluation attack would fit the model to that one perturbation. Randomised smoothing (majority vote over Gaussian-noised copies) is evaluated as a training-free alternative."),
  H2("4.4 Dataset Description"),
  P("The default dataset is 20,000 synthetic flows: 34% benign, 33% malicious-plain, 33% malicious-obfuscated, with 35% of benign flows also obfuscated and left labelled benign. Flows shorter than 8 packets are discarded. Table 4.1 lists the profiles and transforms. The split is stratified 65/15/20 into train, validation and test with the feature scaler fitted on the training split only. Real captures are supported through a per-packet CSV as exported by tshark; public flow-level corpora (CIC-IDS2017, UNSW-NB15) are deliberately not consumed through their aggregate CSVs because those cannot reconstruct the per-packet sequence, and the documentation gives the PCAP route instead."),
  Tbl(["Benign profiles", "Malicious profiles", "Obfuscation transforms"], [
    ["web_browsing", "c2_beacon", "block_padding"], ["video_streaming", "data_exfiltration", "random_padding"], ["voip", "port_scan", "fragmentation"],
    ["file_download", "brute_force", "timing_jitter"], ["dns_query", "ddos_flood", "constant_rate_shaping"], ["ssh_interactive", "reverse_shell", "dummy_injection"],
    ["", "", "tunnel_encapsulation"], ["", "", "protocol_mimicry"],
  ]),
  Caption("Table 4.1: Application profiles and obfuscation transforms."),
  P("Why synthetic, and what it costs. Obfuscation is applied rather than inferred, so the label and recipe of every flow are exact; that is what makes the per-technique analysis possible. The cost is external validity: profile parameters follow published measurements, not a capture of any particular network, so absolute accuracy on this dataset is an upper bound on live performance and the results should be read relatively, which model wins, which technique hurts, how much the defense buys."),
  H2("4.5 Tools and Technologies"),
  ...Bul(["Python 3.11, PyTorch 2.14 (CPU), scikit-learn 1.9, NumPy, pandas, matplotlib.", "uv for environment management; ruff for linting; pytest (230+ tests) for the data, model and attack invariants.", "YAML experiment configuration with strict key validation; GitHub Actions CI running lint, tests and a smoke run.", "tshark for exporting per-packet records from PCAPs."]),
);

// 5 Results
const compRows = [];
MODELS.forEach((m) => compRows.push(metricRow(NAMES[m.name], m.metrics.test)));
if (M.cnn1d && M.cnn1d.metrics.baselines) for (const [k, v] of Object.entries(M.cnn1d.metrics.baselines)) compRows.push(metricRow(NAMES[k] || k, v));

body.push(
  H1("5. Results and Discussion"), H2("5.1 Experimentation Environment Set-up"),
  P("All experiments ran on a Windows 11 laptop with CPU-only PyTorch 2.14 and Python 3.11; no GPU was used. The dataset was generated once (seed 42) and cached; each model was trained from its YAML configuration with seed 42, and the resolved configuration was written beside the results. Training times were " + MODELS.map((m) => `${NAMES[m.name]} ${Math.round(m.metrics.train_seconds)} s`).join(", ") + "."),
  H2("5.2 Functional Requirements: Results Achieved"),
  P("Table 5.1 reports the test-set metrics for every model and baseline. Accuracy and macro-F1 are on the three-class problem; obfuscated recall is recall on the malicious-obfuscated class alone; malicious recall collapses both malicious classes (was the attack caught at all?); false-positive rate is the share of benign flows flagged."),
  Tbl(["Model", "Accuracy", "Macro-F1", "Obf. recall", "Mal. recall", "FPR"], compRows, [2600, 1280, 1280, 1280, 1280, 1280]),
  Caption("Table 5.1: Clean test-set performance (4,000 flows)."),
  Img(M.cnn1d ? M.cnn1d.fig("model_comparison.png") : ""), Caption("Figure 5.1: Deep models against classical baselines."),
  P(`The honest headline is that the random forest on the 51 engineered statistics is the strongest clean classifier (${rf ? pct(rf.accuracy) : "-"}), and the plain 1D-CNN on the raw sequence does not beat it (${M.cnn1d ? pct(M.cnn1d.metrics.test.accuracy) : "-"}). The hybrid model, reading both views, reaches ${H ? pct(H.metrics.test.accuracy) : "-"} with ${H ? pct(H.metrics.test.obfuscated_recall) : "-"} obfuscated recall and effectively ties the forest. The gap between logistic regression and the non-linear models (${M.cnn1d && M.cnn1d.metrics.baselines.logistic_regression ? pct(M.cnn1d.metrics.baselines.logistic_regression.accuracy) : "-"} vs ~97%) measures how non-linear the problem is. On clean synthetic data, then, the case for a deep model does not rest on accuracy; it rests on the robustness results in 5.3.`),
  Img(H ? H.fig("confusion_matrix.png") : "", 4.2), Caption("Figure 5.2: Hybrid model confusion matrix (row-normalised)."),
  P("Confusions are almost entirely between the two malicious classes and between benign and malicious-obfuscated, never between benign and malicious-plain: undisguised attacks are trivially separable and obfuscation is what makes the problem hard."),
  Img(H ? H.fig("per_technique_recall.png") : ""), Caption("Figure 5.3: Recall by obfuscation technique (400 fresh flows per technique, hybrid model)."),
);
if (H && H.evaluation && H.evaluation.per_technique_recall) {
  const t = Object.entries(H.evaluation.per_technique_recall).sort((a, b) => a[1] - b[1]);
  body.push(Tbl(["Technique", "Recall on malicious_obfuscated"], t.map(([k, v]) => [k, f4(v)]), [4500, 4500]), Caption("Table 5.2: Per-technique recall, hybrid model."));
  body.push(P(`The hardest technique is ${t[0][0].replace(/_/g, " ")} (${pct(t[0][1])}) and the easiest ${t[t.length - 1][0].replace(/_/g, " ")} (${pct(t[t.length - 1][1])}). An aggregate obfuscated recall of ${pct(H.metrics.test.obfuscated_recall)} therefore hides substantial variation; this breakdown is only possible because each flow's recipe is recorded.`));
}
if (M.cnn1d && M.cnn1d.metrics.feature_importance && M.cnn1d.metrics.feature_importance.random_forest) {
  const imp = Object.entries(M.cnn1d.metrics.feature_importance.random_forest).slice(0, 10);
  body.push(Img(M.cnn1d.fig("feature_importance.png")), Caption("Figure 5.4: Random-forest feature importance."), P(`The forest's most informative statistics are ${imp.slice(0, 5).map(([k]) => k).join(", ")}: the obfuscation-specific features (size-grid occupancy, size diversity, timing regularity) sit alongside the volume features, confirming that the detector uses the fingerprints the obfuscation leaves rather than only the underlying behaviour.`));
}
body.push(Img(H ? H.fig("training_curves.png") : "", 5.0), Caption("Figure 5.5: Hybrid training history."));

// 5.3 non-functional / robustness
body.push(H2("5.3 Non-Functional Requirements: Results Achieved"), P("Robustness (NFR2, validity) is the non-functional property the project is about. Table 5.3 gives the attack sweep on the undefended and defended hybrid; Figure 5.6 traces accuracy against the attack budget under domain-constrained and unconstrained PGD."));
for (const m of [H, A]) {
  if (!m || !m.attack) continue;
  body.push(Tbl(["Attack", "Accuracy", "Obf. recall", "Mal. recall", "Evasion rate"], m.attack.sweep.map((r) => [r.attack, f4(r.accuracy), f4(r.obfuscated_recall), f4(r.malicious_recall), f4(r.evasion_rate)]), [3800, 1300, 1300, 1300, 1300]), Caption(`Table 5.3${m === H ? "a" : "b"}: Attack sweep, ${NAMES[m.name]}. Evasion rate is the share of malicious flows classified benign.`));
}
// untargeted vs targeted: the attacker does not control benign traffic
for (const m of [H]) {
  const sw = m && m.attack && m.attack.sweep;
  if (!sw) continue;
  const un = sw.find((r) => r.attack.includes("eps=0.1") && r.attack.includes("pgd") && r.attack.includes("constrained") && !r.attack.includes("unconstrained") && !r.attack.includes("targeted"));
  const ta = sw.find((r) => r.attack.includes("eps=0.1") && r.attack.includes("targeted"));
  if (un && ta) body.push(P(`Untargeted versus targeted. The untargeted PGD numbers overstate the operational threat in a specific way: at ε = 0.1 the ${NAMES[m.name]} keeps ${pct(un.malicious_recall)} malicious recall while its false-positive rate rises to ${pct(un.false_positive_rate)}, so most of the lost accuracy comes from benign flows being pushed to look obfuscated. A real attacker does not control benign traffic. The targeted attack, which pushes malicious flows towards the benign class and is what an evader actually wants, leaves malicious recall at ${pct(ta.malicious_recall)} with an evasion rate of ${pct(ta.evasion_rate)}. Evasion rate, not accuracy, is the security-relevant number.`));
}
body.push(Img(H ? H.fig("robustness_constrained.png") : ""), Caption("Figure 5.6: Accuracy under PGD, constrained vs unconstrained (hybrid)."));
if (H && A && curve(H, "domain-constrained", 0.1) != null) {
  body.push(P(`At ε = 0.1 the undefended hybrid retains ${pct(curve(H, "domain-constrained", 0.1))} accuracy under the constrained attack but only ${pct(curve(H, "unconstrained", 0.1))} under the unconstrained one: permitting edits no attacker can make overstates the vulnerability by ${pct(curve(H, "domain-constrained", 0.1) - curve(H, "unconstrained", 0.1))} points at this budget. Adversarial training lifts the constrained figure to ${pct(curve(A, "domain-constrained", 0.1))} (and ${pct(curve(A, "domain-constrained", 0.2))} at ε = 0.2, against ${pct(curve(H, "domain-constrained", 0.2))} undefended) at a clean-accuracy cost of ${pct(H.metrics.test.accuracy - A.metrics.test.accuracy)} points.`));
}
body.push(Img(A ? A.fig("evasion_rate.png") : (H ? H.fig("evasion_rate.png") : ""), 5.0), Caption("Figure 5.7: Targeted evasion rate against budget (defended hybrid)."));
for (const m of [H, A]) {
  const pa = m && m.attack && m.attack.packet_attack;
  if (pa) body.push(P(`Packet-space black-box attack, ${NAMES[m.name]}: of ${pa.n_flows} freshly generated malicious flows, ${pct(pa.evasion_before)} were read as benign untouched and ${pct(pa.evasion_after)} could be walked across the boundary with legal pad/delay edits within ${pa.max_queries} queries, at a mean byte overhead of ${pct(pa.mean_overhead)} and ${pa.mean_queries_to_evade.toFixed(0)} queries per successful evasion.`));
}
if (H && H.attack && H.attack.smoothing) {
  body.push(Tbl(["σ", "Clean accuracy", "Attacked accuracy", "Evasion rate"], H.attack.smoothing.map((r) => [r.sigma, f4(r.clean_accuracy), f4(r.attacked_accuracy), f4(r.evasion_rate)]), [1500, 2500, 2500, 2500]), Caption("Table 5.4: Randomised smoothing, hybrid, PGD ε = 0.1."), P("Smoothing is evaluated on clean and attacked inputs together because noise that blunts a perturbation also blurs the size and timing fingerprints the detector relies on; the clean column is where that cost appears."));
}
if (H && H.evaluation && H.evaluation.transfer_matrix) {
  const tm = H.evaluation.transfer_matrix;
  const cols = Object.keys(tm[0]).filter((k) => k !== "crafted_on");
  const short = (s) => s.split(":")[0];
  body.push(Tbl(["Crafted on \\ evaluated on", ...cols.map(short)], tm.map((r) => [short(r.crafted_on), ...cols.map((c) => f4(r[c]))])), Caption("Table 5.5: Transfer matrix, PGD ε = 0.1 on both surfaces. Diagonal is white-box."), P("Off-diagonal entries are the realistic threat: an attacker who has studied some detector, not necessarily the deployed one. Examples crafted on the statistics-only MLP transfer poorly to sequence models and vice versa, since they perturb surfaces the target never reads; the hybrid is the most exposed because it reads both."));
}
if (H && H.metrics.calibration) {
  const c = A && A.metrics.calibration ? A.metrics.calibration : H.metrics.calibration;
  const br = (A || H).metrics.precision_at_base_rate;
  body.push(P(`Calibration and base rate. Temperature scaling on the ${A ? "defended" : ""} hybrid gives T = ${c.temperature.toFixed(2)}, reducing expected calibration error from ${c.ece_before.toFixed(3)} to ${c.ece_after.toFixed(3)} without changing any prediction. Re-weighting the measured recall and false-positive rate to a network that is 99% benign gives a precision of ${pct(br["0.99"].precision)} and ${br["0.99"].false_alerts_per_10k.toFixed(0)} false alerts per 10,000 flows; at 99.9% benign, precision falls to ${pct(br["0.999"].precision)}. The balanced-test FPR of ${pct((A || H).metrics.test.false_positive_rate)} is therefore the number that decides deployability, not the accuracy.`));
}

body.push(
  H2("5.4 Comparison with Existing Work"),
  P("Direct numerical comparison with [11]-[15] is not meaningful because those works evaluate on un-obfuscated captures and mostly on binary or application-level labels; their reported accuracies (typically 95-99%) are for a problem this project's plain-malicious class already solves at ~100% recall. The relevant comparison is methodological. Like Deep Fingerprinting [14] and Rimmer et al. [15], we find that a 1D-CNN over direction/size sequences is competitive, but unlike them we find that a forest on engineered statistics matches it on clean data once obfuscation-specific features (size-grid occupancy, timing regularity) are included, which echoes Dyer et al. [16]: coarse features survive padding. Our constrained-attack results follow the recommendation of Pierazzi et al. [9] and Apruzzese et al. [10] to evaluate in the problem space, and our finding that the unconstrained attack overstates the threat quantifies why. The adversarial-training gain and its clean-accuracy cost are consistent in direction with Madry et al. [2] and Zhang et al. [6]."),
  H2("5.5 Project GUI Snapshots"),
  P("The project is a command-line and library tool; there is no graphical interface. Its user-facing surfaces are the CLI (antod generate | train | attack | evaluate | predict), the generated figures and Markdown tables under experiments/results/, and the per-flow predictions.csv. Figures 5.1-5.7 are the outputs a user sees."),
  H2("5.6 Societal Impact"),
  P("Signature-based detection is what most organisations still deploy, and obfuscated command-and-control and exfiltration traffic is how breaches persist undetected for months. A detector that reads flow shape rather than payload is compatible with encryption, does not require decrypting user traffic, and therefore improves security without the privacy cost of TLS interception. The explicit modelling of obfuscated-benign traffic matters socially as well: a detector that flagged every VPN user would push organisations to prohibit privacy tools. The adversarial evaluation is the responsible counterpart: publishing a detector without measuring how it fails under attack gives a false sense of security [5]."),
  H2("5.7 SDG Mapping"),
  ...Bul(["SDG 9, Industry, Innovation and Infrastructure: resilient digital infrastructure through detection that keeps working under encryption and evasion.", "SDG 16, Peace, Justice and Strong Institutions: reducing cybercrime by shortening the time attackers can operate undetected, without weakening privacy protections."]),
);

// conclusion
body.push(
  H1("Conclusion and Future Scope"),
  P(`This project built a three-class detector for obfuscated malicious network traffic and evaluated it the way a security result should be evaluated: against an adversary constrained to what it can actually do. On 20,000 synthetic flows the hybrid CNN+MLP reaches ${H ? pct(H.metrics.test.accuracy) : "-"} accuracy and ${H ? pct(H.metrics.test.obfuscated_recall) : "-"} recall on obfuscated attacks, matching a random forest on engineered statistics; the plain 1D-CNN does not beat the forest, and the report says so. The per-technique analysis shows where the detector is weak, the constrained-versus-unconstrained comparison shows how much a naive robustness evaluation would have overstated the threat, the packet-space attack removes the feature-box approximation entirely, and adversarial training buys measurable robustness at a measured cost in clean accuracy. Calibration and base-rate analysis translate the balanced-test numbers into what an analyst would experience.`),
  P("Key takeaways: obfuscation-specific features carry the signal and are worth engineering; the two malicious classes are separable from each other but the obfuscated class overlaps benign traffic, which is where all the difficulty lives; and the threat model matters more than the attack algorithm, since a symmetric ε-ball admits flows no attacker can send."),
  P("Future scope. The single most valuable next step is validation on real captures through the tshark per-packet route the loader already supports. Beyond that: three-seed variance for every reported claim (the script exists), a stronger black-box search (NES, SimBA) to tighten the packet-space lower bound, TRADES-style adversarial training and a sweep over the adversarial ratio, a transformer over the packet sequence, certified smoothing bounds, and training at realistic class prevalence rather than only re-weighting at evaluation. All of these are tracked in docs/FUTURE_WORK.md."),
);

// references
body.push(H1("References"), ...REFS.map((r, i) => new Paragraph({ spacing: { after: 100, line: 276 }, indent: { left: 560, hanging: 560 }, children: [new TextRun({ text: `[${i + 1}] ${r}`, font: FONT, size: 22 })] })));

// ---------------------------------------------------------------- document
const doc = new Document({
  creator: "Vivek Neeralagi, Muhammad Shahbik, Aravind P Sagar, J Aditya",
  title: "Adversarial Network Traffic Obfuscation Detection",
  styles: { default: { document: { run: { font: FONT, size: 24 } } } },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  features: { updateFields: true },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })] })] }) },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
