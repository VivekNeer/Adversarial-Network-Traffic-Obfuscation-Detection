/**
 * Build the book chapter (docs/ANTOD_book_chapter.docx) from the results.
 *
 * Follows docs/templates/Book_Chapter_Guidelines.docx: Title, Introduction,
 * Literature Review, Methodology, Results and Analysis, Discussion, Applications,
 * Conclusion, References. Numbers and figures are read from experiments/results/.
 *
 *   node scripts/build_chapter.js
 */
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, AlignmentType, LevelFormat, PageNumber, Footer } = require("docx");
const C = require("./lib/report_common");
const { MODELS, M, NAMES, pct, f4, rf, H, A, curve, FONT, P, H1, H2, Center, Bul, Caption, Tbl, Img, metricRow } = C;

const OUT = path.join(C.ROOT, "docs", "ANTOD_book_chapter.docx");

const REFS = [
  "Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. ICLR.",
  "Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning models resistant to adversarial attacks. ICLR.",
  "Carlini, N., & Wagner, D. (2017). Towards evaluating the robustness of neural networks. IEEE S&P, 39-57.",
  "Papernot, N., McDaniel, P., & Goodfellow, I. (2016). Transferability in machine learning. arXiv:1605.07277.",
  "Athalye, A., Carlini, N., & Wagner, D. (2018). Obfuscated gradients give a false sense of security. ICML.",
  "Cohen, J., Rosenfeld, E., & Kolter, Z. (2019). Certified adversarial robustness via randomized smoothing. ICML.",
  "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. ICML.",
  "Pierazzi, F., Pendlebury, F., Cortellazzi, J., & Cavallaro, L. (2020). Intriguing properties of adversarial ML attacks in the problem space. IEEE S&P.",
  "Apruzzese, G., Andreolini, M., Ferretti, L., Marchetti, M., & Colajanni, M. (2022). Modeling realistic adversarial attacks against network intrusion detection systems. Digital Threats: Research and Practice, 3(3).",
  "Wang, W., Zhu, M., Zeng, X., Ye, X., & Sheng, Y. (2017). Malware traffic classification using convolutional neural network for representation learning. ICOIN, 712-717.",
  "Lotfollahi, M., Jafari Siavoshani, M., Shirali Hossein Zade, R., & Saberian, M. (2020). Deep Packet: A novel approach for encrypted traffic classification using deep learning. Soft Computing, 24, 1999-2012.",
  "Aceto, G., Ciuonzo, D., Montieri, A., & Pescape, A. (2019). MIMETIC: Mobile encrypted traffic classification using multimodal deep learning. Computer Networks, 165.",
  "Sirinam, P., Imani, M., Juarez, M., & Wright, M. (2018). Deep Fingerprinting: Undermining website fingerprinting defenses with deep learning. ACM CCS, 1928-1943.",
  "Rimmer, V., Preuveneers, D., Juarez, M., Van Goethem, T., & Joosen, W. (2018). Automated website fingerprinting through deep learning. NDSS.",
  "Dyer, K. P., Coull, S. E., Ristenpart, T., & Shrimpton, T. (2012). Peek-a-Boo, I still see you: Why efficient traffic analysis countermeasures fail. IEEE S&P, 332-346.",
  "Wright, C. V., Coull, S. E., & Monrose, F. (2009). Traffic morphing: An efficient defense against statistical traffic analysis. NDSS.",
  "Nasr, M., Bahramali, A., & Houmansadr, A. (2021). Defeating DNN-based traffic analysis systems in real-time with blind adversarial perturbations. USENIX Security.",
  "Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. ICISSP, 108-116.",
  "Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems. MilCIS.",
  "Draper-Gil, G., Lashkari, A. H., Mamun, M. S. I., & Ghorbani, A. A. (2016). Characterization of encrypted and VPN traffic using time-related features. ICISSP, 407-414.",
  "Sommer, R., & Paxson, V. (2010). Outside the closed world: On using machine learning for network intrusion detection. IEEE S&P, 305-316.",
  "Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. NeurIPS.",
  "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR, 12, 2825-2830.",
];

const compRows = [];
MODELS.forEach((m) => compRows.push(metricRow(NAMES[m.name], m.metrics.test)));
if (M.cnn1d && M.cnn1d.metrics.baselines) for (const [k, v] of Object.entries(M.cnn1d.metrics.baselines)) compRows.push(metricRow(NAMES[k] || k, v));

const body = [];

// 1 Title
body.push(
  Center("Learning the Shape of a Flow: Detecting Obfuscated Malicious Network Traffic with Adversarially Trained Deep Networks", 32, true),
  Center("Vivek Neeralagi, Muhammad Shahbik, Aravind P Sagar, J Aditya", 24),
  Center("Department of Computer Science & Engineering, Sahyadri College of Engineering & Management, Mangaluru", 22),
  new Paragraph({ spacing: { before: 200 } }),
  P("Abstract. " + `Signature-based Deep Packet Inspection fails against malware that reshapes its traffic rather than its payload. This chapter presents a deep learning detector that classifies a network flow as benign, malicious-plain or malicious-obfuscated from the shape of the flow alone, and evaluates it under adversarial attacks constrained to what an attacker can physically do: pad a packet or delay it. A packet-level simulator with twelve application profiles and eight obfuscation transforms provides exactly labelled data; a 1D-CNN, a bidirectional GRU, an MLP, a hybrid and classical baselines are compared; robustness is measured with domain-constrained FGSM/PGD, a packet-space black-box attack and cross-architecture transfer; and adversarial training is applied as the defense. The hybrid model reaches ${H ? pct(H.metrics.test.accuracy) : "-"} accuracy and ${H ? pct(H.metrics.test.obfuscated_recall) : "-"} recall on obfuscated attacks, on par with a random forest on engineered statistics${H && A && curve(H, "domain-constrained", 0.1) != null ? `; adversarial training raises constrained-PGD accuracy at ε = 0.1 from ${pct(curve(H, "domain-constrained", 0.1))} to ${pct(curve(A, "domain-constrained", 0.1))} at a clean-accuracy cost of ${pct(H.metrics.test.accuracy - A.metrics.test.accuracy)} points` : ""}. Unconstrained attacks are shown to overstate the threat, and base-rate analysis translates the false-positive rate into what an analyst would experience on a 99%-benign network.`, { italics: true }),
  P("Keywords: network security, DPI evasion, traffic obfuscation, 1D-CNN, adversarial machine learning, adversarial training.", { italics: true }),
);

// 2 Introduction
body.push(
  H1("1. Introduction"),
  P("Deep Packet Inspection (DPI) detects malicious traffic by matching byte patterns inside packets, and it works only while those bytes are visible and recognisable. Neither condition holds any longer. Encryption is the default for application traffic, and evasive malware deliberately reshapes the observable characteristics of what it sends: packets are padded to fixed block sizes, inter-arrival times are jittered or flattened, records are fragmented, the whole flow is tunnelled, or its size and timing distribution is morphed to resemble video streaming. None of these changes the semantics of the traffic; all of them change its shape."),
  P("That shape, the ordered sequence of packet sizes, directions and inter-arrival times together with aggregate flow statistics, survives encryption and is what this work learns from. The question is not whether a neural network can classify undisguised attack traffic (it can, easily) but whether it can recognise attack traffic reshaped to evade detection, and whether it keeps doing so when an adversary who knows the detector exists optimises against it."),
  P("The problem was chosen because it sits where two literatures fail to meet. Deep traffic classifiers are evaluated on captures where nobody was trying to evade them; the traffic-analysis countermeasure literature studies obfuscation from the attacker's side; and adversarial machine learning is almost entirely image-centric, with a perturbation model that admits changes no attacker can make to a flow. Our objectives were to (i) build a labelled dataset in which obfuscation is applied rather than inferred, (ii) compare sequence, statistics and hybrid detectors, (iii) determine which obfuscation techniques defeat them, (iv) measure robustness under attacks restricted to pad-only and delay-only edits, and (v) harden the best model with adversarial training while reporting the cost."),
);

// 3 Literature review
body.push(
  H1("2. Literature Review"),
  P("Deep learning for traffic classification. Wang et al. [10] applied a 1D-CNN to raw packet bytes for malware traffic classification; Lotfollahi et al. [11] (Deep Packet) and Aceto et al. [12] (MIMETIC) extended the idea to encrypted and mobile traffic with convolutional and multimodal architectures. In website fingerprinting, Sirinam et al. [13] and Rimmer et al. [14] showed that a CNN over packet direction sequences defeats published defenses. These works establish that flow shape is learnable, but they evaluate on un-obfuscated captures, rarely include benign traffic that is itself obfuscated, and report no adversarial robustness."),
  P("Traffic obfuscation. Wright et al. [16] introduced traffic morphing, reshaping a flow's packet-size distribution to match a target; Dyer et al. [15] showed that coarse features such as total volume survive most efficient countermeasures, which is the observation behind our volume-preserving mimicry transform; Nasr et al. [17] demonstrated blind adversarial perturbations against DNN traffic analysers on live traffic, motivating our pad-only/delay-only constraint. Draper-Gil et al. [20] characterised VPN traffic by time features, evidence that obfuscated benign traffic exists at scale."),
  P("Adversarial machine learning. FGSM [1] and PGD [2] define the attacks; Carlini and Wagner [3] the evaluation discipline; Papernot et al. [4] transferability; Athalye et al. [5] the warning that gradient-masking defenses give false security, which is why we add a gradient-free attack; Cohen et al. [6] randomised smoothing; Guo et al. [7] temperature scaling. Pierazzi et al. [8] and Apruzzese et al. [9] argue that security-domain attacks must respect problem-space constraints, a recommendation this chapter follows. Sommer and Paxson [21] explain why machine-learning intrusion detection fails in deployment, chiefly through base rates, which motivates our base-rate analysis. Public datasets CIC-IDS2017 [18] and UNSW-NB15 [19] are flow-level and cannot supply the per-packet sequence a convolutional detector consumes."),
  P("The gap. No prior work trains a detector on explicitly obfuscated traffic with an obfuscated-benign class, evaluates it under attacks constrained to physically realisable edits, compares those to a problem-space attack, and reports the result at deployment base rates. This chapter does all four."),
);

// 4 Methodology
body.push(
  H1("3. Methodology"), H2("3.1 Data"),
  P("A packet-level simulator emits (timestamp, size, direction) records from six benign profiles (web browsing, video streaming, VoIP, file download, DNS, interactive SSH) and six malicious profiles (C2 beacon, data exfiltration, port scan, brute force, DDoS flood, reverse shell; the last deliberately overlaps SSH). Eight obfuscation transforms, block padding, random padding, fragmentation, timing jitter, constant-rate shaping, dummy injection, tunnel encapsulation and protocol mimicry, are composed into recipes of one to three steps with sampled intensities. Every transform preserves byte volume and packet validity. The dataset holds 20,000 flows: 34% benign, of which 35% are obfuscated and stay labelled benign; 33% malicious-plain; 33% malicious-obfuscated. Obfuscation is applied rather than inferred, so every label and recipe is exact. Real captures are supported through per-packet CSVs exported by tshark."),
  H2("3.2 Features"),
  P("Two views are extracted. The sequence view encodes the first 128 packets across four channels: signed size (size/MTU carrying the direction sign), log inter-arrival time, direction, and a validity mask. The statistics view is 51 aggregates: volume, size and timing moments, plus size-grid occupancy (the fraction of packets on a 64/128/256-byte boundary, the fingerprint of block padding), size diversity, and timing regularity (coefficient of variation and autocorrelation peak of inter-arrival times). Statistics pass through arcsinh before standardisation, which is invertible; invertibility lets an attack in scaled space be projected back into legal original units."),
  H2("3.3 Models"),
  P("A 1D-CNN with three convolution stages (64/128/128 channels, kernels 7/5/3) and both average and max pooling over time (126k parameters); a bidirectional GRU packed to true flow length; an MLP (256/128/64) over the statistics (56k); a hybrid concatenating the CNN trunk and a statistics trunk at the head (149k); and random forest, RBF SVM and logistic regression baselines on the same scaled statistics. Every model implements forward(sequence, statistics) and declares which surface it reads, so one trainer, one attack loop and one evaluator serve all of them. Training uses AdamW, cosine schedule, label smoothing 0.05, up to 60 epochs with early stopping, and selects on validation macro-F1 rather than accuracy, because collapsing the obfuscated class into the plain class costs little accuracy but much macro-F1."),
  H2("3.4 Attacks and Defense"),
  P("FGSM and PGD are projected at each step onto the intersection of the ε-ball and a one-sided domain box: size magnitude may only grow (to the MTU), log inter-arrival time may only increase, direction and validity mask are frozen, and padding positions are untouched. Statistics are clamped so that proportions stay in [0, 1], counts non-negative and entropies under their 5-bit ceiling. The unconstrained attack is retained for comparison. A packet-space black-box attack edits the raw packets with pad and delay operations, re-runs the feature extractor and queries only output probabilities, in a greedy hill-climb of up to 150 queries; every flow it produces could actually be sent. Adversarial training replaces half of each batch with PGD-5 examples (ε = 0.1, both surfaces), a weaker attack than evaluation uses on purpose. Randomised smoothing, temperature scaling and cross-architecture transfer complete the evaluation."),
  Tbl(["Channel", "Constrained perturbation"], [["signed size", "magnitude may only grow, up to the MTU"], ["log inter-arrival time", "may only increase (delay)"], ["direction", "frozen"], ["validity mask", "frozen"]], [3000, 6000]),
  Caption("Table 1: The pad-only / delay-only feasible set for sequence-space attacks."),
);

// 5 Results
body.push(
  H1("4. Results and Analysis"),
  Tbl(["Model", "Accuracy", "Macro-F1", "Obf. recall", "Mal. recall", "FPR"], compRows, [2600, 1280, 1280, 1280, 1280, 1280]),
  Caption("Table 2: Clean test-set performance on 4,000 flows."),
  Img(M.cnn1d ? M.cnn1d.fig("model_comparison.png") : "", 5.2), Caption("Figure 1: Deep models against classical baselines."),
  P(`Clean performance. The random forest on the engineered statistics is the strongest clean classifier (${rf ? pct(rf.accuracy) : "-"}); the plain 1D-CNN (${M.cnn1d ? pct(M.cnn1d.metrics.test.accuracy) : "-"}) does not beat it, and the hybrid (${H ? pct(H.metrics.test.accuracy) : "-"}, obfuscated recall ${H ? pct(H.metrics.test.obfuscated_recall) : "-"}) ties it. Confusions occur only between the two malicious classes and between benign and malicious-obfuscated; plain attacks are trivially separable.`),
  Img(H ? H.fig("per_technique_recall.png") : "", 5.2), Caption("Figure 2: Recall by obfuscation technique, hybrid model, 400 fresh flows per technique."),
);
if (H && H.evaluation && H.evaluation.per_technique_recall) {
  const t = Object.entries(H.evaluation.per_technique_recall).sort((a, b) => a[1] - b[1]);
  body.push(P(`Per-technique detectability. Recall ranges from ${pct(t[0][1])} for ${t[0][0].replace(/_/g, " ")} to ${pct(t[t.length - 1][1])} for ${t[t.length - 1][0].replace(/_/g, " ")}; the aggregate obfuscated recall hides this spread, and the breakdown is only possible because each flow's recipe was recorded.`));
}
for (const m of [H, A]) {
  if (!m || !m.attack) continue;
  body.push(Tbl(["Attack", "Accuracy", "Obf. recall", "Evasion rate"], m.attack.sweep.map((r) => [r.attack, f4(r.accuracy), f4(r.obfuscated_recall), f4(r.evasion_rate)]), [4500, 1500, 1500, 1500]), Caption(`Table ${m === H ? 3 : 4}: Attack sweep, ${NAMES[m.name]}.`));
}
// untargeted vs targeted: the attacker does not control benign traffic
for (const m of [H]) {
  const sw = m && m.attack && m.attack.sweep;
  if (!sw) continue;
  const un = sw.find((r) => r.attack.includes("eps=0.1") && r.attack.includes("pgd") && r.attack.includes("constrained") && !r.attack.includes("unconstrained") && !r.attack.includes("targeted"));
  const ta = sw.find((r) => r.attack.includes("eps=0.1") && r.attack.includes("targeted"));
  if (un && ta) body.push(P(`Untargeted versus targeted. The untargeted PGD numbers overstate the operational threat in a specific way: at ε = 0.1 the ${NAMES[m.name]} keeps ${pct(un.malicious_recall)} malicious recall while its false-positive rate rises to ${pct(un.false_positive_rate)}, so most of the lost accuracy comes from benign flows being pushed to look obfuscated. A real attacker does not control benign traffic. The targeted attack, which pushes malicious flows towards the benign class and is what an evader actually wants, leaves malicious recall at ${pct(ta.malicious_recall)} with an evasion rate of ${pct(ta.evasion_rate)}. Evasion rate, not accuracy, is the security-relevant number.`));
}
body.push(Img(H ? H.fig("robustness_constrained.png") : "", 5.2), Caption("Figure 3: Accuracy under PGD, constrained vs unconstrained, hybrid."));
if (H && A && curve(H, "domain-constrained", 0.1) != null) {
  body.push(P(`Robustness. At ε = 0.1 the undefended hybrid retains ${pct(curve(H, "domain-constrained", 0.1))} under the constrained attack and ${pct(curve(H, "unconstrained", 0.1))} under the unconstrained one, so ignoring physical validity overstates the vulnerability by ${pct(curve(H, "domain-constrained", 0.1) - curve(H, "unconstrained", 0.1))} points. Adversarial training lifts the constrained figure to ${pct(curve(A, "domain-constrained", 0.1))} at ε = 0.1 and ${pct(curve(A, "domain-constrained", 0.2))} at ε = 0.2 (undefended: ${pct(curve(H, "domain-constrained", 0.2))}), for a clean-accuracy cost of ${pct(H.metrics.test.accuracy - A.metrics.test.accuracy)} points.`));
}
for (const m of [H, A]) {
  const pa = m && m.attack && m.attack.packet_attack;
  if (pa) body.push(P(`Packet-space black-box attack, ${NAMES[m.name]}: of ${pa.n_flows} freshly generated malicious flows, ${pct(pa.evasion_before)} were read as benign untouched and ${pct(pa.evasion_after)} could be walked across the boundary with legal pad/delay edits within ${pa.max_queries} queries, at a mean byte overhead of ${pct(pa.mean_overhead)} and ${pa.mean_queries_to_evade.toFixed(0)} queries per successful evasion.`));
}
if (H && A && H.attack && A.attack && H.attack.packet_attack && A.attack.packet_attack) {
  const h = H.attack.packet_attack, a = A.attack.packet_attack;
  body.push(P(`Adversarial training does not transfer to this attack: the packet-space evasion rate is ${pct(h.evasion_after)} for the undefended hybrid and ${pct(a.evasion_after)} for the defended one, while the same defense cut gradient-PGD evasion by more than half. The training attack lives in feature space under an L-infinity budget; the packet-space search moves along a different path (a few large pads and delays on a subset of packets) that the defended model never saw. The two attacks measure different things, and a defense evaluated against only the attack it was trained on would have looked far better than it is. The attacker's price is the other half of the result: the successful evasions cost on average ${pct(a.mean_overhead)} extra bytes, so a detector that forces an exfiltration channel to nearly double its volume has still raised the attacker's cost substantially even where it is eventually evaded.`));
}
if (H && H.evaluation && H.evaluation.transfer_matrix) {
  const tm = H.evaluation.transfer_matrix; const cols = Object.keys(tm[0]).filter((k) => k !== "crafted_on"); const short = (s) => s.split(":")[0];
  body.push(Tbl(["Crafted on \\ evaluated on", ...cols.map(short)], tm.map((r) => [short(r.crafted_on), ...cols.map((c) => f4(r[c]))])), Caption("Table 5: Transfer matrix, PGD ε = 0.1. Diagonal is white-box."));
}
if ((A || H) && (A || H).metrics.precision_at_base_rate) {
  const m = A || H; const br = m.metrics.precision_at_base_rate; const c = m.metrics.calibration;
  body.push(P(`Calibration and base rate. Temperature scaling (T = ${c.temperature.toFixed(2)}) reduces expected calibration error from ${c.ece_before.toFixed(3)} to ${c.ece_after.toFixed(3)}. Re-weighted to a 99%-benign network, the ${NAMES[m.name]} achieves ${pct(br["0.99"].precision)} precision with ${br["0.99"].false_alerts_per_10k.toFixed(0)} false alerts per 10,000 flows; at 99.9% benign, precision falls to ${pct(br["0.999"].precision)}.`));
}

// 6 Discussion
body.push(
  H1("5. Discussion"),
  P("Interpretation. Three findings carry the chapter. First, on clean synthetic data the engineered statistics, once they include obfuscation-specific features, are as informative as the raw sequence: the forest matches the hybrid and beats the plain CNN. The convolution's advantage is not accuracy. Second, the threat model matters more than the attack algorithm: the same PGD loses a large fraction of its power when restricted to pad-only and delay-only edits, and the packet-space attack, which cannot cheat at all, is a lower bound that the gradient attacks should be read against. Third, adversarial training buys robustness at a clean-accuracy cost that is measurable and modest, and the defended model's false-positive rate, not its accuracy, is what decides whether it could be deployed."),
  P("Strengths. Exact labels with recorded recipes; an obfuscated-benign class; attacks that respect physics; a gradient-free attack as a check on gradient masking; a single training loop for defended and undefended models so the difference is the defense alone; calibration and base-rate reporting; full reproducibility from one configuration file and seed, with 230+ tests guarding the data and attack invariants."),
  P("Limitations. The data are synthetic, so absolute accuracy is an upper bound and conclusions should be read relatively. The statistics-space gradient attack perturbs aggregates that are not all jointly realisable; the packet-space attack is the corrective. The black-box search is greedy, so its evasion rate is a lower bound. Adversarial training was run against one attack configuration. Results are from a single seed; a three-seed variance script exists and should precede any claim that one model beats another. Training sees only the first 128 packets, although sliding-window inference covers longer flows."),
  P("Contribution. The chapter contributes a reproducible pipeline and an evaluation discipline for obfuscated-traffic detection: apply the obfuscation yourself, include obfuscated benign traffic, constrain the adversary to its physical means, check with a problem-space attack, and report at deployment base rates."),
);

// 7 Applications
body.push(
  H1("6. Applications"),
  ...Bul([
    "Encrypted-traffic intrusion detection at a network boundary, compatible with TLS and VPNs and without decrypting user traffic.",
    "Command-and-control beacon and exfiltration detection in enterprise SOCs, where the periodicity and asymmetry features carry the signal.",
    "Robustness auditing of any flow-based detector: the constrained attack suite and the packet-space attack can be pointed at a third-party model through the common forward(sequence, statistics) interface.",
    "Labelled-data generation for detector training where captures cannot be shared, since the simulator reproduces from a seed.",
  ]),
  P("Extensions: validation on real captures through the tshark per-packet route the loader already supports; stronger black-box search (NES, SimBA); TRADES-style training and a sweep over the adversarial ratio; a transformer over the packet sequence; certified smoothing bounds; and training at realistic class prevalence."),
);

// 8 Conclusion
body.push(
  H1("7. Conclusion"),
  P(`We built a three-class detector for obfuscated malicious network traffic from flow shape alone and evaluated it against an adversary constrained to what it can physically do. The hybrid CNN+MLP reaches ${H ? pct(H.metrics.test.accuracy) : "-"} accuracy and ${H ? pct(H.metrics.test.obfuscated_recall) : "-"} obfuscated recall, tying a random forest on engineered statistics; the per-technique analysis exposes where it is weak; the constrained-versus-unconstrained comparison quantifies how much a naive evaluation would have overstated the threat; adversarial training buys measurable robustness at a measured cost. The key takeaways are that obfuscation-specific features are worth engineering, that the obfuscated class overlaps benign traffic and that is where all the difficulty lives, and that in security the threat model matters more than the attack algorithm. Future work is validation on real captures, multi-seed variance, stronger black-box search and principled robust training objectives.`),
);

// 9 References
body.push(H1("References"), ...REFS.map((r, i) => new Paragraph({ spacing: { after: 100, line: 276 }, indent: { left: 560, hanging: 560 }, children: [new TextRun({ text: `[${i + 1}] ${r}`, font: FONT, size: 22 })] })));

const doc = new Document({
  creator: "Vivek Neeralagi, Muhammad Shahbik, Aravind P Sagar, J Aditya",
  title: "Learning the Shape of a Flow",
  styles: { default: { document: { run: { font: FONT, size: 24 } } } },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }, { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }] },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
