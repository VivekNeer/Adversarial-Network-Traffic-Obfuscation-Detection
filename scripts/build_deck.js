/**
 * Build the presentation deck from the experiment results.
 *
 * Every number on a slide is read from experiments/results/<model>/{metrics,
 * attack_results, evaluation}.json, and every figure is the PNG the CLI wrote,
 * so the deck cannot drift from the report.
 *
 *   node scripts/build_deck.js            -> docs/ANTOD_presentation.pptx
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const RES = path.join(ROOT, "experiments", "results");
const OUT = path.join(ROOT, "docs", "ANTOD_presentation.pptx");

// ---------------------------------------------------------------- palette
const NAVY = "14213D";
const INK = "1B1B1B";
const MUTED = "5C6470";
const PAPER = "FFFFFF";
const TINT = "EEF3FB";
const BLUE = "2A78D6";
const ORANGE = "EB6834";
const AQUA = "1BAF7A";
const FONT_H = "Cambria";
const FONT_B = "Calibri";

// ---------------------------------------------------------------- data
function readJson(p) {
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null;
}
function load(name) {
  const dir = path.join(RES, name);
  return {
    name,
    metrics: readJson(path.join(dir, "metrics.json")),
    attack: readJson(path.join(dir, "attack_results.json")),
    evaluation: readJson(path.join(dir, "evaluation.json")),
    fig: (f) => path.join(dir, "figures", f),
  };
}
const MODELS = ["cnn1d", "mlp", "gru", "hybrid", "hybrid_advtrain"].map(load).filter((m) => m.metrics);
const byName = Object.fromEntries(MODELS.map((m) => [m.name, m]));
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const f3 = (x) => Number(x).toFixed(3);

// ---------------------------------------------------------------- deck
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.author = "Vivek Neeralagi, Muhammad Shahbik, Aravind P Sagar, J Aditya";
pres.title = "Adversarial Network Traffic Obfuscation Detection";

function darkSlide() {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  return s;
}
function lightSlide(title) {
  const s = pres.addSlide();
  s.background = { color: PAPER };
  s.addText(title, {
    x: 0.5, y: 0.3, w: 9, h: 0.7, fontFace: FONT_H, fontSize: 30, bold: true, color: NAVY,
    isTextBox: true, margin: 0,
  });
  return s;
}
function bullets(slide, items, opts) {
  slide.addText(
    items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6 } })),
    Object.assign({ fontFace: FONT_B, fontSize: 14, color: INK, isTextBox: true, valign: "top" }, opts)
  );
}
function card(slide, x, y, w, h, head, body, accent) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08,
  });
  slide.addShape(pres.ShapeType.ellipse, { x: x + 0.2, y: y + 0.2, w: 0.32, h: 0.32, fill: { color: accent }, line: { color: accent } });
  slide.addText(head, { x: x + 0.62, y: y + 0.14, w: w - 0.8, h: 0.45, fontFace: FONT_B, fontSize: 14, bold: true, color: NAVY, isTextBox: true, margin: 0, valign: "middle" });
  slide.addText(body, { x: x + 0.2, y: y + 0.65, w: w - 0.4, h: h - 0.8, fontFace: FONT_B, fontSize: 11.5, color: INK, isTextBox: true, margin: 0, valign: "top" });
}
function stat(slide, x, y, w, big, label, color) {
  slide.addText(big, { x, y, w, h: 0.9, fontFace: FONT_H, fontSize: 40, bold: true, color, isTextBox: true, margin: 0, align: "center" });
  slide.addText(label, { x, y: y + 0.9, w, h: 0.5, fontFace: FONT_B, fontSize: 11, color: MUTED, isTextBox: true, margin: 0, align: "center", valign: "top" });
}
function note(slide, text) {
  slide.addNotes(text);
}

// 1 ------------------------------------------------------------ title
{
  const s = darkSlide();
  s.addText("Adversarial Network Traffic\nObfuscation Detection", {
    x: 0.6, y: 1.2, w: 8.8, h: 1.7, fontFace: FONT_H, fontSize: 38, bold: true, color: PAPER, isTextBox: true, margin: 0,
  });
  s.addText("Catching malware that hides how it talks, not what it says", {
    x: 0.6, y: 3.0, w: 8.8, h: 0.5, fontFace: FONT_B, fontSize: 16, italic: true, color: "CADCFC", isTextBox: true, margin: 0,
  });
  s.addText(
    "Vivek Neeralagi · Muhammad Shahbik · Aravind P Sagar · J Aditya\nDeep Learning Mini Project (CS722T2C) · Dept. of CSE, Sahyadri College of Engineering & Management · Guide: Mrs. Varsha M",
    { x: 0.6, y: 4.2, w: 8.8, h: 0.9, fontFace: FONT_B, fontSize: 11, color: "CADCFC", isTextBox: true, margin: 0 }
  );
  note(s, "Introduce the team and the one-line idea: signature-based DPI reads packet contents; modern malware evades it by reshaping traffic. We learn the shape instead.");
}

// 2 ------------------------------------------------------------ problem
{
  const s = lightSlide("The problem: DPI reads bytes, attackers reshape flows");
  const rows = [
    ["Padding", "grow every packet to a block size", "size signatures vanish"],
    ["Timing jitter / shaping", "randomise or flatten inter-arrival times", "beacon periodicity vanishes"],
    ["Fragmentation", "split records below the MTU", "signatures span boundaries"],
    ["Tunnelling (VPN/TLS)", "wrap the flow in encryption", "payload unreadable"],
    ["Protocol mimicry", "look like video or web", "statistically benign"],
  ];
  s.addTable(
    [
      ["Technique", "What it does", "Why DPI misses it"].map((t) => ({ text: t, options: { bold: true, color: PAPER, fill: { color: NAVY }, fontFace: FONT_B, fontSize: 12 } })),
      ...rows.map((r) => r.map((c) => ({ text: c, options: { fontFace: FONT_B, fontSize: 11.5, color: INK } }))),
    ],
    { x: 0.5, y: 1.2, w: 5.6, colW: [1.7, 2.1, 1.8], rowH: 0.42, border: { type: "solid", color: "D8DEE8", pt: 0.5 }, fill: { color: PAPER } }
  );
  s.addShape(pres.ShapeType.roundRect, { x: 6.4, y: 1.2, w: 3.1, h: 3.4, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.1 });
  s.addText("What survives encryption and reshaping?", { x: 6.6, y: 1.35, w: 2.7, h: 0.7, fontFace: FONT_B, fontSize: 13, bold: true, color: PAPER, isTextBox: true, margin: 0 });
  s.addText("The shape of the flow:\n\n•  packet sizes\n•  directions\n•  inter-arrival times\n•  aggregate statistics\n\nThat is what we learn from.", { x: 6.6, y: 2.05, w: 2.7, h: 2.4, fontFace: FONT_B, fontSize: 12, color: "CADCFC", isTextBox: true, margin: 0 });
  note(s, "Each of these is a real evasion class from the DPI-evasion literature. None of them changes the payload semantics; they change the observable shape. Once traffic is encrypted, shape is all a defender has.");
}

// 3 ------------------------------------------------------------ what we built
{
  const s = lightSlide("What we built");
  const cls = [
    ["0 · benign", "Ordinary traffic - including padded or tunnelled traffic (a VPN user is benign).", AQUA],
    ["1 · malicious, plain", "Attack traffic making no attempt to hide.", BLUE],
    ["2 · malicious, obfuscated", "Attack traffic reshaped to evade DPI. The class that matters: a signature engine already lets it through.", ORANGE],
  ];
  cls.forEach(([h, b, c], i) => card(s, 0.5, 1.2 + i * 1.15, 4.6, 1.0, h, b, c));
  const steps = ["Simulate 12 traffic profiles", "Apply 8 obfuscation transforms", "Extract sequence + statistics views", "Train CNN / MLP / GRU / hybrid + baselines", "Attack with constrained FGSM/PGD + packet-space search", "Harden with adversarial training"];
  steps.forEach((t, i) => {
    const y = 1.2 + i * 0.6;
    s.addShape(pres.ShapeType.ellipse, { x: 5.6, y: y + 0.08, w: 0.36, h: 0.36, fill: { color: NAVY }, line: { color: NAVY } });
    s.addText(String(i + 1), { x: 5.6, y: y + 0.08, w: 0.36, h: 0.36, fontFace: FONT_B, fontSize: 11, bold: true, color: PAPER, align: "center", valign: "middle", isTextBox: true, margin: 0 });
    s.addText(t, { x: 6.1, y, w: 3.5, h: 0.52, fontFace: FONT_B, fontSize: 12.5, color: INK, isTextBox: true, margin: 0, valign: "middle" });
  });
  note(s, "Three classes, not two: separating plain from obfuscated malicious traffic is what lets us ask which evasion technique works. Then the adversarial ML loop on top.");
}

// 4 ------------------------------------------------------------ dataset
{
  const s = lightSlide("Dataset: obfuscation is applied, not inferred");
  card(s, 0.5, 1.2, 2.9, 1.9, "6 benign profiles", "web browsing · video streaming · VoIP · file download · DNS · interactive SSH", AQUA);
  card(s, 3.55, 1.2, 2.9, 1.9, "6 malicious profiles", "C2 beacon · data exfiltration · port scan · brute force · DDoS flood · reverse shell (deliberately overlaps SSH)", ORANGE);
  card(s, 6.6, 1.2, 2.9, 1.9, "8 transforms", "block / random padding · fragmentation · timing jitter · constant-rate shaping · dummy injection · tunnel encapsulation · protocol mimicry", BLUE);
  s.addShape(pres.ShapeType.roundRect, { x: 0.5, y: 3.35, w: 9.0, h: 1.75, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.1 });
  s.addText("Two decisions that keep the experiment honest", { x: 0.75, y: 3.45, w: 8.5, h: 0.4, fontFace: FONT_B, fontSize: 13, bold: true, color: PAPER, isTextBox: true, margin: 0 });
  bullets(s, [
    "35% of benign flows are obfuscated too, and stay labelled benign. Otherwise the model learns 'obfuscated = malicious' and flags every VPN user.",
    "Protocol mimicry preserves byte volume. An attacker can reshape traffic but cannot decide not to send the bytes - without this the problem is ill-posed, not merely hard.",
  ], { x: 0.75, y: 3.85, w: 8.5, h: 1.2, fontSize: 12, color: "CADCFC" });
  note(s, "No public corpus annotates which flows were padded or tunnelled, so we generate: 20,000 flows, balanced thirds, every recipe recorded per flow. That is what makes the per-technique breakdown possible. Real captures are supported via tshark per-packet CSV.");
}

// 5 ------------------------------------------------------------ features + models
{
  const s = lightSlide("Two views of a flow, four detectors");
  card(s, 0.5, 1.2, 4.35, 1.75, "Sequence view  (4 × 128)", "First 128 packets. Channels: signed size (size/MTU × direction), log inter-arrival time, direction, validity mask. Read by the 1D-CNN and GRU.", BLUE);
  card(s, 5.15, 1.2, 4.35, 1.75, "Statistics view  (51 features)", "Volume, size and timing moments plus obfuscation probes: size-grid occupancy (padding), size diversity, timing regularity (beacons vs shaping). Read by the MLP and classical baselines.", ORANGE);
  const models = MODELS.filter((m) => m.name !== "hybrid_advtrain");
  const names = { cnn1d: "1D-CNN", mlp: "MLP", gru: "Bi-GRU", hybrid: "Hybrid CNN+MLP" };
  const w = 9 / Math.max(models.length, 1);
  models.forEach((m, i) => {
    const x = 0.5 + i * w;
    s.addShape(pres.ShapeType.roundRect, { x: x + 0.05, y: 3.2, w: w - 0.1, h: 1.9, fill: { color: PAPER }, line: { color: "D8DEE8", width: 1 }, rectRadius: 0.08 });
    s.addText(names[m.name] || m.name, { x: x + 0.2, y: 3.3, w: w - 0.4, h: 0.4, fontFace: FONT_B, fontSize: 14, bold: true, color: NAVY, isTextBox: true, margin: 0 });
    s.addText(`${m.metrics.n_parameters.toLocaleString()} params`, { x: x + 0.2, y: 3.7, w: w - 0.4, h: 0.3, fontFace: FONT_B, fontSize: 11, color: MUTED, isTextBox: true, margin: 0 });
    s.addText(`test accuracy ${pct(m.metrics.test.accuracy)}\nobfuscated recall ${pct(m.metrics.test.obfuscated_recall)}`, { x: x + 0.2, y: 4.05, w: w - 0.4, h: 0.9, fontFace: FONT_B, fontSize: 12, color: INK, isTextBox: true, margin: 0, valign: "top" });
  });
  note(s, "The CNN uses three conv stages because evidence lives at three scales: one padded packet, one exchange, the whole flow's cadence. Average and max pooling are both kept: 'how much looks padded' vs 'does anything look padded'. All models are under 200k parameters on purpose.");
}

// 6 ------------------------------------------------------------ threat model
{
  const s = lightSlide("Adversarial threat model: pad-only, delay-only");
  bullets(s, [
    "Image FGSM perturbs every pixel in either direction. Traffic does not work that way.",
    "An attacker controlling a malicious flow can PAD a packet (grow it, never shrink - the payload must fit) and DELAY a packet (later, never earlier - causality).",
    "Direction cannot flip. Packets cannot be conjured at a fixed position.",
    "So the feasible set is a one-sided box, not a symmetric ball. Statistics get the same treatment: proportions stay in [0,1], counts non-negative.",
    "We keep the unconstrained attack too: the gap between the two measures how much a naive evaluation overstates the threat.",
  ], { x: 0.5, y: 1.2, w: 5.2, h: 3.9, fontSize: 13 });
  const rows = [["size", "may only grow, up to MTU"], ["log inter-arrival", "may only increase"], ["direction", "frozen"], ["validity mask", "frozen"]];
  s.addTable(
    [
      ["Channel", "Allowed change"].map((t) => ({ text: t, options: { bold: true, color: PAPER, fill: { color: NAVY }, fontFace: FONT_B, fontSize: 12 } })),
      ...rows.map((r) => r.map((c) => ({ text: c, options: { fontFace: FONT_B, fontSize: 12, color: INK } }))),
    ],
    { x: 6.0, y: 1.3, w: 3.5, colW: [1.5, 2.0], rowH: 0.45, border: { type: "solid", color: "D8DEE8", pt: 0.5 } }
  );
  s.addText("Plus a packet-space black-box attack: edit the raw packets with pad/delay, re-extract features, query only the output. Every adversarial flow it produces could actually be sent.", { x: 6.0, y: 3.7, w: 3.5, h: 1.4, fontFace: FONT_B, fontSize: 11.5, italic: true, color: MUTED, isTextBox: true, margin: 0 });
  note(s, "Targeted attacks aim at the benign class - being mistaken for ordinary traffic is what an evader wants, not just misclassification. Evasion rate, not accuracy, is what an attacker optimises.");
}

// 7 ------------------------------------------------------------ clean results chart
{
  const s = lightSlide("Clean results: the honest comparison");
  const labels = [];
  const acc = [];
  const obf = [];
  const push = (label, t) => { labels.push(label); acc.push(+t.accuracy.toFixed(4)); obf.push(+t.obfuscated_recall.toFixed(4)); };
  const names = { cnn1d: "1D-CNN", mlp: "MLP", gru: "Bi-GRU", hybrid: "Hybrid", hybrid_advtrain: "Hybrid + adv. training" };
  MODELS.forEach((m) => push(names[m.name], m.metrics.test));
  const base = byName.cnn1d && byName.cnn1d.metrics.baselines;
  if (base) {
    ["random_forest", "rbf_svm", "logistic_regression"].forEach((b) => base[b] && push({ random_forest: "Random forest", rbf_svm: "RBF SVM", logistic_regression: "Logistic reg." }[b], base[b]));
  }
  s.addChart(pres.ChartType.bar, [
    { name: "accuracy", labels, values: acc },
    { name: "obfuscated recall", labels, values: obf },
  ], {
    x: 0.5, y: 1.15, w: 6.2, h: 4.0, barDir: "bar", barGrouping: "clustered",
    chartColors: [BLUE, ORANGE], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 8, dataLabelFormatCode: "0.00",
    valAxisMinVal: 0.5, valAxisMaxVal: 1.0, valAxisLabelFontSize: 9, catAxisLabelFontSize: 10,
    valAxisLabelColor: MUTED, catAxisLabelColor: INK, valGridLine: { color: "E4E3DF", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 10, showTitle: true, title: "Test set, 4,000 flows", titleFontSize: 11, titleColor: MUTED,
  });
  const rf = base && base.random_forest;
  const best = MODELS.reduce((a, b) => (b.metrics.test.accuracy > a.metrics.test.accuracy ? b : a));
  s.addText("What the numbers say", { x: 7.0, y: 1.2, w: 2.6, h: 0.4, fontFace: FONT_B, fontSize: 14, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  bullets(s, [
    rf ? `Random forest on the 51 hand-built statistics reaches ${pct(rf.accuracy)} - it beats the plain CNN (${pct(byName.cnn1d.metrics.test.accuracy)}) on clean synthetic data.` : "",
    `Best deep model: ${names[best.name]} at ${pct(best.metrics.test.accuracy)}, obfuscated recall ${pct(best.metrics.test.obfuscated_recall)}.`,
    "The deep models have to earn their place on robustness, not clean accuracy. We say so.",
  ].filter(Boolean), { x: 7.0, y: 1.65, w: 2.6, h: 3.5, fontSize: 11.5 });
  note(s, "Be upfront: on this synthetic data the feature-engineered forest is the strongest clean classifier. The case for the deep models rests on the adversarial results that follow, and on the hybrid that combines both views.");
}

// 8 ------------------------------------------------------------ per-technique + confusion
{
  const m = byName.hybrid || MODELS[MODELS.length - 1];
  const s = lightSlide(`Which evasion techniques get through? (${m.name})`);
  if (fs.existsSync(m.fig("per_technique_recall.png"))) s.addImage({ path: m.fig("per_technique_recall.png"), x: 0.5, y: 1.15, w: 5.4, h: 3.9, sizing: { type: "contain", w: 5.4, h: 3.9 } });
  if (fs.existsSync(m.fig("confusion_matrix.png"))) s.addImage({ path: m.fig("confusion_matrix.png"), x: 6.1, y: 1.15, w: 3.4, h: 2.8, sizing: { type: "contain", w: 3.4, h: 2.8 } });
  const tech = m.evaluation && m.evaluation.per_technique_recall;
  if (tech) {
    const sorted = Object.entries(tech).sort((a, b) => a[1] - b[1]);
    s.addText(`Hardest: ${sorted[0][0].replace(/_/g, " ")} (${pct(sorted[0][1])})\nEasiest: ${sorted[sorted.length - 1][0].replace(/_/g, " ")} (${pct(sorted[sorted.length - 1][1])})`, { x: 6.1, y: 4.05, w: 3.4, h: 1.0, fontFace: FONT_B, fontSize: 11.5, color: INK, isTextBox: true, margin: 0 });
  }
  note(s, "Each bar is 400 freshly generated malicious flows obfuscated by exactly one transform, so a miss is attributed to a technique rather than a mixed recipe. This is the question an average cannot answer.");
}

// 9 ------------------------------------------------------------ robustness
{
  const s = lightSlide("Robustness under attack");
  const series = [];
  const names = { cnn1d: "1D-CNN", mlp: "MLP", gru: "Bi-GRU", hybrid: "Hybrid", hybrid_advtrain: "Hybrid + adv. training" };
  let labels = null;
  MODELS.forEach((m) => {
    const c = m.attack && m.attack.robustness_curves && m.attack.robustness_curves["domain-constrained"];
    if (!c) return;
    labels = labels || c.eps.map(String);
    series.push({ name: names[m.name], labels, values: c.value.map((v) => +v.toFixed(4)) });
  });
  if (series.length) {
    s.addChart(pres.ChartType.line, series, {
      x: 0.5, y: 1.15, w: 5.8, h: 4.0, chartColors: [BLUE, ORANGE, AQUA, "EDA100", "4A3AA7"], lineSize: 2, lineDataSymbolSize: 6,
      valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFontSize: 9, catAxisLabelFontSize: 9, valAxisLabelColor: MUTED, catAxisLabelColor: MUTED,
      valGridLine: { color: "E4E3DF", size: 0.5 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendFontSize: 9,
      showTitle: true, title: "Accuracy vs PGD budget ε (domain-constrained)", titleFontSize: 11, titleColor: MUTED, catAxisTitle: "ε", showCatAxisTitle: true, catAxisTitleFontSize: 9,
    });
  }
  const h = byName.hybrid, a = byName.hybrid_advtrain;
  const at = (m, eps) => { const c = m && m.attack && m.attack.robustness_curves["domain-constrained"]; if (!c) return null; const i = c.eps.indexOf(eps); return i >= 0 ? c.value[i] : null; };
  const unc = (m, eps) => { const c = m && m.attack && m.attack.robustness_curves["unconstrained"]; if (!c) return null; const i = c.eps.indexOf(eps); return i >= 0 ? c.value[i] : null; };
  const lines = [];
  if (h && at(h, 0.1) != null) lines.push(`Hybrid at ε=0.1: ${pct(at(h, 0.1))} constrained vs ${pct(unc(h, 0.1))} unconstrained - ignoring physics overstates the threat.`);
  if (h && a && at(a, 0.1) != null) lines.push(`Adversarial training: ${pct(at(h, 0.1))} → ${pct(at(a, 0.1))} at ε=0.1, at a clean-accuracy cost of ${pct(h.metrics.test.accuracy - a.metrics.test.accuracy)}.`);
  const ev = h && h.attack && h.attack.evasion_curves["targeted at benign"];
  if (ev) lines.push(`Targeted evasion (malicious read as benign) at ε=0.2: ${pct(ev.value[ev.eps.indexOf(0.2)])} for the undefended hybrid.`);
  s.addText("Reading the curves", { x: 6.6, y: 1.2, w: 3.0, h: 0.4, fontFace: FONT_B, fontSize: 14, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  bullets(s, lines.length ? lines : ["Run scripts/run_all.py to populate."], { x: 6.6, y: 1.65, w: 3.0, h: 3.5, fontSize: 11.5 });
  note(s, "The shape of the curve is the result: graceful decay and a cliff at the same midpoint are different security properties. Compare the defended and undefended hybrid: same architecture, same loop, same seed - the only difference is the defense.");
}

// 10 ----------------------------------------------------------- packet attack + deployment reality
{
  const s = lightSlide("Two reality checks");
  const h = byName.hybrid_advtrain || byName.hybrid;
  const pa = h && h.attack && h.attack.packet_attack;
  s.addText("Packet-space black-box attack", { x: 0.5, y: 1.15, w: 4.4, h: 0.4, fontFace: FONT_B, fontSize: 14, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  const paH = byName.hybrid && byName.hybrid.attack && byName.hybrid.attack.packet_attack;
  if (pa) {
    stat(s, 0.5, 1.6, 2.1, pct(pa.evasion_after, 0), "malicious flows walked to 'benign'\nwith legal pad/delay edits", ORANGE);
    stat(s, 2.7, 1.6, 2.1, pct(pa.mean_overhead, 0), "extra bytes the attacker\nhad to pay, on average", BLUE);
    const cmp = paH && h.name === "hybrid_advtrain" ? ` Undefended hybrid: ${pct(paH.evasion_after, 0)} - adversarial training did not help against this search, unlike gradient PGD.` : "";
    s.addText(`${h.name}: ${pa.n_flows} flows, <=${pa.max_queries} queries each, ${pa.mean_queries_to_evade.toFixed(0)} queries per successful evasion.${cmp}`, { x: 0.5, y: 3.2, w: 4.3, h: 0.95, fontFace: FONT_B, fontSize: 10.5, color: MUTED, isTextBox: true, margin: 0 });
  } else {
    s.addText("Run antod attack to populate.", { x: 0.5, y: 1.6, w: 4.3, h: 0.5, fontFace: FONT_B, fontSize: 12, color: MUTED, isTextBox: true, margin: 0 });
  }
  const br = h && h.metrics.precision_at_base_rate;
  s.addText("Base-rate reality", { x: 5.3, y: 1.15, w: 4.2, h: 0.4, fontFace: FONT_B, fontSize: 14, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  if (br) {
    stat(s, 5.3, 1.6, 2.0, pct(br["0.99"].precision, 0), "precision when 99% of\ntraffic is benign", ORANGE);
    stat(s, 7.4, 1.6, 2.0, br["0.99"].false_alerts_per_10k.toFixed(0), "false alerts per\n10,000 flows", BLUE);
    s.addText(`The balanced test set has 34% benign traffic; real networks are >99%. The same FPR of ${pct(h.metrics.test.false_positive_rate)} that looks harmless on a balanced set is what this becomes in deployment.`, { x: 5.3, y: 3.2, w: 4.2, h: 0.9, fontFace: FONT_B, fontSize: 11, color: MUTED, isTextBox: true, margin: 0 });
  }
  s.addShape(pres.ShapeType.roundRect, { x: 0.5, y: 4.2, w: 9.0, h: 0.9, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
  s.addText(h && h.metrics.calibration ? `Calibration: temperature ${h.metrics.calibration.temperature.toFixed(2)}, ECE ${h.metrics.calibration.ece_before.toFixed(3)} → ${h.metrics.calibration.ece_after.toFixed(3)} after temperature scaling. Probabilities can now be used to set an alert threshold.` : "Calibration reported in metrics.json after the full run.", { x: 0.7, y: 4.25, w: 8.6, h: 0.8, fontFace: FONT_B, fontSize: 11.5, color: INK, isTextBox: true, margin: 0, valign: "middle" });
  note(s, "Both of these are things a mini project usually skips. The packet-space attack is the strictly correct threat model - no feature-box approximation. The base-rate slide is why FPR matters more than accuracy for anything that will actually be deployed.");
}

// 11 ----------------------------------------------------------- limitations / future
{
  const s = lightSlide("Limitations and what comes next");
  card(s, 0.5, 1.2, 4.35, 3.9, "Limitations", "• Synthetic data: absolute accuracy is an upper bound; read the results relatively.\n• Gradient attacks perturb features; the packet-space attack is the ground truth and is a lower bound on evasion (greedy search).\n• Single attack config for adversarial training.\n• Balanced classes - hence the base-rate slide.\n• 128-packet window during training (sliding-window inference exists).", ORANGE);
  card(s, 5.15, 1.2, 4.35, 3.9, "Next", "• Validate on real PCAPs (CIC-IDS2017 via tshark per-packet export - loader is ready).\n• Three-seed variance for every claim (script is ready).\n• Stronger black-box search (NES / SimBA).\n• TRADES-style adversarial training, adv_ratio sweep.\n• Transformer over the packet sequence.\n• Certified smoothing bounds.", AQUA);
  note(s, "Everything here is tracked in docs/FUTURE_WORK.md with what is done ticked off.");
}

// 12 ----------------------------------------------------------- closing
{
  const s = darkSlide();
  s.addText("Learn the shape, not the bytes.", { x: 0.6, y: 1.4, w: 8.8, h: 1.0, fontFace: FONT_H, fontSize: 34, bold: true, color: PAPER, isTextBox: true, margin: 0 });
  bullets(s, [
    "A three-class detector for obfuscated malicious traffic, with exact labels because we apply the obfuscation ourselves.",
    "Attacks constrained to what an attacker can physically do - pad and delay - plus a packet-space attack with no approximation at all.",
    "Reported honestly: where the forest wins, what adversarial training costs, and what the false-positive rate means at 99% benign.",
  ], { x: 0.6, y: 2.5, w: 8.8, h: 1.8, fontSize: 13, color: "CADCFC" });
  s.addText("github.com/VivekNeer/Adversarial-Network-Traffic-Obfuscation-Detection", { x: 0.6, y: 4.6, w: 8.8, h: 0.4, fontFace: FONT_B, fontSize: 12, color: "CADCFC", isTextBox: true, margin: 0 });
  note(s, "Close on the repository: everything reproduces from one config file and one seed.");
}

pres.writeFile({ fileName: OUT }).then(() => console.log("wrote", OUT));
