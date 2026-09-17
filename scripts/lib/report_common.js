/** Shared data loading and docx helpers for the report and book chapter. */
const fs = require("fs");
const path = require("path");
const {
  Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ImageRun, PageBreak, ShadingType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const RES = path.join(ROOT, "experiments", "results");

// ---------------------------------------------------------------- data
const readJson = (p) => (fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null);
const load = (name) => ({
  name,
  metrics: readJson(path.join(RES, name, "metrics.json")),
  attack: readJson(path.join(RES, name, "attack_results.json")),
  evaluation: readJson(path.join(RES, name, "evaluation.json")),
  fig: (f) => path.join(RES, name, "figures", f),
});
const MODELS = ["cnn1d", "mlp", "gru", "hybrid", "hybrid_advtrain"].map(load).filter((m) => m.metrics);
const M = Object.fromEntries(MODELS.map((m) => [m.name, m]));
const NAMES = { cnn1d: "1D-CNN", mlp: "MLP", gru: "Bi-GRU", hybrid: "Hybrid CNN+MLP", hybrid_advtrain: "Hybrid + adversarial training", random_forest: "Random forest", rbf_svm: "RBF SVM", logistic_regression: "Logistic regression" };
const pct = (x, d = 2) => (100 * x).toFixed(d) + "%";
const f4 = (x) => Number(x).toFixed(4);
const best = MODELS.filter((m) => m.name !== "hybrid_advtrain").reduce((a, b) => (b.metrics.test.accuracy > a.metrics.test.accuracy ? b : a), MODELS[0]);
const rf = M.cnn1d && M.cnn1d.metrics.baselines && M.cnn1d.metrics.baselines.random_forest;
const H = M.hybrid, A = M.hybrid_advtrain;
const curve = (m, key, eps) => { const c = m && m.attack && m.attack.robustness_curves && m.attack.robustness_curves[key]; if (!c) return null; const i = c.eps.indexOf(eps); return i >= 0 ? c.value[i] : null; };

// ---------------------------------------------------------------- helpers
const FONT = "Times New Roman";
const SP = { line: 360 };
const P = (text, o = {}) => new Paragraph({ spacing: { ...SP, after: 120 }, alignment: o.align || AlignmentType.JUSTIFIED, children: Array.isArray(text) ? text : [new TextRun({ text, font: FONT, size: 24, bold: o.bold, italics: o.italics })] });
const R = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || 24, bold: o.bold, italics: o.italics });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 240, ...SP }, children: [new TextRun({ text, font: FONT, size: 32, bold: true, color: "000000" })] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120, ...SP }, children: [new TextRun({ text, font: FONT, size: 28, bold: true, color: "000000" })] });
const Center = (text, size = 24, bold = false) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120, ...SP }, children: [new TextRun({ text, font: FONT, size, bold })] });
const Break = () => new Paragraph({ children: [new PageBreak()] });
const Bul = (items) => items.map((t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { ...SP, after: 60 }, children: [R(t)] }));
const Num = (items) => items.map((t) => new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { ...SP, after: 60 }, children: [R(t)] }));
const Caption = (text) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 240 }, children: [new TextRun({ text, font: FONT, size: 22, italics: true })] });

function Tbl(header, rows, widths) {
  const total = 9000;
  const w = widths || header.map(() => Math.floor(total / header.length));
  const cell = (t, i, head) => new TableCell({
    width: { size: w[i], type: WidthType.DXA },
    shading: head ? { type: ShadingType.CLEAR, fill: "E7E6E6", color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 80, right: 80 },
    children: [new Paragraph({ spacing: { line: 276 }, children: [new TextRun({ text: String(t), font: FONT, size: 20, bold: head })] })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: w,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }), ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))],
  });
}
function Img(p, widthIn = 5.5) {
  if (!fs.existsSync(p)) return P(`[figure missing: ${path.basename(p)} - run scripts/run_all.py]`, { italics: true });
  const png = fs.readFileSync(p);
  // read PNG dimensions to keep the aspect ratio
  const w = png.readUInt32BE(16), h = png.readUInt32BE(20);
  const width = Math.round(widthIn * 96), height = Math.round((widthIn * 96 * h) / w);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, children: [new ImageRun({ type: "png", data: png, transformation: { width, height } })] });
}
const metricRow = (name, t) => [name, f4(t.accuracy), f4(t.macro_f1), f4(t.obfuscated_recall), f4(t.malicious_recall), f4(t.false_positive_rate)];


module.exports = { ROOT, RES, MODELS, M, NAMES, pct, f4, best, rf, H, A, curve, FONT, SP, P, R, H1, H2, Center, Break, Bul, Num, Caption, Tbl, Img, metricRow };
