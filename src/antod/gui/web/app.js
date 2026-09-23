/**
 * ANTOD Web Dashboard Controller
 * Handles state, API fetching, tab switching, and dynamic rendering.
 */

// Application State
const state = {
  currentModelId: 'hybrid',
  models: [],
  modelDetails: {},
  comparison: null,
  inspector: {
    page: 1,
    limit: 50,
    filter: 'all',
    profile: 'all',
    search: '',
    total: 0,
    totalPages: 1,
  },
  runningTaskId: null,
  pollTimer: null,
};

// DOM Elements
const elements = {
  modelSelect: document.getElementById('modelSelect'),
  tabButtons: document.querySelectorAll('.tab-btn'),
  tabViews: document.querySelectorAll('.tab-view'),
  kpiGrid: document.getElementById('kpiGrid'),
  activeModelTag: document.getElementById('activeModelTag'),
  activeModelSummary: document.getElementById('activeModelSummary'),
  leaderboardBody: document.getElementById('leaderboardBody'),
  attackSweepBody: document.getElementById('attackSweepBody'),
  attackModelTag: document.getElementById('attackModelTag'),
  cmImage: document.getElementById('cmImage'),
  trainingCurvesImage: document.getElementById('trainingCurvesImage'),
  featureImportanceImage: document.getElementById('featureImportanceImage'),
  modelComparisonImage: document.getElementById('modelComparisonImage'),
  robustnessConstrainedImage: document.getElementById('robustnessConstrainedImage'),
  evasionRateImage: document.getElementById('evasionRateImage'),
  perTechniqueImage: document.getElementById('perTechniqueImage'),
  perProfileImage: document.getElementById('perProfileImage'),
  techniqueTableBody: document.getElementById('techniqueTableBody'),
  profileTableBody: document.getElementById('profileTableBody'),
  smoothingCardBody: document.getElementById('smoothingCardBody'),
  packetAttackCardBody: document.getElementById('packetAttackCardBody'),
  transferMatrixBody: document.getElementById('transferMatrixBody'),
  flowsTableBody: document.getElementById('flowsTableBody'),
  totalFlowsCount: document.getElementById('totalFlowsCount'),
  pageIndicator: document.getElementById('pageIndicator'),
  prevPageBtn: document.getElementById('prevPageBtn'),
  nextPageBtn: document.getElementById('nextPageBtn'),
  filterType: document.getElementById('filterType'),
  filterProfile: document.getElementById('filterProfile'),
  filterSearch: document.getElementById('filterSearch'),
  applyFiltersBtn: document.getElementById('applyFiltersBtn'),
  predictModel: document.getElementById('predictModel'),
  predictProfile: document.getElementById('predictProfile'),
  predictRecipe: document.getElementById('predictRecipe'),
  predictSeed: document.getElementById('predictSeed'),
  runPredictionBtn: document.getElementById('runPredictionBtn'),
  predictStatusBadge: document.getElementById('predictStatusBadge'),
  predictResultBody: document.getElementById('predictResultBody'),
  cliSubcommand: document.getElementById('cliSubcommand'),
  cliConfig: document.getElementById('cliConfig'),
  cliEpochs: document.getElementById('cliEpochs'),
  cliFlows: document.getElementById('cliFlows'),
  cliDevice: document.getElementById('cliDevice'),
  cliGeneratedCmd: document.getElementById('cliGeneratedCmd'),
  copyCmdBtn: document.getElementById('copyCmdBtn'),
  executeCmdBtn: document.getElementById('executeCmdBtn'),
  cliTerminalOutput: document.getElementById('cliTerminalOutput'),
  cliStatusBadge: document.getElementById('cliStatusBadge'),
};

// Utilities
const pct = (val, dec = 2) => `${(val * 100).toFixed(dec)}%`;
const num = (val, dec = 4) => (val !== undefined && val !== null ? Number(val).toFixed(dec) : '-');

// Initialize Dashboard
async function init() {
  setupTabs();
  setupEventListeners();
  await loadModels();
  await loadComparison();
  await switchModel(state.currentModelId);
  await loadFlows();
  updateCliCommand();
}

// Tab Switching
function setupTabs() {
  elements.tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      elements.tabButtons.forEach(b => b.classList.remove('active'));
      elements.tabViews.forEach(v => v.classList.remove('active'));

      btn.classList.add('active');
      const view = document.getElementById(`view-${targetTab}`);
      if (view) view.classList.add('active');
    });
  });
}

// Event Listeners
function setupEventListeners() {
  elements.modelSelect.addEventListener('change', async (e) => {
    await switchModel(e.target.value);
  });

  // Flow Inspector Filter Controls
  elements.applyFiltersBtn.addEventListener('click', () => {
    state.inspector.page = 1;
    state.inspector.filter = elements.filterType.value;
    state.inspector.profile = elements.filterProfile.value;
    state.inspector.search = elements.filterSearch.value.trim();
    loadFlows();
  });

  elements.prevPageBtn.addEventListener('click', () => {
    if (state.inspector.page > 1) {
      state.inspector.page--;
      loadFlows();
    }
  });

  elements.nextPageBtn.addEventListener('click', () => {
    if (state.inspector.page < state.inspector.totalPages) {
      state.inspector.page++;
      loadFlows();
    }
  });

  // Live Predictor Controls
  elements.runPredictionBtn.addEventListener('click', runLivePrediction);

  // CLI Studio Controls
  const cliInputs = [elements.cliSubcommand, elements.cliConfig, elements.cliEpochs, elements.cliFlows, elements.cliDevice];
  cliInputs.forEach(el => el.addEventListener('input', updateCliCommand));

  elements.copyCmdBtn.addEventListener('click', () => {
    const cmd = elements.cliGeneratedCmd.innerText;
    navigator.clipboard.writeText(cmd).then(() => {
      const orig = elements.copyCmdBtn.innerText;
      elements.copyCmdBtn.innerText = 'Copied to Clipboard!';
      setTimeout(() => elements.copyCmdBtn.innerText = orig, 1800);
    });
  });

  elements.executeCmdBtn.addEventListener('click', executeCliCommand);
}

// Load Models List
async function loadModels() {
  try {
    const res = await fetch('/api/models');
    state.models = await res.json();

    // Populate model selectors
    elements.modelSelect.innerHTML = '';
    elements.predictModel.innerHTML = '';
    state.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = `${m.name} (${pct(m.accuracy)} Acc)`;
      if (m.id === state.currentModelId) opt.selected = true;
      elements.modelSelect.appendChild(opt);

      const optPredict = opt.cloneNode(true);
      elements.predictModel.appendChild(optPredict);
    });
  } catch (err) {
    console.error('Failed to load models:', err);
  }
}

// Load Cross-Model Comparison Data
async function loadComparison() {
  try {
    const res = await fetch('/api/comparison');
    state.comparison = await res.json();
    renderLeaderboard();
  } catch (err) {
    console.error('Failed to load comparison data:', err);
  }
}

// Switch Active Model
async function switchModel(modelId) {
  state.currentModelId = modelId;
  elements.modelSelect.value = modelId;
  elements.predictModel.value = modelId;

  try {
    const res = await fetch(`/api/model/${modelId}`);
    const details = await res.json();
    state.modelDetails[modelId] = details;

    renderOverview();
    renderAdversarial();
    renderTechniques();
    updateFigures();
    loadFlows();
  } catch (err) {
    console.error(`Failed to load details for ${modelId}:`, err);
  }
}

// Render Overview View & KPIs
function renderOverview() {
  const details = state.modelDetails[state.currentModelId];
  if (!details) return;

  const m = details.metrics;
  const test = m.test || {};
  const calib = m.calibration || {};
  const br = m.precision_at_base_rate || {};
  const br99 = br['0.99'] || {};

  elements.activeModelTag.textContent = details.metrics.name || state.currentModelId;

  // KPI Grid
  elements.kpiGrid.innerHTML = `
    <div class="metric-card highlight-blue">
      <div class="metric-name">Balanced Accuracy</div>
      <div class="metric-stat">${pct(test.accuracy || 0)}</div>
      <div class="metric-detail">3-Class Test Split (4,001 flows)</div>
    </div>
    <div class="metric-card highlight-amber">
      <div class="metric-name">Obfuscated Recall</div>
      <div class="metric-stat">${pct(test.obfuscated_recall || 0)}</div>
      <div class="metric-detail">DPI-Evading Attacks Detected</div>
    </div>
    <div class="metric-card highlight-emerald">
      <div class="metric-name">Malicious Recall</div>
      <div class="metric-stat">${pct(test.malicious_recall || 0)}</div>
      <div class="metric-detail">Plain + Obfuscated Traffic</div>
    </div>
    <div class="metric-card highlight-rose">
      <div class="metric-name">Base-Rate Precision (99%)</div>
      <div class="metric-stat">${pct(br99.precision || 0)}</div>
      <div class="metric-detail">${(br99.false_alerts_per_10k || 0).toFixed(1)} alerts / 10k real flows</div>
    </div>
    <div class="metric-card">
      <div class="metric-name">False Positive Rate</div>
      <div class="metric-stat">${pct(test.false_positive_rate || 0)}</div>
      <div class="metric-detail">Benign Traffic Flagged</div>
    </div>
    <div class="metric-card">
      <div class="metric-name">Expected Calibration Error</div>
      <div class="metric-stat">${num(calib.ece_after, 4)}</div>
      <div class="metric-detail">Temperature Scaling T = ${num(calib.temperature, 2)}</div>
    </div>
  `;

  // Active Model Summary Box
  elements.activeModelSummary.innerHTML = `
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.8rem;">
      <div><span style="color: var(--text-tertiary)">Architecture:</span> <strong style="font-family: var(--font-mono); color: var(--text-primary)">${m.model}</strong></div>
      <div><span style="color: var(--text-tertiary)">Defense Mode:</span> <strong style="font-family: var(--font-mono); color: var(--text-primary)">${m.defense}</strong></div>
      <div><span style="color: var(--text-tertiary)">Parameters:</span> <strong style="font-family: var(--font-mono); color: var(--text-primary)">${(m.n_parameters || 0).toLocaleString()}</strong></div>
      <div><span style="color: var(--text-tertiary)">Training Time:</span> <strong style="font-family: var(--font-mono); color: var(--text-primary)">${num(m.train_seconds, 1)}s (Epoch ${m.best_epoch})</strong></div>
      <div><span style="color: var(--text-tertiary)">Macro F1:</span> <strong style="font-family: var(--font-mono); color: var(--text-primary)">${pct(test.macro_f1 || 0)}</strong></div>
      <div><span style="color: var(--text-tertiary)">ROC-AUC Macro:</span> <strong style="font-family: var(--font-mono); color: var(--text-primary)">${num(test.roc_auc_macro, 4)}</strong></div>
    </div>
  `;
}

// Render Leaderboard & Baselines
function renderLeaderboard() {
  if (!state.comparison || !state.comparison.models) return;

  const rows = state.comparison.models;
  elements.leaderboardBody.innerHTML = rows.map(r => {
    const isBaseline = !!r.is_baseline;
    const isSelected = r.id === state.currentModelId;
    const typeBadge = isBaseline
      ? `<span class="badge grey">Classical</span>`
      : r.defense === 'adversarial_training'
      ? `<span class="badge cyan">Adv-Trained</span>`
      : `<span class="badge blue">Deep Model</span>`;

    return `
      <tr class="${isSelected ? 'highlight' : ''}">
        <td><strong>${r.name}</strong> ${isSelected ? '<span class="badge green">Active</span>' : ''}</td>
        <td>${typeBadge}</td>
        <td>${pct(r.accuracy)}</td>
        <td>${pct(r.macro_f1)}</td>
        <td><strong style="color: #fbbf24">${pct(r.obfuscated_recall)}</strong></td>
        <td><strong style="color: #34d399">${pct(r.malicious_recall)}</strong></td>
        <td>${pct(r.false_positive_rate)}</td>
        <td><span class="text-muted">${r.n_parameters ? r.n_parameters.toLocaleString() + ' params' : 'Scikit-Learn'}</span></td>
      </tr>
    `;
  }).join('');
}

// Render Adversarial Robustness Tab
function renderAdversarial() {
  const details = state.modelDetails[state.currentModelId];
  if (!details) return;

  const attacks = details.attacks || {};
  const sweep = attacks.sweep || [];
  elements.attackModelTag.textContent = state.currentModelId;

  elements.attackSweepBody.innerHTML = sweep.map(s => {
    const isClean = s.attack === 'clean';
    return `
      <tr class="${isClean ? 'highlight' : ''}">
        <td><strong>${s.attack}</strong> ${isClean ? '<span class="badge green">Baseline</span>' : ''}</td>
        <td>${pct(s.accuracy)}</td>
        <td>${pct(s.macro_f1)}</td>
        <td>${pct(s.obfuscated_recall)}</td>
        <td>${pct(s.malicious_recall)}</td>
        <td><strong style="color: ${s.evasion_rate > 0.4 ? '#f87171' : '#34d399'}">${pct(s.evasion_rate)}</strong></td>
        <td>${num(s.stats_linf, 3)}</td>
        <td>${num(s.seq_linf, 3)}</td>
      </tr>
    `;
  }).join('');

  // Smoothing Defense
  const smoothing = attacks.smoothing || [];
  if (smoothing.length > 0) {
    elements.smoothingCardBody.innerHTML = `
      <p class="text-muted mb-2">Gaussian noise injection (σ) provides empirical smoothing against L2/Linf feature perturbations.</p>
      <div class="table-responsive">
        <table class="data-table">
          <thead><tr><th>Noise (σ)</th><th>Clean Acc</th><th>Robust Acc</th><th>Evasion Rate</th></tr></thead>
          <tbody>
            ${smoothing.map(sm => `
              <tr>
                <td><strong>σ = ${sm.sigma}</strong></td>
                <td>${pct(sm.clean_accuracy)}</td>
                <td>${pct(sm.robust_accuracy)}</td>
                <td>${pct(sm.evasion_rate)}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;
  } else {
    elements.smoothingCardBody.innerHTML = `<p class="text-muted">No smoothing sweep data recorded for this run.</p>`;
  }

  // Packet-Space Attack
  const pa = attacks.packet_attack;
  if (pa) {
    elements.packetAttackCardBody.innerHTML = `
      <p class="text-muted mb-2">Black-box heuristic attack operating strictly on physical packet arrays (padding & delays).</p>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px;">
        <div><span class="text-muted">Evasion Before Attack:</span> <strong>${pct(pa.evasion_before)}</strong></div>
        <div><span class="text-muted">Evasion After Attack:</span> <strong style="color: #fbbf24">${pct(pa.evasion_after)}</strong></div>
        <div><span class="text-muted">Mean Bandwidth Overhead:</span> <strong>${pct(pa.mean_overhead)}</strong></div>
        <div><span class="text-muted">Mean Queries to Evade:</span> <strong>${(pa.mean_queries_to_evade || 0).toFixed(0)}</strong></div>
      </div>
    `;
  } else {
    elements.packetAttackCardBody.innerHTML = `<p class="text-muted">No packet-space attack data for this run.</p>`;
  }
}

// Render Techniques & Profiles
function renderTechniques() {
  const details = state.modelDetails[state.currentModelId];
  if (!details) return;

  const ev = details.evaluation || {};
  const tech = ev.per_technique_recall || {};
  const prof = ev.per_profile_accuracy || {};

  elements.techniqueTableBody.innerHTML = Object.entries(tech)
    .sort((a, b) => a[1] - b[1])
    .map(([name, recall]) => `
      <tr>
        <td><code>${name}</code></td>
        <td><strong style="color: ${recall < 0.85 ? '#f87171' : '#34d399'}">${pct(recall)}</strong></td>
      </tr>
    `).join('');

  elements.profileTableBody.innerHTML = Object.entries(prof)
    .sort((a, b) => a[1] - b[1])
    .map(([name, acc]) => `
      <tr>
        <td><strong>${name}</strong></td>
        <td>${pct(acc)}</td>
      </tr>
    `).join('');

  // Transfer Matrix
  const tm = ev.transfer_matrix || [];
  if (tm.length > 0) {
    const cols = Object.keys(tm[0]).filter(k => k !== 'crafted_on');
    elements.transferMatrixBody.innerHTML = `
      <p class="text-muted mb-2">Attacks crafted on rows, evaluated on columns (lower accuracy = higher adversarial transferability):</p>
      <div class="table-responsive">
        <table class="data-table">
          <thead>
            <tr><th>Crafted On \\ Target</th>${cols.map(c => `<th>${c}</th>`).join('')}</tr>
          </thead>
          <tbody>
            ${tm.map(r => `
              <tr>
                <td><strong>${r.crafted_on}</strong></td>
                ${cols.map(c => `<td>${pct(r[c])}</td>`).join('')}
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;
  } else {
    elements.transferMatrixBody.innerHTML = `<p class="text-muted">Transfer matrix is computed when evaluating alongside sibling checkpoints.</p>`;
  }
}

// Update Image Figures
function updateFigures() {
  const m = state.currentModelId;
  elements.cmImage.src = `/results/${m}/figures/confusion_matrix.png`;
  elements.trainingCurvesImage.src = `/results/${m}/figures/training_curves.png`;
  elements.featureImportanceImage.src = `/results/${m}/figures/feature_importance.png`;
  elements.modelComparisonImage.src = `/results/${m}/figures/model_comparison.png`;
  elements.robustnessConstrainedImage.src = `/results/${m}/figures/robustness_constrained.png`;
  elements.evasionRateImage.src = `/results/${m}/figures/evasion_rate.png`;
  elements.perTechniqueImage.src = `/results/${m}/figures/per_technique_recall.png`;
  elements.perProfileImage.src = `/results/${m}/figures/per_profile_accuracy.png`;
}

// Load Flow Inspector Rows
async function loadFlows() {
  const { page, limit, filter, profile, search } = state.inspector;
  const m = state.currentModelId;

  try {
    const query = new URLSearchParams({ page, limit, filter, profile, search }).toString();
    const res = await fetch(`/api/predictions/${m}?${query}`);
    const data = await res.json();

    state.inspector.total = data.total;
    state.inspector.totalPages = data.total_pages;

    elements.totalFlowsCount.textContent = data.total.toLocaleString();
    elements.pageIndicator.textContent = `Page ${data.page} of ${data.total_pages}`;
    elements.prevPageBtn.disabled = data.page <= 1;
    elements.nextPageBtn.disabled = data.page >= data.total_pages;

    elements.flowsTableBody.innerHTML = data.rows.map(r => {
      const bPct = Math.round(r.benign * 100);
      const pPct = Math.round(r.malicious_plain * 100);
      const oPct = Math.round(r.malicious_obfuscated * 100);

      const statusBadge = r.correct
        ? `<span class="tag tag-green">VALID</span>`
        : `<span class="tag tag-rose">MISMATCH</span>`;

      return `
        <tr>
          <td style="font-family: var(--font-mono)">#${r.index}</td>
          <td><strong style="color: var(--text-primary)">${r.profile}</strong></td>
          <td><code style="color: var(--text-code); font-size: 0.74rem;">${r.recipe}</code></td>
          <td><span class="tag tag-slate">${r.y_true}</span></td>
          <td><span class="tag ${r.correct ? 'tag-green' : 'tag-rose'}">${r.y_pred}</span></td>
          <td>${statusBadge}</td>
          <td>
            <div class="meter-row">
              <div class="meter-track">
                <div class="meter-fill benign" style="width: ${bPct}%" title="Benign: ${bPct}%"></div>
                <div class="meter-fill plain" style="width: ${pPct}%" title="Plain: ${pPct}%"></div>
                <div class="meter-fill obfuscated" style="width: ${oPct}%" title="Obfuscated: ${oPct}%"></div>
              </div>
              <span style="color: var(--text-tertiary)">${bPct}% / ${pPct}% / ${oPct}%</span>
            </div>
          </td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    console.error('Failed to load flows:', err);
  }
}

// Live Predictor Handler
async function runLivePrediction() {
  const payload = {
    model: elements.predictModel.value,
    profile: elements.predictProfile.value,
    recipe: elements.predictRecipe.value,
    seed: parseInt(elements.predictSeed.value) || 42,
  };

  elements.runPredictionBtn.disabled = true;
  elements.predictStatusBadge.textContent = 'Synthesizing & Classifying...';
  elements.predictStatusBadge.className = 'tag tag-amber';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (data.error) {
      elements.predictResultBody.innerHTML = `<div style="padding: 24px; text-align: center; color: #fb7185">Inference Error: ${data.error}</div>`;
      elements.predictStatusBadge.textContent = 'Failed';
      elements.predictStatusBadge.className = 'tag tag-rose';
      return;
    }

    const p = data.probabilities;
    const s = data.flow_summary;
    const isCorrect = data.predicted_class === data.true_class;

    elements.predictStatusBadge.textContent = isCorrect ? 'CLASSIFIED (CONGRUENT)' : 'EVASION DETECTED (MISMATCH)';
    elements.predictStatusBadge.className = isCorrect ? 'tag tag-green' : 'tag tag-rose';

    // Professional SVG icons
    const iconShield = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>`;
    const iconAlert = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`;
    const iconWarning = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;

    const statusIcon = data.predicted_class === 'benign' ? iconShield : data.predicted_class === 'malicious_plain' ? iconAlert : iconWarning;

    elements.predictResultBody.innerHTML = `
      <div class="verdict-box">
        <div class="verdict-top">
          <div class="verdict-indicator ${data.predicted_class}">
            ${statusIcon}
          </div>
          <div class="verdict-head-text">
            <h4>${data.predicted_class.toUpperCase().replace('_', ' ')}</h4>
            <div style="font-size: 0.78rem; color: var(--text-tertiary)">Ground Truth Label: <strong style="color: var(--text-primary)">${data.true_class}</strong></div>
          </div>
        </div>

        <div style="display: flex; flex-direction: column; gap: 8px;">
          <div>
            <div style="display: flex; justify-content: space-between; font-size: 0.76rem; margin-bottom: 3px;">
              <span style="color: var(--text-secondary)">Benign Probability</span>
              <strong style="font-family: var(--font-mono)">${pct(p.benign)}</strong>
            </div>
            <div style="height: 5px; background: #1e293b; border-radius: 2px; overflow: hidden;">
              <div style="height: 100%; width: ${p.benign * 100}%; background: var(--color-benign);"></div>
            </div>
          </div>

          <div>
            <div style="display: flex; justify-content: space-between; font-size: 0.76rem; margin-bottom: 3px;">
              <span style="color: var(--text-secondary)">Malicious Plain Probability</span>
              <strong style="font-family: var(--font-mono)">${pct(p.malicious_plain)}</strong>
            </div>
            <div style="height: 5px; background: #1e293b; border-radius: 2px; overflow: hidden;">
              <div style="height: 100%; width: ${p.malicious_plain * 100}%; background: var(--color-plain);"></div>
            </div>
          </div>

          <div>
            <div style="display: flex; justify-content: space-between; font-size: 0.76rem; margin-bottom: 3px;">
              <span style="color: var(--text-secondary)">Malicious Obfuscated Probability</span>
              <strong style="font-family: var(--font-mono)">${pct(p.malicious_obfuscated)}</strong>
            </div>
            <div style="height: 5px; background: #1e293b; border-radius: 2px; overflow: hidden;">
              <div style="height: 100%; width: ${p.malicious_obfuscated * 100}%; background: var(--color-obfuscated);"></div>
            </div>
          </div>
        </div>

        <div class="verdict-meta-grid">
          <div><span style="color: var(--text-tertiary)">Packets:</span> <strong style="font-family: var(--font-mono)">${s.n_packets}</strong></div>
          <div><span style="color: var(--text-tertiary)">Duration:</span> <strong style="font-family: var(--font-mono)">${s.duration_seconds}s</strong></div>
          <div><span style="color: var(--text-tertiary)">Total Volume:</span> <strong style="font-family: var(--font-mono)">${s.total_bytes.toLocaleString()} B</strong></div>
          <div><span style="color: var(--text-tertiary)">Avg Size:</span> <strong style="font-family: var(--font-mono)">${s.avg_packet_size} B</strong></div>
        </div>
      </div>
    `;
  } catch (err) {
    console.error('Prediction failed:', err);
    elements.predictResultBody.innerHTML = `<div style="padding: 24px; text-align: center; color: #fb7185">Inference Execution Failed: ${err.message}</div>`;
  } finally {
    elements.runPredictionBtn.disabled = false;
  }
}

// CLI Studio Helpers
function updateCliCommand() {
  const sub = elements.cliSubcommand.value;
  const cfg = elements.cliConfig.value;
  const epochs = elements.cliEpochs.value;
  const flows = elements.cliFlows.value;
  const device = elements.cliDevice.value;

  let cmd = `antod ${sub} --config ${cfg}`;
  if (epochs) cmd += ` --epochs ${epochs}`;
  if (flows) cmd += ` --n-flows ${flows}`;
  if (device) cmd += ` --device ${device}`;

  elements.cliGeneratedCmd.innerText = cmd;
}

async function executeCliCommand() {
  const sub = elements.cliSubcommand.value;
  const cfg = elements.cliConfig.value;
  const epochs = elements.cliEpochs.value;
  const flows = elements.cliFlows.value;
  const device = elements.cliDevice.value;

  const args = [sub, '--config', cfg];
  if (epochs) args.push('--epochs', epochs);
  if (flows) args.push('--n-flows', flows);
  if (device) args.push('--device', device);

  elements.executeCmdBtn.disabled = true;
  elements.cliStatusBadge.textContent = 'Running...';
  elements.cliStatusBadge.className = 'badge amber';
  elements.cliTerminalOutput.innerText = `[ANTOD Runner] Launching: antod ${args.join(' ')}\n\n`;

  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ args }),
    });
    const data = await res.json();
    state.runningTaskId = data.task_id;
    pollCliOutput();
  } catch (err) {
    elements.cliTerminalOutput.innerText += `\nFailed to start task: ${err.message}\n`;
    elements.cliStatusBadge.textContent = 'Failed';
    elements.cliStatusBadge.className = 'badge red';
    elements.executeCmdBtn.disabled = false;
  }
}

function pollCliOutput() {
  if (state.pollTimer) clearInterval(state.pollTimer);

  state.pollTimer = setInterval(async () => {
    if (!state.runningTaskId) {
      clearInterval(state.pollTimer);
      return;
    }

    try {
      const res = await fetch(`/api/run/status?id=${state.runningTaskId}`);
      const task = await res.json();

      elements.cliTerminalOutput.innerText = task.output || 'Waiting for output...';
      elements.cliTerminalOutput.scrollTop = elements.cliTerminalOutput.scrollHeight;

      if (task.status === 'done') {
        clearInterval(state.pollTimer);
        elements.cliStatusBadge.textContent = 'Completed (Exit Code 0)';
        elements.cliStatusBadge.className = 'badge green';
        elements.executeCmdBtn.disabled = false;
      } else if (task.status === 'failed') {
        clearInterval(state.pollTimer);
        elements.cliStatusBadge.textContent = `Failed (Exit Code ${task.returncode})`;
        elements.cliStatusBadge.className = 'badge red';
        elements.executeCmdBtn.disabled = false;
      }
    } catch (err) {
      console.error('Error polling CLI task:', err);
    }
  }, 1000);
}

// Start application
document.addEventListener('DOMContentLoaded', init);
