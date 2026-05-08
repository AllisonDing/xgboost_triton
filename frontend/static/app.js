/**
 * Fraud Detection Frontend
 * ========================
 * This file handles all interaction between the browser UI and the FastAPI backend.
 *
 * Data flow for a single prediction:
 *
 *   1. User clicks "Random" or enters a transaction index
 *   2. GET /api/transactions/{index}  → receive 466 features
 *   3. User clicks "Run Inference"
 *   4. POST /api/predict/transaction/{index}  → FastAPI fetches features,
 *      preprocesses (fillna -999), casts to FP32, calls Triton HTTP API,
 *      Triton runs XGBoost Python backend, returns fraud_probability
 *   5. Browser renders probability gauge + risk badge
 *
 * For batch:
 *   POST /api/predict/batch  { transaction_indices: [0..N] }
 *   Triton processes the whole batch in one forward pass (dynamic batching)
 */

const API = '/api';   // nginx proxies /api/* → FastAPI on port 8080

// ── State ─────────────────────────────────────────────────────────────────────
let currentTx = null;   // currently loaded transaction object
let healthTimer = null;

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  pollHealth();
  healthTimer = setInterval(pollHealth, 8000);
});

// ── Health polling ────────────────────────────────────────────────────────────
async function pollHealth() {
  try {
    const h = await fetchJSON(`${API}/health`);
    renderHealth(h);
  } catch {
    renderHealth(null);
  }
}

function renderHealth(h) {
  const bar = document.getElementById('health-bar');
  if (!h) {
    bar.innerHTML = pill('API', false) + pill('Triton', false) + pill('Model', false);
    return;
  }
  bar.innerHTML =
    pill('API',    h.api_status   === 'healthy')   +
    pill('Triton', h.triton_status === 'healthy')  +
    pill('Model',  h.model_status  === 'ready')    +
    (h.data_loaded
      ? `<span class="status-pill status-ok"><span class="dot"></span>${fmt(h.total_test_transactions)} transactions</span>`
      : `<span class="status-pill status-err"><span class="dot"></span>Data not loaded — run notebooks first</span>`);
}

function pill(label, ok) {
  const cls = ok ? 'status-ok' : 'status-err';
  return `<span class="status-pill ${cls}"><span class="dot"></span>${label}</span>`;
}

// ── Transaction loading ───────────────────────────────────────────────────────
async function loadRandomTransaction() {
  await loadTransaction(null);
}

async function loadTransactionByInput() {
  const idx = parseInt(document.getElementById('tx-index').value, 10);
  if (isNaN(idx) || idx < 0) { toast('Enter a valid transaction index', true); return; }
  await loadTransaction(idx);
}

async function loadTransaction(index) {
  const panel = document.getElementById('tx-features');
  panel.innerHTML = '<span class="spinner"></span> Loading ...';
  document.getElementById('predict-btn').disabled = true;
  clearResult();

  try {
    const url = index === null ? `${API}/transactions/random` : `${API}/transactions/${index}`;
    const tx = await fetchJSON(url);
    currentTx = tx;

    document.getElementById('tx-index').value = tx.index;
    renderFeatures(tx);
    document.getElementById('predict-btn').disabled = false;
  } catch (err) {
    panel.innerHTML = `<div class="placeholder">Failed to load transaction: ${err.message}</div>`;
    toast(err.message, true);
  }
}

function renderFeatures(tx) {
  const panel = document.getElementById('tx-features');
  const txId = tx.transaction_id ? `TransactionID <span>${tx.transaction_id}</span>` : `Row <span>${tx.index}</span>`;

  let rows = '';
  for (const [name, val] of Object.entries(tx.top_features)) {
    rows += `<div class="feat-row">
               <span class="feat-name">${name}</span>
               <span class="feat-val mono">${val === -999 ? 'N/A' : val.toFixed(4)}</span>
             </div>`;
  }

  panel.innerHTML = `
    <div class="tx-meta">Index <span>${tx.index}</span> &nbsp;·&nbsp; ${txId}</div>
    <p class="section-title" style="font-size:.7rem;margin-bottom:.5rem">Top 10 features by importance</p>
    <div class="feature-grid">${rows}</div>
    <p style="font-size:.72rem;color:var(--text-muted);margin-top:.5rem">
      Showing 10 / ${Object.keys(tx.features).length || '466'} features
      (anonymised Vesta dataset — V* are engineered behavioural signals)
    </p>`;
}

// ── Single prediction ─────────────────────────────────────────────────────────
async function runPrediction() {
  if (!currentTx) { toast('Load a transaction first', true); return; }
  const btn = document.getElementById('predict-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';

  try {
    const res = await fetchJSON(`${API}/predict/transaction/${currentTx.index}`, {
      method: 'POST',
    });
    renderResult(res);
  } catch (err) {
    toast(err.message, true);
    document.getElementById('result-area').innerHTML =
      `<div class="placeholder">Inference failed: ${err.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Run Inference →';
  }
}

function renderResult(res) {
  const prob   = res.result.fraud_probability;
  const risk   = res.result.risk_level;
  const pct    = Math.round(prob * 100);
  const color  = { HIGH: '#f85149', MEDIUM: '#d29922', LOW: '#e3b341', NONE: '#76b900' }[risk];

  // SVG circle gauge
  const R  = 56;
  const C  = 2 * Math.PI * R;
  const offset = C * (1 - prob);

  document.getElementById('result-area').innerHTML = `
    <div class="gauge-wrap">
      <svg class="gauge-svg" viewBox="0 0 130 130">
        <circle class="gauge-bg"   cx="65" cy="65" r="${R}"/>
        <circle class="gauge-fill" cx="65" cy="65" r="${R}"
                stroke="${color}"
                stroke-dasharray="${C}"
                stroke-dashoffset="${offset}"
                style="transition:stroke-dashoffset .6s ease"/>
      </svg>
      <div class="gauge-label">
        <span class="gauge-pct" style="color:${color}">${pct}%</span>
        <span class="gauge-sub">fraud prob</span>
      </div>
    </div>

    <div>
      <span class="risk-badge risk-${risk}">${risk} RISK</span>
    </div>

    <div class="result-meta">
      <strong>${res.result.is_fraud ? '⚠ Flagged as FRAUD' : '✓ Transaction clean'}</strong><br>
      Inference latency: <strong>${res.latency_ms.toFixed(2)} ms</strong><br>
      Path: Browser → FastAPI → Triton → XGBoost
    </div>`;
}

function clearResult() {
  document.getElementById('result-area').innerHTML =
    '<div class="placeholder">Load a transaction then click<br><em>Run Inference →</em></div>';
}

// ── Batch prediction ──────────────────────────────────────────────────────────
async function runBatch() {
  const n = parseInt(document.getElementById('batch-size').value, 10);
  if (isNaN(n) || n < 1 || n > 5000) { toast('Enter 1–5000 for batch size', true); return; }

  const btn = document.getElementById('batch-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Running ...';
  document.getElementById('batch-stats').innerHTML = '';
  document.getElementById('batch-table-wrap').innerHTML = '';

  const indices = Array.from({ length: n }, (_, i) => i);

  try {
    const res = await fetchJSON(`${API}/predict/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transaction_indices: indices }),
    });
    renderBatchResults(res);
  } catch (err) {
    toast(err.message, true);
    document.getElementById('batch-stats').innerHTML =
      `<div class="placeholder">Batch inference failed: ${err.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Run Batch Test';
  }
}

function renderBatchResults(res) {
  // Summary stats
  const fraudPct = ((res.fraud_count / res.total_transactions) * 100).toFixed(1);
  document.getElementById('batch-stats').innerHTML = `
    <div class="stats-grid">
      <div class="stat-card"><div class="stat-value">${fmt(res.total_transactions)}</div><div class="stat-label">Transactions</div></div>
      <div class="stat-card"><div class="stat-value" style="color:var(--red)">${res.fraud_count}</div><div class="stat-label">Fraud Detected (${fraudPct}%)</div></div>
      <div class="stat-card"><div class="stat-value" style="color:var(--orange)">${res.high_risk_count}</div><div class="stat-label">HIGH Risk</div></div>
      <div class="stat-card"><div class="stat-value">${res.avg_latency_ms.toFixed(2)}<small style="font-size:.6em">ms</small></div><div class="stat-label">Avg Latency / tx</div></div>
      <div class="stat-card"><div class="stat-value">${fmt(Math.round(res.throughput_tps))}</div><div class="stat-label">Throughput (tx/s)</div></div>
    </div>`;

  // Results table (show first 50)
  const rows = res.results.slice(0, 50).map(r => {
    const p      = r.result.fraud_probability;
    const pct    = Math.round(p * 100);
    const color  = { HIGH: '#f85149', MEDIUM: '#d29922', LOW: '#e3b341', NONE: '#76b900' }[r.result.risk_level];
    const fraud  = r.result.is_fraud;
    return `
      <tr class="${fraud ? 'fraud-row' : ''}">
        <td>${r.transaction_index}</td>
        <td>
          <div class="prob-bar-wrap">
            <div class="prob-bar" style="width:${pct}px;max-width:100px;background:${color}"></div>
            <span class="prob-text mono">${(p * 100).toFixed(1)}%</span>
          </div>
        </td>
        <td><span class="risk-badge risk-${r.result.risk_level}" style="padding:.15rem .55rem;font-size:.72rem">${r.result.risk_level}</span></td>
        <td>${fraud ? '<span style="color:var(--red)">⚠ FRAUD</span>' : '<span style="color:var(--green)">✓ Clean</span>'}</td>
      </tr>`;
  }).join('');

  const note = res.results.length > 50
    ? `<p style="font-size:.75rem;color:var(--text-muted);margin-top:.5rem">Showing first 50 of ${res.results.length} results</p>`
    : '';

  document.getElementById('batch-table-wrap').innerHTML = `
    <div class="results-table-wrap">
      <table>
        <thead><tr><th>Index</th><th>Fraud Probability</th><th>Risk</th><th>Verdict</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    ${note}`;
}

// ── Utilities ─────────────────────────────────────────────────────────────────
async function fetchJSON(url, options = {}) {
  const res = await fetch(url, options);
  const body = await res.json().catch(() => ({ detail: 'Non-JSON response' }));
  if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
  return body;
}

function fmt(n) {
  return n.toLocaleString();
}

let toastTimer = null;
function toast(msg, isErr = false) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'show' + (isErr ? ' toast-err' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = ''; }, 3500);
}
