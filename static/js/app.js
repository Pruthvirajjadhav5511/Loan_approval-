/**
 * FinRisk AI - Interactive Application Logic
 * Integrates ML Inference, Database CRUD, Analytics Charts, and Model Operations.
 */

// Global State
const state = {
  currentTab: 'tab-predict',
  records: {
    page: 1,
    size: 10,
    search: '',
    risk_tier: '',
    prediction: '',
    total: 0,
    totalPages: 1,
    items: []
  },
  analytics: null,
  modelInfo: null,
  charts: {
    tierChart: null,
    purposeChart: null,
    featureChart: null
  },
  activeModalRecord: null
};

// ============================================================================
// Initialization & Event Listeners
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initFormControls();
  initPresetButtons();
  initPredictionForm();
  initRecordsTab();
  initModal();
  initRetrainHandler();
  checkSystemHealth();

  // Load initial tab data
  fetchModelInfo();
});

// ============================================================================
// System Health Check
// ============================================================================
async function checkSystemHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    const indicator = document.getElementById('system-status-indicator');
    const label = document.getElementById('status-label');

    if (data.status === 'healthy') {
      indicator.innerHTML = `<span class="status-dot online"></span><span class="status-text">${data.stored_records} Records Persisted</span>`;
    } else {
      indicator.innerHTML = `<span class="status-dot" style="background:#EF4444;"></span><span class="status-text">Degraded</span>`;
    }
  } catch (err) {
    console.error('Health check failed', err);
  }
}

// ============================================================================
// Navigation Tabs
// ============================================================================
function initNavigation() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetTabId = tab.getAttribute('data-tab');
      switchTab(targetTabId);
    });
  });

  const jumpBtn = document.getElementById('jump-to-records-btn');
  if (jumpBtn) {
    jumpBtn.addEventListener('click', () => {
      switchTab('tab-records');
    });
  }
}

function switchTab(tabId) {
  state.currentTab = tabId;

  // Update nav buttons
  document.querySelectorAll('.nav-tab').forEach(b => {
    b.classList.toggle('active', b.getAttribute('data-tab') === tabId);
  });

  // Update panes
  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.toggle('active', pane.id === tabId);
  });

  // Trigger data fetches on tab entrance
  if (tabId === 'tab-records') {
    fetchRecords();
  } else if (tabId === 'tab-analytics') {
    fetchAnalytics();
  } else if (tabId === 'tab-model') {
    fetchModelInfo();
  }
}

// ============================================================================
// Form Controls & Sliders
// ============================================================================
function initFormControls() {
  const creditSlider = document.getElementById('credit_score');
  const creditVal = document.getElementById('credit_score_val');
  if (creditSlider && creditVal) {
    creditSlider.addEventListener('input', (e) => {
      creditVal.textContent = e.target.value;
    });
  }

  const dtiSlider = document.getElementById('debt_to_income_ratio');
  const dtiVal = document.getElementById('dti_val');
  if (dtiSlider && dtiVal) {
    dtiSlider.addEventListener('input', (e) => {
      dtiVal.textContent = `${e.target.value}%`;
    });
  }
}

// ============================================================================
// Presets for 1-Click Evaluation
// ============================================================================
function initPresetButtons() {
  const presets = {
    prime: {
      applicant_name: "Dr. Sophia Sterling",
      age: 41,
      annual_income: 145000,
      credit_score: 795,
      loan_amount: 25000,
      loan_term_months: "36",
      debt_to_income_ratio: 14.5,
      employment_years: 9.5,
      home_ownership: "OWN",
      loan_purpose: "home_improvement",
      has_prior_default: false
    },
    moderate: {
      applicant_name: "Jordan Hayes",
      age: 29,
      annual_income: 64000,
      credit_score: 645,
      loan_amount: 18000,
      loan_term_months: "48",
      debt_to_income_ratio: 31.5,
      employment_years: 3.2,
      home_ownership: "RENT",
      loan_purpose: "debt_consolidation",
      has_prior_default: false
    },
    subprime: {
      applicant_name: "Damian Frost",
      age: 35,
      annual_income: 38000,
      credit_score: 535,
      loan_amount: 32000,
      loan_term_months: "60",
      debt_to_income_ratio: 47.0,
      employment_years: 0.8,
      home_ownership: "RENT",
      loan_purpose: "personal",
      has_prior_default: true
    }
  };

  const setForm = (data) => {
    document.getElementById('applicant_name').value = data.applicant_name;
    document.getElementById('age').value = data.age;
    document.getElementById('annual_income').value = data.annual_income;
    document.getElementById('credit_score').value = data.credit_score;
    document.getElementById('credit_score_val').textContent = data.credit_score;
    document.getElementById('loan_amount').value = data.loan_amount;
    document.getElementById('loan_term_months').value = data.loan_term_months;
    document.getElementById('debt_to_income_ratio').value = data.debt_to_income_ratio;
    document.getElementById('dti_val').textContent = `${data.debt_to_income_ratio}%`;
    document.getElementById('employment_years').value = data.employment_years;
    document.getElementById('home_ownership').value = data.home_ownership;
    document.getElementById('loan_purpose').value = data.loan_purpose;
    document.getElementById('has_prior_default').checked = data.has_prior_default;
  };

  document.getElementById('preset-prime-btn').addEventListener('click', () => {
    setForm(presets.prime);
    showToast('Loaded Prime Applicant Profile', 'info');
  });

  document.getElementById('preset-moderate-btn').addEventListener('click', () => {
    setForm(presets.moderate);
    showToast('Loaded Moderate Risk Applicant Profile', 'info');
  });

  document.getElementById('preset-subprime-btn').addEventListener('click', () => {
    setForm(presets.subprime);
    showToast('Loaded High Risk Applicant Profile', 'info');
  });
}

// ============================================================================
// ML Prediction Form Submission & Persistence
// ============================================================================
function initPredictionForm() {
  const form = document.getElementById('prediction-form');
  const submitBtn = document.getElementById('predict-submit-btn');
  const spinner = document.getElementById('predict-spinner');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Payload construction
    const payload = {
      applicant_name: document.getElementById('applicant_name').value.trim(),
      age: parseInt(document.getElementById('age').value, 10),
      annual_income: parseFloat(document.getElementById('annual_income').value),
      credit_score: parseInt(document.getElementById('credit_score').value, 10),
      loan_amount: parseFloat(document.getElementById('loan_amount').value),
      loan_term_months: parseInt(document.getElementById('loan_term_months').value, 10),
      employment_years: parseFloat(document.getElementById('employment_years').value),
      debt_to_income_ratio: parseFloat(document.getElementById('debt_to_income_ratio').value),
      home_ownership: document.getElementById('home_ownership').value,
      loan_purpose: document.getElementById('loan_purpose').value,
      has_prior_default: document.getElementById('has_prior_default').checked
    };

    submitBtn.disabled = true;
    spinner.classList.remove('hidden');

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const record = await response.json();
      renderPredictionResult(record);
      showToast(`Decision: ${record.prediction} (Persisted as Record #${record.id})`, record.prediction === 'APPROVED' ? 'success' : 'error');
      checkSystemHealth();
    } catch (err) {
      console.error(err);
      showToast(`Error: ${err.message}`, 'error');
    } finally {
      submitBtn.disabled = false;
      spinner.classList.add('hidden');
    }
  });
}

function renderPredictionResult(record) {
  const emptyState = document.getElementById('result-empty-state');
  const resultContent = document.getElementById('result-content');
  const banner = document.getElementById('decision-banner');
  const badge = document.getElementById('decision-badge');
  const sub = document.getElementById('decision-sub');
  const modelTag = document.getElementById('model-version-tag');

  emptyState.classList.add('hidden');
  resultContent.classList.remove('hidden');

  const isApproved = record.prediction === 'APPROVED';
  banner.className = `decision-banner ${isApproved ? 'approved' : 'rejected'}`;
  badge.textContent = record.prediction;
  sub.textContent = isApproved
    ? `Underwriting criteria fulfilled with ${(record.approval_probability * 100).toFixed(1)}% probability`
    : `Elevated default risk detected (${(record.approval_probability * 100).toFixed(1)}% approval probability)`;

  modelTag.textContent = `Model: ${record.model_version}`;

  // Animate Circular Gauge
  const score = Math.round(record.risk_score);
  const gaugeMeter = document.getElementById('gauge-meter');
  const gaugeScore = document.getElementById('gauge-score');
  gaugeScore.textContent = score;

  // Circumference = 2 * PI * 50 = ~314.15
  const circumference = 314.15;
  const offset = circumference - (score / 100) * circumference;
  gaugeMeter.style.strokeDashoffset = offset;

  // Dynamic meter color based on risk score
  if (score < 30) {
    gaugeMeter.style.stroke = 'var(--success)';
  } else if (score < 60) {
    gaugeMeter.style.stroke = 'var(--warning)';
  } else {
    gaugeMeter.style.stroke = 'var(--danger)';
  }

  // Stat Pills
  document.getElementById('pill-risk-tier').textContent = record.risk_tier;
  document.getElementById('pill-apr').textContent = `${record.suggested_interest_rate.toFixed(2)}% APR`;
  document.getElementById('pill-confidence').textContent = `${record.confidence_score.toFixed(1)}%`;

  // Render Explainability Factors
  const factorsList = document.getElementById('factors-list');
  factorsList.innerHTML = '';

  if (record.top_factors && record.top_factors.length > 0) {
    record.top_factors.forEach(f => {
      const card = document.createElement('div');
      const impactClass = f.impact.toLowerCase();
      card.className = `factor-card ${impactClass}`;
      card.innerHTML = `
        <span class="factor-tag">${f.impact}</span>
        <div class="factor-desc">
          <strong>${f.feature}:</strong> ${f.description}
        </div>
      `;
      factorsList.appendChild(card);
    });
  }

  // Database Confirmation
  document.getElementById('db-saved-id').textContent = `#${record.id}`;
}

// ============================================================================
// Database Records Table & Retrieval
// ============================================================================
function initRecordsTab() {
  const searchInput = document.getElementById('records-search-input');
  const tierFilter = document.getElementById('filter-tier');
  const decisionFilter = document.getElementById('filter-decision');
  const refreshBtn = document.getElementById('refresh-records-btn');
  const exportBtn = document.getElementById('export-csv-btn');
  const prevBtn = document.getElementById('prev-page-btn');
  const nextBtn = document.getElementById('next-page-btn');

  // Debounced live search
  let debounceTimer;
  searchInput.addEventListener('input', (e) => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      state.records.search = e.target.value.trim();
      state.records.page = 1;
      fetchRecords();
    }, 300);
  });

  tierFilter.addEventListener('change', (e) => {
    state.records.risk_tier = e.target.value;
    state.records.page = 1;
    fetchRecords();
  });

  decisionFilter.addEventListener('change', (e) => {
    state.records.prediction = e.target.value;
    state.records.page = 1;
    fetchRecords();
  });

  refreshBtn.addEventListener('click', () => {
    fetchRecords();
    showToast('Records refreshed', 'info');
  });

  exportBtn.addEventListener('click', exportRecordsToCSV);

  prevBtn.addEventListener('click', () => {
    if (state.records.page > 1) {
      state.records.page--;
      fetchRecords();
    }
  });

  nextBtn.addEventListener('click', () => {
    if (state.records.page < state.records.totalPages) {
      state.records.page++;
      fetchRecords();
    }
  });
}

async function fetchRecords() {
  const tbody = document.getElementById('records-tbody');
  tbody.innerHTML = `<tr><td colspan="11" style="text-align:center; padding: 2rem;"><span class="btn-spinner" style="display:inline-block;"></span> Loading records from database...</td></tr>`;

  const params = new URLSearchParams({
    page: state.records.page,
    size: state.records.size
  });
  if (state.records.search) params.append('search', state.records.search);
  if (state.records.risk_tier) params.append('risk_tier', state.records.risk_tier);
  if (state.records.prediction) params.append('prediction', state.records.prediction);

  try {
    const res = await fetch(`/api/predictions?${params.toString()}`);
    if (!res.ok) throw new Error('Failed to retrieve records');

    const data = await res.json();
    state.records.items = data.items;
    state.records.total = data.total;
    state.records.totalPages = data.total_pages;

    renderRecordsTable(data);
  } catch (err) {
    console.error(err);
    tbody.innerHTML = `<tr><td colspan="11" style="text-align:center; padding: 2rem; color:var(--danger);">Failed to load database records: ${err.message}</td></tr>`;
  }
}

function renderRecordsTable(data) {
  const tbody = document.getElementById('records-tbody');
  tbody.innerHTML = '';

  if (!data.items || data.items.length === 0) {
    tbody.innerHTML = `<tr><td colspan="11" style="text-align:center; padding: 2.5rem; color:var(--text-muted);">No prediction records match the current filter.</td></tr>`;
    updatePaginationControls(0, 0, 1, 1);
    return;
  }

  data.items.forEach(item => {
    const tr = document.createElement('tr');

    const isApp = item.prediction === 'APPROVED';
    const decisionBadge = `<span class="badge ${isApp ? 'badge-success' : 'badge-danger'}">${item.prediction}</span>`;

    let tierBadgeClass = 'badge-success';
    if (item.risk_tier === 'Moderate Risk') tierBadgeClass = 'badge-warning';
    if (item.risk_tier === 'High Risk') tierBadgeClass = 'badge-danger';

    const formattedDate = new Date(item.created_at).toLocaleString([], {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });

    tr.innerHTML = `
      <td class="table-id">#${item.id}</td>
      <td>
        <div class="table-applicant">${escapeHtml(item.applicant_name)}</div>
        <div class="table-meta-sub">${item.age} yrs &bull; ${item.home_ownership}</div>
      </td>
      <td><strong>${item.credit_score}</strong></td>
      <td>$${Math.round(item.annual_income).toLocaleString()}</td>
      <td><strong>$${Math.round(item.loan_amount).toLocaleString()}</strong></td>
      <td>${item.debt_to_income_ratio.toFixed(1)}%</td>
      <td><span class="badge ${tierBadgeClass}">${item.risk_score.toFixed(0)} (${item.risk_tier.split(' ')[0]})</span></td>
      <td>${decisionBadge}</td>
      <td><span class="badge" style="font-size:0.68rem;">${item.review_status}</span></td>
      <td style="color:var(--text-dim); font-size:0.8rem;">${formattedDate}</td>
      <td>
        <div class="action-btn-group">
          <button type="button" class="icon-action-btn" title="View & Edit Details" onclick="openRecordModal(${item.id})">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
          </button>
          <button type="button" class="icon-action-btn delete" title="Delete Record" onclick="deleteRecord(${item.id})">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
          </button>
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  });

  updatePaginationControls(data.items.length, data.total, data.page, data.total_pages);
}

function updatePaginationControls(count, total, page, totalPages) {
  const info = document.getElementById('pagination-info');
  const indicator = document.getElementById('page-indicator');
  const prevBtn = document.getElementById('prev-page-btn');
  const nextBtn = document.getElementById('next-page-btn');

  info.textContent = `Showing ${count} of ${total} records`;
  indicator.textContent = `Page ${page} of ${Math.max(1, totalPages)}`;

  prevBtn.disabled = page <= 1;
  nextBtn.disabled = page >= totalPages;
}

// ============================================================================
// Record Modal & Status Override
// ============================================================================
function initModal() {
  const overlay = document.getElementById('record-modal-overlay');
  const closeBtn = document.getElementById('modal-close-btn');
  const dismissBtn = document.getElementById('modal-dismiss-btn');
  const saveBtn = document.getElementById('modal-save-btn');

  const closeModal = () => overlay.classList.add('hidden');

  closeBtn.addEventListener('click', closeModal);
  dismissBtn.addEventListener('click', closeModal);
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) closeModal();
  });

  saveBtn.addEventListener('click', async () => {
    if (!state.activeModalRecord) return;

    const newStatus = document.getElementById('modal-review-status').value;
    const newNotes = document.getElementById('modal-notes').value;

    try {
      const res = await fetch(`/api/predictions/${state.activeModalRecord.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review_status: newStatus, notes: newNotes })
      });

      if (!res.ok) throw new Error('Failed to update record');

      showToast(`Record #${state.activeModalRecord.id} updated successfully!`, 'success');
      closeModal();
      fetchRecords();
    } catch (err) {
      showToast(err.message, 'error');
    }
  });
}

window.openRecordModal = async function(id) {
  try {
    const res = await fetch(`/api/predictions/${id}`);
    if (!res.ok) throw new Error('Record not found');

    const record = await res.json();
    state.activeModalRecord = record;

    const modalBody = document.getElementById('modal-body-content');
    modalBody.innerHTML = `
      <div class="modal-grid-two">
        <div class="modal-field">
          <span class="modal-field-label">Applicant Name</span>
          <span class="modal-field-val">${escapeHtml(record.applicant_name)}</span>
        </div>
        <div class="modal-field">
          <span class="modal-field-label">Reference UUID</span>
          <span class="modal-field-val" style="font-family:var(--font-mono); font-size:0.75rem;">${record.reference_id}</span>
        </div>
      </div>

      <div class="modal-grid-two">
        <div class="modal-field">
          <span class="modal-field-label">Requested Loan</span>
          <span class="modal-field-val">$${Math.round(record.loan_amount).toLocaleString()} (${record.loan_term_months} mos)</span>
        </div>
        <div class="modal-field">
          <span class="modal-field-label">Annual Income & DTI</span>
          <span class="modal-field-val">$${Math.round(record.annual_income).toLocaleString()} &bull; ${record.debt_to_income_ratio.toFixed(1)}% DTI</span>
        </div>
      </div>

      <div class="modal-grid-two">
        <div class="modal-field">
          <span class="modal-field-label">ML Model Decision</span>
          <span class="modal-field-val">
            <span class="badge ${record.prediction === 'APPROVED' ? 'badge-success' : 'badge-danger'}">${record.prediction}</span>
            (${record.risk_tier} &bull; Score ${record.risk_score.toFixed(0)})
          </span>
        </div>
        <div class="modal-field">
          <span class="modal-field-label">Suggested APR</span>
          <span class="modal-field-val">${record.suggested_interest_rate.toFixed(2)}% APR</span>
        </div>
      </div>

      <div class="form-group" style="margin-top:0.6rem;">
        <label for="modal-review-status">Review / Override Status:</label>
        <select id="modal-review-status">
          <option value="AUTO_DECISION" ${record.review_status === 'AUTO_DECISION' ? 'selected' : ''}>AUTO_DECISION</option>
          <option value="APPROVED_OVERRIDE" ${record.review_status === 'APPROVED_OVERRIDE' ? 'selected' : ''}>APPROVED_OVERRIDE (Manual Underwriter Sign-off)</option>
          <option value="REJECTED_OVERRIDE" ${record.review_status === 'REJECTED_OVERRIDE' ? 'selected' : ''}>REJECTED_OVERRIDE (High Leverage Declined)</option>
          <option value="FLAGGED" ${record.review_status === 'FLAGGED' ? 'selected' : ''}>FLAGGED (Requires Secondary Verification)</option>
          <option value="ARCHIVED" ${record.review_status === 'ARCHIVED' ? 'selected' : ''}>ARCHIVED</option>
        </select>
      </div>

      <div class="form-group">
        <label for="modal-notes">Underwriting Audit Notes:</label>
        <textarea id="modal-notes" rows="3" style="background:rgba(13,18,31,0.7); border:1px solid var(--bg-card-border); border-radius:var(--radius-sm); padding:0.6rem; color:#fff; font-family:inherit; font-size:0.85rem; width:100%; outline:none;" placeholder="Enter notes or justification...">${record.notes || ''}</textarea>
      </div>
    `;

    document.getElementById('record-modal-overlay').classList.remove('hidden');
  } catch (err) {
    showToast(err.message, 'error');
  }
};

window.deleteRecord = async function(id) {
  if (!confirm(`Are you sure you want to delete prediction record #${id} from the database?`)) {
    return;
  }

  try {
    const res = await fetch(`/api/predictions/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete record');

    showToast(`Record #${id} deleted from database`, 'info');
    fetchRecords();
    checkSystemHealth();
  } catch (err) {
    showToast(err.message, 'error');
  }
};

// ============================================================================
// Export CSV Functionality
// ============================================================================
async function exportRecordsToCSV() {
  try {
    const res = await fetch('/api/predictions?page=1&size=100');
    const data = await res.json();
    if (!data.items || data.items.length === 0) {
      showToast('No records available to export', 'info');
      return;
    }

    const headers = [
      'ID', 'Reference', 'ApplicantName', 'Age', 'AnnualIncome', 'CreditScore',
      'LoanAmount', 'TermMonths', 'DTI', 'HomeOwnership', 'Purpose', 'PriorDefault',
      'Prediction', 'RiskScore', 'RiskTier', 'SuggestedAPR', 'ReviewStatus', 'CreatedAt'
    ];

    const rows = data.items.map(item => [
      item.id,
      item.reference_id,
      `"${item.applicant_name.replace(/"/g, '""')}"`,
      item.age,
      item.annual_income,
      item.credit_score,
      item.loan_amount,
      item.loan_term_months,
      item.debt_to_income_ratio,
      item.home_ownership,
      item.loan_purpose,
      item.has_prior_default,
      item.prediction,
      item.risk_score,
      item.risk_tier,
      item.suggested_interest_rate,
      item.review_status,
      item.created_at
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `finrisk_predictions_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    showToast('Exported CSV with database records', 'success');
  } catch (err) {
    showToast(`CSV Export error: ${err.message}`, 'error');
  }
}

// ============================================================================
// Portfolio Analytics & Charts
// ============================================================================
async function fetchAnalytics() {
  const refreshBtn = document.getElementById('refresh-analytics-btn');
  if (refreshBtn) refreshBtn.disabled = true;

  try {
    const res = await fetch('/api/stats');
    if (!res.ok) throw new Error('Failed to fetch statistics');

    const stats = await res.json();
    state.analytics = stats;

    // Render KPI Cards
    document.getElementById('kpi-total-apps').textContent = stats.total_applications.toLocaleString();
    document.getElementById('kpi-approval-rate').textContent = `${stats.approval_rate.toFixed(1)}%`;
    document.getElementById('kpi-approved-count').textContent = stats.approved_count.toLocaleString();
    document.getElementById('kpi-rejected-count').textContent = stats.rejected_count.toLocaleString();
    document.getElementById('kpi-avg-risk').textContent = stats.avg_risk_score.toFixed(1);
    document.getElementById('kpi-loan-volume').textContent = `$${Math.round(stats.total_loan_volume).toLocaleString()}`;
    document.getElementById('kpi-avg-loan').textContent = `$${Math.round(stats.avg_loan_amount).toLocaleString()}`;

    // Render Charts
    renderAnalyticsCharts(stats);
  } catch (err) {
    console.error(err);
    showToast(`Analytics error: ${err.message}`, 'error');
  } finally {
    if (refreshBtn) refreshBtn.disabled = false;
  }
}

function renderAnalyticsCharts(stats) {
  // Chart 1: Risk Tier Donut
  const tierCtx = document.getElementById('tierChart').getContext('2d');
  if (state.charts.tierChart) state.charts.tierChart.destroy();

  const tierLabels = Object.keys(stats.risk_tier_breakdown);
  const tierData = Object.values(stats.risk_tier_breakdown);

  state.charts.tierChart = new Chart(tierCtx, {
    type: 'doughnut',
    data: {
      labels: tierLabels,
      datasets: [{
        data: tierData,
        backgroundColor: ['#10B981', '#F59E0B', '#EF4444'],
        borderWidth: 2,
        borderColor: '#0D121F'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#9CA3AF', font: { family: 'Plus Jakarta Sans', size: 12 } }
        }
      },
      cutout: '65%'
    }
  });

  // Chart 2: Loan Purpose Bar Chart
  const purposeCtx = document.getElementById('purposeChart').getContext('2d');
  if (state.charts.purposeChart) state.charts.purposeChart.destroy();

  const purposeLabels = Object.keys(stats.purpose_breakdown).map(k => k.replace('_', ' '));
  const purposeData = Object.values(stats.purpose_breakdown);

  state.charts.purposeChart = new Chart(purposeCtx, {
    type: 'bar',
    data: {
      labels: purposeLabels,
      datasets: [{
        label: 'Applications',
        data: purposeData,
        backgroundColor: '#6366F1',
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          ticks: { color: '#9CA3AF', font: { size: 11 } },
          grid: { display: false }
        },
        y: {
          ticks: { color: '#9CA3AF', precision: 0 },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        }
      }
    }
  });
}

// ============================================================================
// ML Model Information & Retraining
// ============================================================================
async function fetchModelInfo() {
  try {
    const res = await fetch('/api/model/info');
    if (!res.ok) throw new Error('Failed to fetch model info');

    const info = await res.json();
    state.modelInfo = info;

    document.getElementById('model-algo-name').textContent = info.algorithm;
    document.getElementById('model-meta-version').textContent = info.version;
    document.getElementById('model-train-samples').textContent = info.training_samples.toLocaleString();

    const trainedDate = new Date(info.trained_at).toLocaleString([], {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
    });
    document.getElementById('model-trained-date').textContent = trainedDate;

    // Metrics KPIs
    document.getElementById('metric-accuracy').textContent = `${(info.accuracy * 100).toFixed(1)}%`;
    document.getElementById('metric-roc-auc').textContent = info.roc_auc.toFixed(3);
    document.getElementById('metric-f1').textContent = info.f1_score.toFixed(3);
    document.getElementById('metric-prec-rec').textContent = `${(info.precision * 100).toFixed(0)}% / ${(info.recall * 100).toFixed(0)}%`;

    // Render Feature Importance Chart
    renderFeatureImportanceChart(info.feature_importances);
  } catch (err) {
    console.error(err);
  }
}

function renderFeatureImportanceChart(importances) {
  const ctx = document.getElementById('featureImportanceChart').getContext('2d');
  if (state.charts.featureChart) state.charts.featureChart.destroy();

  const labels = Object.keys(importances).map(l => l.replace(/_/g, ' '));
  const values = Object.values(importances);

  state.charts.featureChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Gini Importance %',
        data: values,
        backgroundColor: '#06B6D4',
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          ticks: { color: '#9CA3AF', callback: val => `${val}%` },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        },
        y: {
          ticks: { color: '#E2E8F0', font: { size: 11 } },
          grid: { display: false }
        }
      }
    }
  });
}

function initRetrainHandler() {
  const retrainBtn = document.getElementById('retrain-btn');
  const spinner = document.getElementById('retrain-spinner');
  const resultBox = document.getElementById('retrain-result-box');
  const samplesInput = document.getElementById('retrain-samples');

  retrainBtn.addEventListener('click', async () => {
    const samples = parseInt(samplesInput.value, 10) || 5000;
    retrainBtn.disabled = true;
    spinner.classList.remove('hidden');
    resultBox.classList.add('hidden');

    try {
      const res = await fetch(`/api/model/retrain?samples=${samples}`, {
        method: 'POST'
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Retraining failed');
      }

      const data = await res.json();
      resultBox.classList.remove('hidden');
      resultBox.innerHTML = `
        <strong>⚡ Retraining Complete!</strong><br>
        Version: <code>${data.new_version}</code><br>
        New ROC-AUC: <strong>${data.metrics.roc_auc.toFixed(4)}</strong> &bull; Accuracy: <strong>${(data.metrics.accuracy * 100).toFixed(1)}%</strong><br>
        Artifacts saved & hot-reloaded into memory.
      `;

      showToast(`Model ${data.new_version} retrained successfully!`, 'success');
      fetchModelInfo();
      checkSystemHealth();
    } catch (err) {
      showToast(`Retrain error: ${err.message}`, 'error');
    } finally {
      retrainBtn.disabled = false;
      spinner.classList.add('hidden');
    }
  });
}

// ============================================================================
// Utilities
// ============================================================================
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
}
