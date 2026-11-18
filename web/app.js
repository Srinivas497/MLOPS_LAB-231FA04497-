const $ = (id) => document.getElementById(id);
const number = (v, digits = 1) => Number(v || 0).toFixed(digits);
const percent = (v) => `${(Number(v || 0) * 100).toFixed(1)}%`;

function showToast(message, isError = false) {
  const toast = $('toast');
  toast.textContent = message;
  toast.classList.toggle('error-text', isError);
  toast.classList.add('show');
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove('show'), 3200);
}

function setDataset(prefix, stats) {
  const p = prefix.toLowerCase();
  $(`${p}Count`).textContent = stats.records.toLocaleString();
  $(`${p}Rate`).textContent = `${percent(stats.missed_rate)} missed`;
  $(`${p}Bar`).style.width = `${Math.max(2, stats.missed_rate * 100)}%`;
  $(`${p}RateLarge`).textContent = percent(stats.missed_rate);
  $(`${p}RateTrack`).style.width = `${stats.missed_rate * 100}%`;
  $(`${p}Missed`).textContent = stats.missed.toLocaleString();
  $(`${p}Total`).textContent = stats.records.toLocaleString();
}

function renderMetrics(metrics) {
  const selected = metrics.selected_model;
  const entries = ['decision_tree', 'gradient_boosting'].filter((key) => metrics[key]);
  const champion = metrics[selected]?.metrics;
  $('championName').textContent = selected ? selected.replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase()) : 'Unavailable';
  $('championRecall').textContent = `Recall ${champion ? percent(champion.recall_missed) : '--'}`;
  $('modelRows').innerHTML = entries.map((key) => {
    const m = metrics[key].metrics;
    const isChampion = key === selected;
    const display = key.replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
    return `<tr class="${isChampion ? 'champion-row' : ''}">
      <td>${display}</td><td>${number(m.accuracy, 3)}</td><td>${number(m.precision_missed, 3)}</td>
      <td class="recall-value">${number(m.recall_missed, 3)}</td><td>${number(m.f1_missed, 3)}</td>
      <td>${number(m.roc_auc, 3)}</td><td><span class="${isChampion ? 'champion-badge' : 'candidate-badge'}">${isChampion ? 'CHAMPION' : 'CANDIDATE'}</span></td>
    </tr>`;
  }).join('') || '<tr><td colspan="7" class="loading-cell">Train a model to view comparison metrics.</td></tr>';
}

async function refreshDashboard() {
  try {
    const response = await fetch('/api/dashboard');
    if (!response.ok) throw new Error('Dashboard data unavailable');
    const data = await response.json();
    setDataset('v1', data.datasets.v1);
    setDataset('v2', data.datasets.v2);
    renderMetrics(data.metrics || {});
    $('registryName').textContent = data.registry.name;
    $('registryVersion').textContent = data.registry.latest_version ? `Version ${data.registry.latest_version}` : 'Not registered';
    $('modelState').textContent = data.health === 'ready' ? 'Model online' : 'Model not trained';
    $('modelState').classList.toggle('error-text', data.health !== 'ready');
    if (data.training.status === 'running') setTrainingButton(true);
  } catch (error) {
    $('modelState').textContent = 'Dashboard unavailable';
    showToast(error.message, true);
  }
}

function setTrainingButton(running) {
  const button = $('trainButton');
  button.disabled = running;
  button.innerHTML = running ? '<span class="button-icon">◌</span> Training models…' : '<span class="button-icon">⟳</span> Retrain & register';
}

async function startTraining() {
  setTrainingButton(true);
  try {
    const response = await fetch('/api/train', { method: 'POST' });
    if (!response.ok) {
      const body = await response.json();
      throw new Error(body.detail || 'Could not start training');
    }
    showToast('Training started. Comparing both classifiers…');
    let finished = false;
    while (!finished) {
      await new Promise(resolve => setTimeout(resolve, 1800));
      const status = await (await fetch('/api/train/status')).json();
      if (status.status === 'complete') {
        finished = true;
        showToast('Champion registered. Dashboard refreshed.');
        await refreshDashboard();
      } else if (status.status === 'failed') {
        finished = true;
        throw new Error(status.error || 'Training failed');
      }
    }
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setTrainingButton(false);
  }
}

$('predictionForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const request = Object.fromEntries(form.entries());
  for (const key of ['expected_train_delay_min','platform_to_bus_stop_distance_m','passenger_walking_speed_mps','station_crowd_density','scheduled_transfer_time_min']) request[key] = Number(request[key]);
  const button = event.currentTarget.querySelector('button[type="submit"]');
  button.disabled = true;
  button.querySelector('span:first-child').textContent = 'Estimating…';
  try {
    const response = await fetch('/predict', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Prediction failed');
    const atRisk = result.prediction === 'Connection at Risk';
    const panel = $('predictionResult');
    panel.classList.toggle('risk-result', atRisk);
    panel.classList.toggle('likely-result', !atRisk);
    panel.querySelector('.result-empty').innerHTML = `
      <div class="result-orb">${atRisk ? '!' : '✓'}</div>
      <div class="result-kicker">${atRisk ? 'ACTION MAY BE NEEDED' : 'TRANSFER LOOKS FEASIBLE'}</div>
      <div class="result-probability">${percent(result.missed_connection_probability)}</div>
      <h3>${result.prediction}</h3>
      <p class="result-sub">Estimated probability of missing the scheduled feeder bus.</p>
      <div class="result-mini"><span>ESTIMATED WALK <b>${number(result.estimated_walking_time_min, 2)} min</b></span><span>TRANSFER MARGIN <b>${number(result.transfer_time_margin_min, 2)} min</b></span></div>`;
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
    button.querySelector('span:first-child').textContent = 'Estimate connection';
  }
});

$('trainButton').addEventListener('click', startTraining);
$('refreshButton').addEventListener('click', refreshDashboard);
$('refreshData').addEventListener('click', () => { refreshDashboard(); showToast('Data and model metrics refreshed.'); });
$('resetForm').addEventListener('click', () => { $('predictionForm').reset(); });
refreshDashboard();
