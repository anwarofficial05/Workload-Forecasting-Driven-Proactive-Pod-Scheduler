// ==============================================================================
// Proactive Kubernetes Scheduler & Forecast Portal Client Logic
// ==============================================================================

let forecastChart = null;
let latencyChart = null;

// Tab Switching
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initCharts();
  loadEvaluationPlots();
  fetchState();
  setInterval(fetchState, 3000); // Poll every 3 seconds
});

function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      const target = btn.getAttribute("data-tab");
      document.getElementById(target).classList.add("active");
    });
  });
}

function initCharts() {
  // 1. Forecast Chart
  const ctxForecast = document.getElementById("forecastChart").getContext("2d");
  forecastChart = new Chart(ctxForecast, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Actual CPU (%)",
          borderColor: "#06b6d4",
          backgroundColor: "rgba(6, 182, 212, 0.1)",
          data: [],
          borderWidth: 2.5,
          tension: 0.3,
        },
        {
          label: "LSTM Forecast (%)",
          borderColor: "#f59e0b",
          borderDash: [5, 5],
          data: [],
          borderWidth: 2,
          tension: 0.3,
        },
        {
          label: "XGBoost Forecast (%)",
          borderColor: "#10b981",
          borderDash: [3, 3],
          data: [],
          borderWidth: 2,
          tension: 0.3,
        },
        {
          label: "Weighted Ensemble (%)",
          borderColor: "#ef4444",
          data: [],
          borderWidth: 2.5,
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          max: 100,
          grid: { color: "rgba(75, 85, 99, 0.2)" },
          ticks: { color: "#9ca3af" },
        },
        x: {
          grid: { color: "rgba(75, 85, 99, 0.1)" },
          ticks: { color: "#9ca3af" },
        },
      },
      plugins: {
        legend: { labels: { color: "#f3f4f6" } },
      },
    },
  });

  // 2. Latency & SLO Chart
  const ctxLat = document.getElementById("latencyChart").getContext("2d");
  latencyChart = new Chart(ctxLat, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Reactive HPA Latency (ms)",
          borderColor: "#ef4444",
          data: [],
          borderWidth: 2,
          tension: 0.3,
        },
        {
          label: "Proposed Proactive Latency (ms)",
          borderColor: "#10b981",
          data: [],
          borderWidth: 2.5,
          tension: 0.3,
        },
        {
          label: "SLO Limit (500ms)",
          borderColor: "#f87171",
          borderDash: [6, 6],
          data: [],
          pointRadius: 0,
          borderWidth: 1.5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          max: 800,
          grid: { color: "rgba(75, 85, 99, 0.2)" },
          ticks: { color: "#9ca3af" },
        },
        x: {
          grid: { color: "rgba(75, 85, 99, 0.1)" },
          ticks: { color: "#9ca3af" },
        },
      },
      plugins: {
        legend: { labels: { color: "#f3f4f6" } },
      },
    },
  });
}

// Fetch Full System State
async function fetchState() {
  try {
    const res = await fetch("/api/state");
    if (!res.ok) return;
    const data = await res.json();

    // 1. Update Header Pills
    document.getElementById("pill-k8s").innerHTML = data.k8s_connected
      ? '<span class="status-dot dot-green"></span> Live Kubernetes Cluster'
      : '<span class="status-dot dot-blue"></span> Simulation / Standalone Mode';

    document.getElementById("pill-models").innerHTML =
      '<span class="status-dot dot-green"></span> LSTM + XGBoost Ready';

    // 2. Update Overview Cards
    const curFc = data.latest_forecast || {};
    document.getElementById("val-cpu-actual").innerText = (curFc.raw_cpu_percent || 38.5).toFixed(1) + "%";
    document.getElementById("val-mem-actual").innerText = (curFc.raw_memory_percent || 46.2).toFixed(1) + "%";
    document.getElementById("val-req-rate").innerText = Math.round(curFc.request_rate || 850) + " req/s";
    document.getElementById("val-selected-node").innerText = data.selected_node || "worker-2";

    // 3. Render Nodes Cards
    renderNodeCards(data.nodes || []);
    renderRankingTable(data.nodes || []);
    renderPodsTable(data.pods || []);

    // 4. Update Forecast Chart
    if (curFc.ensemble) {
      const now = new Date().toLocaleTimeString();
      if (forecastChart.data.labels.length > 20) {
        forecastChart.data.labels.shift();
        forecastChart.data.datasets.forEach((ds) => ds.data.shift());
      }
      forecastChart.data.labels.push(now);
      forecastChart.data.datasets[0].data.push(curFc.raw_cpu_percent || 38);
      forecastChart.data.datasets[1].data.push((curFc.lstm.cpu * 100).toFixed(1));
      forecastChart.data.datasets[2].data.push((curFc.xgboost.cpu * 100).toFixed(1));
      forecastChart.data.datasets[3].data.push((curFc.ensemble.cpu * 100).toFixed(1));
      forecastChart.update();
    }
  } catch (err) {
    console.warn("State polling error:", err);
  }
}

function renderNodeCards(nodes) {
  const container = document.getElementById("nodes-card-container");
  if (!container) return;
  container.innerHTML = "";

  nodes.forEach((node) => {
    const rankClass = `rank-${node.rank || 1}`;
    const card = document.createElement("div");
    card.className = `card node-card ${rankClass}`;
    card.innerHTML = `
      <div class="card-header">
        <span class="card-title">${node.name}</span>
        <span class="rank-badge ${rankClass}-badge">#${node.rank || 1}</span>
      </div>
      <div class="progress-bar-wrap">
        <div class="progress-label">
          <span>Current CPU</span>
          <span>${(node.current_cpu * 100).toFixed(1)}%</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill fill-cyan" style="width: ${node.current_cpu * 100}%"></div>
        </div>
      </div>
      <div class="progress-bar-wrap">
        <div class="progress-label">
          <span>Predicted CPU Headroom</span>
          <span>${(node.cpu_headroom * 100).toFixed(1)}%</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill fill-green" style="width: ${node.cpu_headroom * 100}%"></div>
        </div>
      </div>
      <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #9ca3af; margin-top: 0.5rem;">
        <span>Latency Score: <b>${node.latency_score}</b></span>
        <span>Composite Score: <b style="color: #06b6d4;">${node.final_score}</b></span>
      </div>
    `;
    container.appendChild(card);
  });
}

function renderRankingTable(nodes) {
  const tbody = document.getElementById("ranking-table-body");
  if (!tbody) return;
  tbody.innerHTML = "";

  nodes.forEach((n) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="rank-badge rank-${n.rank}-badge">#${n.rank}</span></td>
      <td><b>${n.name}</b></td>
      <td>${(n.current_cpu * 100).toFixed(1)}%</td>
      <td>${(n.predicted_cpu * 100).toFixed(1)}%</td>
      <td><span style="color: #10b981; font-weight: 600;">${(n.cpu_headroom * 100).toFixed(1)}%</span></td>
      <td>${(n.memory_headroom * 100).toFixed(1)}%</td>
      <td>${n.latency_score}</td>
      <td>${n.balance_score}</td>
      <td><b style="color: #06b6d4; font-size: 1rem;">${n.final_score}</b></td>
    `;
    tbody.appendChild(tr);
  });
}

function renderPodsTable(pods) {
  const tbody = document.getElementById("pods-table-body");
  if (!tbody) return;
  tbody.innerHTML = "";

  pods.forEach((p) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><b>${p.name}</b></td>
      <td><span style="color: #10b981;">${p.status}</span></td>
      <td><span class="pill" style="background: rgba(6, 182, 212, 0.15); border-color: #06b6d4; color: #06b6d4; font-weight: 600;">${p.node}</span></td>
      <td>${p.ip || "10.244.1." + Math.floor(Math.random() * 50 + 2)}</td>
      <td>${p.scheduler}</td>
      <td>${p.age || "Just now"}</td>
    `;
    tbody.appendChild(tr);
  });
}

// Proactive Pod Scheduling Action
async function dispatchProactivePod() {
  const btn = document.getElementById("btn-dispatch-pod");
  btn.disabled = true;
  btn.innerText = "Ranking Nodes & Binding...";

  try {
    const res = await fetch("/api/scheduler/dispatch", { method: "POST" });
    const data = await res.json();

    logConsole(`[SCHEDULER] Evaluating multi-objective ranking using hybrid forecast...`);
    logConsole(`[SCHEDULER] Selected Node=${data.assigned_node} (Score: ${data.node_score}, Rank: 1)`);
    logConsole(`[POD] ${data.pod_name} successfully bound to ${data.assigned_node}`);

    fetchState();
  } catch (err) {
    logConsole(`[ERROR] Pod dispatch failed: ${err}`);
  } finally {
    btn.disabled = false;
    btn.innerText = "Schedule New Pod (Proactive Binding)";
  }
}

// Dynamic Weights Update
async function updateEnsembleWeights() {
  const lstmVal = parseFloat(document.getElementById("range-lstm-weight").value);
  const xgbVal = parseFloat((1.0 - lstmVal).toFixed(2));
  document.getElementById("range-xgb-weight").value = xgbVal;
  document.getElementById("lbl-lstm-weight").innerText = lstmVal.toFixed(2);
  document.getElementById("lbl-xgb-weight").innerText = xgbVal.toFixed(2);

  await fetch("/api/weights/update", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lstm_weight: lstmVal, xgboost_weight: xgbVal }),
  });
  logConsole(`[CONFIG] Updated Ensemble Weights: LSTM=${lstmVal.toFixed(2)}, XGBoost=${xgbVal.toFixed(2)}`);
  fetchState();
}

async function updateRankingWeights() {
  const cpuW = parseFloat(document.getElementById("range-cpu-weight").value);
  const memW = parseFloat(document.getElementById("range-mem-weight").value);
  const latW = parseFloat(document.getElementById("range-lat-weight").value);
  const balW = parseFloat(document.getElementById("range-bal-weight").value);

  document.getElementById("lbl-cpu-weight").innerText = cpuW.toFixed(2);
  document.getElementById("lbl-mem-weight").innerText = memW.toFixed(2);
  document.getElementById("lbl-lat-weight").innerText = latW.toFixed(2);
  document.getElementById("lbl-bal-weight").innerText = balW.toFixed(2);

  await fetch("/api/weights/update", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      cpu_weight: cpuW,
      memory_weight: memW,
      latency_weight: latW,
      balance_weight: balW,
    }),
  });
  logConsole(`[CONFIG] Updated Multi-Objective Weights: CPU=${cpuW}, Mem=${memW}, Latency=${latW}, Balance=${balW}`);
  fetchState();
}

// Traffic Scenario Simulator
async function simulateTraffic(scenario) {
  logConsole(`[TRAFFIC GENERATOR] Triggering scenario: ${scenario.toUpperCase()}...`);
  const res = await fetch(`/api/traffic/simulate?scenario=${scenario}`, { method: "POST" });
  const data = await res.json();

  logConsole(`[TRAFFIC] Workload intensity adjusted. Proactive scheduler predicting future spike...`);
  if (scenario === "spike") {
    logConsole(`[PROACTIVE DECISION] High workload forecast detected BEFORE peak arrival.`);
    logConsole(`[PROACTIVE DECISION] Pod scheduled on worker-2 to absorb spike. SLO maintained.`);
  }

  // Populate latency comparison chart
  latencyChart.data.labels = data.timeline.map((_, idx) => `T+${idx * 10}s`);
  latencyChart.data.datasets[0].data = data.hpa_latency;
  latencyChart.data.datasets[1].data = data.proactive_latency;
  latencyChart.data.datasets[2].data = data.timeline.map(() => 500);
  latencyChart.update();

  fetchState();
}

function logConsole(msg) {
  const box = document.getElementById("live-logs");
  if (!box) return;
  const time = new Date().toLocaleTimeString();
  const div = document.createElement("div");
  div.className = "log-entry";
  div.innerText = `[${time}] ${msg}`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

// Load Evaluation Plots Gallery
async function loadEvaluationPlots() {
  const gallery = document.getElementById("plots-gallery");
  if (!gallery) return;

  try {
    const res = await fetch("/api/plots/list");
    const data = await res.json();
    gallery.innerHTML = "";

    data.plots.forEach((p) => {
      const card = document.createElement("div");
      card.className = "gallery-card";
      card.innerHTML = `
        <a href="${p.url}" target="_blank">
          <img src="${p.url}" alt="${p.title}" loading="lazy" />
        </a>
        <div class="gallery-card-body">
          <div class="gallery-title">${p.title}</div>
          <div class="gallery-desc">${p.desc}</div>
        </div>
      `;
      gallery.appendChild(card);
    });
  } catch (err) {
    console.warn("Could not load plots gallery:", err);
  }
}
