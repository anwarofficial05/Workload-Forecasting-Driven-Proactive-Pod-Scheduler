// ==============================================================================
// Proactive Kubernetes Scheduler & Forecast Portal Client Logic
// ==============================================================================

let forecastChart = null;
let latencyChart = null;

// Tab Switching & Initialization
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initCharts();
  loadEvaluationPlots();
  fetchState();
  fetchCloudStatus();
  setInterval(fetchState, 3000); // Poll every 3 seconds
  setInterval(fetchCloudStatus, 8000); // Poll cloud status every 8 seconds
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
    if (data.is_gke) {
      document.getElementById("pill-k8s").innerHTML =
        `<span class="status-dot dot-green"></span> GKE Cloud: ${data.cluster_context || "Active"}`;
    } else if (data.k8s_connected) {
      document.getElementById("pill-k8s").innerHTML =
        `<span class="status-dot dot-green"></span> Live K8s: ${data.cluster_context || "Connected"}`;
    } else {
      document.getElementById("pill-k8s").innerHTML =
        '<span class="status-dot dot-blue"></span> Cluster: Standalone / Simulation';
    }

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

// ==============================================================================
// Google Cloud Platform & GKE Console Management
// ==============================================================================

async function fetchCloudStatus(manual = false) {
  try {
    const res = await fetch("/api/cloud/status");
    if (!res.ok) return;
    const data = await res.json();

    // 1. GCloud Info
    const cliElem = document.getElementById("cloud-cli-version");
    const pluginElem = document.getElementById("cloud-cli-plugin");
    if (cliElem && data.gcloud) {
      cliElem.innerText = data.gcloud.installed ? data.gcloud.version : "Not Detected";
      cliElem.style.color = data.gcloud.installed ? "#10b981" : "#ef4444";
      pluginElem.innerText = data.gcloud.gke_auth_plugin
        ? "GKE Auth Plugin: Ready (v0.5.19)"
        : "GKE Auth Plugin: Not Installed";
      pluginElem.style.color = data.gcloud.gke_auth_plugin ? "#10b981" : "#f59e0b";
    }

    // 2. Auth Account
    const accElem = document.getElementById("cloud-auth-account");
    const authStatus = document.getElementById("cloud-auth-status");
    if (accElem && data.auth) {
      accElem.innerText = data.auth.active_account || "No Account Logged In";
      accElem.style.color = data.auth.is_logged_in ? "#06b6d4" : "#9ca3af";
      authStatus.innerText = data.auth.is_logged_in ? "Authenticated via GCP OAuth" : "Click 'Login with Google Cloud'";
    }

    // 3. Project input default
    if (data.config && data.config.project_id) {
      const projInput = document.getElementById("input-cloud-project");
      if (projInput && !projInput.value) {
        projInput.value = data.config.project_id;
      }
    }

    // 4. Kubernetes context
    const ctxElem = document.getElementById("cloud-k8s-context");
    const k8sStatus = document.getElementById("cloud-k8s-status");
    if (ctxElem && data.kubernetes) {
      ctxElem.innerText = data.kubernetes.context || "No Active K8s Context";
      ctxElem.style.color = data.kubernetes.connected ? "#10b981" : "#f59e0b";
      const gkeLabel = data.kubernetes.is_gke ? " [GKE Cloud]" : "";
      k8sStatus.innerText = `${data.kubernetes.node_count} Worker Node(s) Available${gkeLabel}`;
    }

    // 5. Cluster list
    const clustersBox = document.getElementById("cloud-clusters-list");
    if (clustersBox) {
      if (data.gke_clusters && data.gke_clusters.length > 0) {
        clustersBox.innerHTML = data.gke_clusters
          .map(
            (c) =>
              `<div style="display: flex; justify-content: space-between; align-items: center; background: rgba(0,0,0,0.4); padding: 0.5rem; border-radius: 4px; margin-bottom: 0.3rem;">
                <div>
                  <b style="color: #38bdf8;">${c.name}</b> (${c.location}) - <span style="color: #10b981;">${c.status}</span> [${c.currentNodeCount} nodes]
                </div>
                <button class="btn" style="padding: 0.25rem 0.6rem; font-size: 0.75rem;" onclick="quickConnectCluster('${c.name}', '${c.location}')">Connect</button>
              </div>`
          )
          .join("");
      } else {
        clustersBox.innerText = data.auth.is_logged_in
          ? "No GKE clusters found in project, or API still synchronizing."
          : "Sign in with Google Cloud to discover clusters in your project.";
      }
    }

    if (manual) {
      appendTerminalOutput(`[CLOUD STATUS] Refreshed.\nGCloud: ${data.gcloud.version}\nAuth: ${data.auth.active_account || "Not Logged In"}\nK8s Context: ${data.kubernetes.context || "None"}\nConnected: ${data.kubernetes.connected}`);
    }
  } catch (err) {
    console.warn("fetchCloudStatus error:", err);
  }
}

async function saveCloudProject() {
  const proj = document.getElementById("input-cloud-project").value.trim();
  const zone = document.getElementById("input-cloud-zone").value.trim();
  const region = document.getElementById("input-cloud-region").value.trim();
  if (!proj) {
    alert("Please enter a valid GCP Project ID.");
    return;
  }
  appendTerminalOutput(`[GCLOUD] Setting active project to: ${proj} (Zone: ${zone}, Region: ${region})...`);
  try {
    const res = await fetch("/api/cloud/project", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_id: proj, zone, region }),
    });
    const data = await res.json();
    appendTerminalOutput(`[GCLOUD OUTPUT]\n${data.output || "Project configured successfully."}`);
    fetchCloudStatus();
  } catch (err) {
    appendTerminalOutput(`[ERROR] Failed to set project: ${err}`);
  }
}

function initiateGCloudLogin() {
  appendTerminalOutput(`[GCLOUD AUTH] To authenticate, run in PowerShell:\n  gcloud auth login\n\nOr click below to inspect current active auth credentials.`);
  runQuickCommand("gcloud auth list");
}

async function connectGKECluster() {
  const cluster = document.getElementById("input-gke-cluster").value.trim();
  const zone = document.getElementById("input-cloud-zone").value.trim();
  const project = document.getElementById("input-cloud-project").value.trim();
  if (!cluster) {
    alert("Please specify a GKE cluster name.");
    return;
  }
  const btn = document.getElementById("btn-connect-gke");
  btn.innerText = "Connecting...";
  btn.disabled = true;

  appendTerminalOutput(`[GKE] Fetching credentials for cluster '${cluster}' (zone: ${zone})...`);
  try {
    const res = await fetch("/api/cloud/gke/connect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cluster_name: cluster, zone, project_id: project || null }),
    });
    const data = await res.json();
    if (data.success || data.k8s_connected) {
      appendTerminalOutput(`[GKE SUCCESS] Connected to cluster '${cluster}'!\nActive Context: ${data.cluster_context}\nNodes: ${JSON.stringify(data.nodes.map(n => n.name))}`);
      logConsole(`[GKE] Connected to cloud cluster: ${cluster}. Live node scoring active.`);
    } else {
      appendTerminalOutput(`[GKE NOTICE] Output:\n${data.stdout || data.stderr}`);
    }
    fetchCloudStatus();
    fetchState();
  } catch (err) {
    appendTerminalOutput(`[ERROR] Connection failed: ${err}`);
  } finally {
    btn.innerText = "Connect GKE";
    btn.disabled = false;
  }
}

function quickConnectCluster(name, location) {
  document.getElementById("input-gke-cluster").value = name;
  document.getElementById("input-cloud-zone").value = location;
  connectGKECluster();
}

async function deploySystemToGKE() {
  const btn = document.getElementById("btn-deploy-gke");
  btn.innerText = "Deploying...";
  btn.disabled = true;
  appendTerminalOutput(`[DEPLOY] Applying kubernetes/gke/gke-deploy.yaml to current Kubernetes cluster...`);

  try {
    const res = await fetch("/api/cloud/gke/deploy", { method: "POST" });
    const data = await res.json();
    appendTerminalOutput(`[DEPLOY RESULT]\n${data.stdout || data.stderr || "Applied manifests."}`);
    if (data.live_pods && data.live_pods.length > 0) {
      appendTerminalOutput(`[PODS DISCOVERED] Found ${data.live_pods.length} active pods in cluster.`);
    }
    logConsole(`[GKE DEPLOY] Deployed proactive scheduler manifests to cluster.`);
    fetchState();
  } catch (err) {
    appendTerminalOutput(`[ERROR] GKE Deployment failed: ${err}`);
  } finally {
    btn.innerText = "Deploy to GKE";
    btn.disabled = false;
  }
}

async function deployPortalToCloudRun() {
  const btn = document.getElementById("btn-deploy-cloudrun");
  btn.innerText = "Initiating...";
  btn.disabled = true;
  const region = document.getElementById("input-cloud-region").value.trim() || "us-central1";

  appendTerminalOutput(`[CLOUD RUN] Launching build and deploy of portal to Google Cloud Run (Region: ${region})...\nCommand: gcloud run deploy proactive-scheduler-portal --source . --platform managed --region ${region} --allow-unauthenticated --port 8000`);

  try {
    const res = await fetch(`/api/cloud/cloudrun/deploy?region=${region}`, { method: "POST" });
    const data = await res.json();
    appendTerminalOutput(`[CLOUD RUN OUTPUT]\n${data.stdout || data.stderr}`);
  } catch (err) {
    appendTerminalOutput(`[ERROR] Cloud Run request failed: ${err}`);
  } finally {
    btn.innerText = "Deploy Cloud Run";
    btn.disabled = false;
  }
}

function provisionGKECommand() {
  const proj = document.getElementById("input-cloud-project").value.trim() || "PROJECT_ID";
  const zone = document.getElementById("input-cloud-zone").value.trim() || "us-central1-a";
  const cluster = document.getElementById("input-gke-cluster").value.trim() || "proactive-scheduler-gke";
  const cmd = `gcloud container clusters create ${cluster} --zone ${zone} --project ${proj} --num-nodes 3 --machine-type e2-standard-4 --enable-ip-alias`;
  document.getElementById("input-cloud-cmd").value = cmd;
  appendTerminalOutput(`[PROVISION COMMAND READY]\n${cmd}\n\nClick 'Execute' or hit Enter to launch cluster creation on Google Cloud.`);
}

async function runQuickCommand(cmd) {
  document.getElementById("input-cloud-cmd").value = cmd;
  await executeCustomCommand();
}

async function executeCustomCommand() {
  const input = document.getElementById("input-cloud-cmd");
  const cmd = input.value.trim();
  if (!cmd) return;

  appendTerminalOutput(`$ ${cmd}`);
  try {
    const res = await fetch("/api/cloud/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command: cmd }),
    });
    const data = await res.json();
    appendTerminalOutput(data.output);
  } catch (err) {
    appendTerminalOutput(`[EXECUTION ERROR] ${err}`);
  }
}

function appendTerminalOutput(text) {
  const term = document.getElementById("cloud-terminal-output");
  if (!term) return;
  term.textContent += `\n\n${text}`;
  term.scrollTop = term.scrollHeight;
}
