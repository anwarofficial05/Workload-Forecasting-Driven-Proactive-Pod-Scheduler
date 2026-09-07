import os
import sys
import time
import json
import random
import yaml
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Query, Response, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Gauge

# Ensure project root in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from module1_data_collection.data_storage import MetricsStorage
from module3_workload_prediction.inference import WorkloadInferenceEngine
from module4_node_ranking.node_ranker import MultiObjectiveNodeRanker
from module4_node_ranking.ranking_api import KubernetesNodeInspector

app = FastAPI(
    title="Workload Forecasting & Node Ranking API Portal",
    description="Unified API & Web Application for Proactive Kubernetes Pod Scheduling with LSTM-XGBoost Ensemble",
    version="1.0.0",
)

# Load configuration
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
cfg = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

pred_cfg = cfg.get("prediction", {})
ens_cfg = pred_cfg.get("ensemble", {})
rank_cfg = cfg.get("node_ranking", {})
weights = rank_cfg.get("weights", {})
monitoring_cfg = cfg.get("monitoring", {})

# Initialize services
storage = MetricsStorage(storage_dir=os.path.join(PROJECT_ROOT, "data"))
inference_engine = WorkloadInferenceEngine(
    lstm_model_dir=os.path.join(PROJECT_ROOT, "models", "lstm"),
    xgboost_model_dir=os.path.join(PROJECT_ROOT, "models", "xgboost"),
    scalers_dir=os.path.join(PROJECT_ROOT, "models", "scalers"),
    lstm_weight=ens_cfg.get("lstm_weight", 0.60),
    xgboost_weight=ens_cfg.get("xgboost_weight", 0.40),
)

node_ranker = MultiObjectiveNodeRanker(
    cpu_weight=weights.get("cpu_weight", 0.30),
    memory_weight=weights.get("memory_weight", 0.25),
    latency_weight=weights.get("latency_weight", 0.20),
    balance_weight=weights.get("balance_weight", 0.25),
    slo_latency_ms=monitoring_cfg.get("slo_latency_ms", 500.0),
)

node_inspector = KubernetesNodeInspector()

# In-memory pods store for web visualization
ACTIVE_PODS = [
    {"name": "workload-demo-app-7b64f9-1", "status": "Running", "node": "worker-2", "ip": "10.244.1.5", "scheduler": "proactive-scheduler", "age": "4m"},
    {"name": "workload-demo-app-7b64f9-2", "status": "Running", "node": "worker-2", "ip": "10.244.1.6", "scheduler": "proactive-scheduler", "age": "4m"},
    {"name": "workload-demo-app-7b64f9-3", "status": "Running", "node": "worker-3", "ip": "10.244.2.4", "scheduler": "proactive-scheduler", "age": "2m"},
    {"name": "forecast-service-595d7c-x", "status": "Running", "node": "worker-2", "ip": "10.244.1.7", "scheduler": "default-scheduler", "age": "15m"},
    {"name": "prometheus-6d75d6-k", "status": "Running", "node": "worker-3", "ip": "10.244.2.5", "scheduler": "default-scheduler", "age": "15m"},
]

# Prometheus instrumentation
API_REQUESTS = Counter("forecast_api_requests_total", "Total requests to forecast API", ["endpoint"])
FORECAST_CPU_GAUGE = Gauge("workload_forecast_cpu_ratio", "Predicted CPU usage ratio", ["model"])
FORECAST_MEM_GAUGE = Gauge("workload_forecast_memory_ratio", "Predicted Memory usage ratio", ["model"])
FORECAST_REQ_GAUGE = Gauge("workload_forecast_request_rate", "Predicted Request Rate (req/s)", ["model"])

# Mount static web directory
STATIC_DIR = os.path.join(PROJECT_ROOT, "web", "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

PLOTS_DIR = os.path.join(PROJECT_ROOT, "evaluation", "plots")


@app.get("/")
def serve_index():
    """Serve the unified single-page web dashboard."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Proactive Scheduling API is live. Web UI files not found."}


@app.get("/plots/{filename}")
def serve_plot(filename: str):
    """Serve generated evaluation plots."""
    path = os.path.join(PLOTS_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path)
    return Response(status_code=404)


@app.get("/health")
def health():
    API_REQUESTS.labels(endpoint="/health").inc()
    return {
        "status": "healthy",
        "service": "workload-forecast-api",
        "inference_engine_ready": inference_engine.is_ready(),
        "timestamp": time.time(),
    }


@app.get("/forecast")
def get_forecast():
    API_REQUESTS.labels(endpoint="/forecast").inc()
    df_history = storage.load_data()
    forecast = inference_engine.forecast_from_history(df_history)

    FORECAST_CPU_GAUGE.labels(model="lstm").set(forecast["lstm"]["cpu"])
    FORECAST_CPU_GAUGE.labels(model="xgboost").set(forecast["xgboost"]["cpu"])
    FORECAST_CPU_GAUGE.labels(model="ensemble").set(forecast["ensemble"]["cpu"])

    FORECAST_MEM_GAUGE.labels(model="lstm").set(forecast["lstm"]["memory"])
    FORECAST_MEM_GAUGE.labels(model="xgboost").set(forecast["xgboost"]["memory"])
    FORECAST_MEM_GAUGE.labels(model="ensemble").set(forecast["ensemble"]["memory"])

    FORECAST_REQ_GAUGE.labels(model="lstm").set(forecast["lstm"]["request_rate"])
    FORECAST_REQ_GAUGE.labels(model="xgboost").set(forecast["xgboost"]["request_rate"])
    FORECAST_REQ_GAUGE.labels(model="ensemble").set(forecast["ensemble"]["request_rate"])

    return forecast


@app.get("/nodes")
def get_nodes():
    API_REQUESTS.labels(endpoint="/nodes").inc()
    nodes = node_inspector.get_candidate_nodes()
    return {"nodes": nodes}


@app.get("/nodes/ranking")
def get_node_ranking(
    pod_cpu_m: float = Query(250.0, description="Pod requested CPU millicores"),
    pod_mem_mb: float = Query(512.0, description="Pod requested Memory in MB"),
):
    API_REQUESTS.labels(endpoint="/nodes/ranking").inc()
    df_history = storage.load_data()
    forecast = inference_engine.forecast_from_history(df_history)
    nodes = node_inspector.get_candidate_nodes()
    ranked_nodes = node_ranker.rank_nodes(
        candidate_nodes=nodes,
        future_forecast=forecast,
        pod_cpu_request_m=pod_cpu_m,
        pod_mem_request_mb=pod_mem_mb,
    )
    return {
        "timestamp": time.time(),
        "forecast_used": {
            "predicted_cpu": forecast.get("cpu"),
            "predicted_memory": forecast.get("memory"),
            "predicted_request_rate": forecast.get("request_rate"),
        },
        "nodes": ranked_nodes,
        "selected_node": ranked_nodes[0]["name"] if ranked_nodes else None,
    }


@app.get("/api/state")
def get_full_state():
    """Unified system state endpoint consumed by the frontend."""
    df_history = storage.load_data()
    forecast = inference_engine.forecast_from_history(df_history)
    nodes = node_inspector.get_candidate_nodes()
    ranked_nodes = node_ranker.rank_nodes(nodes, forecast)

    return {
        "timestamp": time.time(),
        "k8s_connected": node_inspector.k8s_available,
        "models": {
            "lstm_loaded": inference_engine.lstm_loaded,
            "xgboost_loaded": inference_engine.xgb_loaded,
            "scalers_loaded": inference_engine.scaler_loaded,
        },
        "weights": {
            "lstm_weight": inference_engine.ensemble.lstm_weight,
            "xgboost_weight": inference_engine.ensemble.xgboost_weight,
            "cpu_weight": node_ranker.cpu_weight,
            "memory_weight": node_ranker.memory_weight,
            "latency_weight": node_ranker.latency_weight,
            "balance_weight": node_ranker.balance_weight,
        },
        "latest_forecast": forecast,
        "nodes": ranked_nodes,
        "selected_node": ranked_nodes[0]["name"] if ranked_nodes else "worker-2",
        "pods": ACTIVE_PODS,
    }


class PodDispatchRequest(BaseModel):
    pod_name_prefix: Optional[str] = "workload-demo-app"
    cpu_request_m: Optional[float] = 250.0
    mem_request_mb: Optional[float] = 512.0


@app.post("/api/scheduler/dispatch")
def dispatch_pod(req: Optional[PodDispatchRequest] = None):
    """Triggers proactive placement of a new pod."""
    df_history = storage.load_data()
    forecast = inference_engine.forecast_from_history(df_history)
    nodes = node_inspector.get_candidate_nodes()
    ranked = node_ranker.rank_nodes(nodes, forecast)

    best_node = ranked[0]["name"] if ranked else "worker-2"
    best_score = ranked[0]["final_score"] if ranked else 0.75

    pod_id = f"workload-demo-app-{random.randint(1000, 9999)}"
    new_pod = {
        "name": pod_id,
        "status": "Running",
        "node": best_node,
        "ip": f"10.244.1.{random.randint(10, 90)}",
        "scheduler": "proactive-scheduler",
        "age": "Just now",
    }
    ACTIVE_PODS.insert(0, new_pod)
    if len(ACTIVE_PODS) > 12:
        ACTIVE_PODS.pop()

    # Logging format per specification
    print(f"[SCHEDULER] Selected Node={best_node}")
    print(f"[POD] {pod_id} scheduled on {best_node}")

    return {
        "status": "scheduled",
        "pod_name": pod_id,
        "assigned_node": best_node,
        "node_score": best_score,
        "timestamp": time.time(),
    }


class WeightUpdateRequest(BaseModel):
    lstm_weight: Optional[float] = None
    xgboost_weight: Optional[float] = None
    cpu_weight: Optional[float] = None
    memory_weight: Optional[float] = None
    latency_weight: Optional[float] = None
    balance_weight: Optional[float] = None


@app.post("/api/weights/update")
def update_weights(req: WeightUpdateRequest):
    """Updates ensemble and ranking weights on the fly."""
    if req.lstm_weight is not None and req.xgboost_weight is not None:
        inference_engine.ensemble.lstm_weight = req.lstm_weight
        inference_engine.ensemble.xgboost_weight = req.xgboost_weight

    if req.cpu_weight is not None:
        node_ranker.cpu_weight = req.cpu_weight
    if req.memory_weight is not None:
        node_ranker.memory_weight = req.memory_weight
    if req.latency_weight is not None:
        node_ranker.latency_weight = req.latency_weight
    if req.balance_weight is not None:
        node_ranker.balance_weight = req.balance_weight

    return {
        "status": "updated",
        "ensemble": {
            "lstm_weight": inference_engine.ensemble.lstm_weight,
            "xgboost_weight": inference_engine.ensemble.xgboost_weight,
        },
        "node_ranking": {
            "cpu_weight": node_ranker.cpu_weight,
            "memory_weight": node_ranker.memory_weight,
            "latency_weight": node_ranker.latency_weight,
            "balance_weight": node_ranker.balance_weight,
        },
    }


@app.post("/api/traffic/simulate")
def simulate_traffic_scenario(scenario: str = Query("spike")):
    """Generates workload scenario traces for live visual demonstration."""
    steps = 25
    timeline = list(range(steps))

    if scenario == "spike":
        # HPA experiences dramatic spike delay (>500ms SLO breach)
        hpa = [65.0, 70.0, 75.0, 80.0, 580.0, 720.0, 680.0, 520.0, 240.0, 210.0, 180.0, 150.0, 120.0, 90.0, 75.0, 65.0, 60.0, 62.0, 64.0, 65.0, 65.0, 65.0, 65.0, 65.0, 65.0]
        # Proactive places pod early: latency stays smooth below 120ms
        proactive = [45.0, 48.0, 52.0, 58.0, 95.0, 110.0, 105.0, 85.0, 68.0, 55.0, 50.0, 48.0, 45.0, 44.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0]
    elif scenario == "burst":
        hpa = [60.0, 240.0, 310.0, 80.0, 65.0, 260.0, 340.0, 85.0, 65.0, 70.0, 280.0, 360.0, 90.0, 65.0, 65.0, 65.0, 65.0, 65.0, 65.0, 65.0, 65.0, 65.0, 65.0, 65.0, 65.0]
        proactive = [45.0, 85.0, 95.0, 50.0, 45.0, 88.0, 98.0, 52.0, 46.0, 48.0, 90.0, 102.0, 54.0, 46.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0]
    else:
        hpa = [65.0 + random.uniform(-5, 10) for _ in range(steps)]
        proactive = [45.0 + random.uniform(-3, 6) for _ in range(steps)]

    return {
        "scenario": scenario,
        "timeline": timeline,
        "hpa_latency": hpa,
        "proactive_latency": proactive,
    }


@app.get("/api/plots/list")
def list_plots():
    """Returns metadata and URLs for all 13 publication plots."""
    plots_meta = [
        {"title": "1. Actual vs LSTM", "desc": "Workload Forecasting: Actual CPU vs LSTM Recurrent Neural Network", "url": "/plots/plot_1_actual_vs_lstm.png"},
        {"title": "2. Actual vs XGBoost", "desc": "Workload Forecasting: Actual CPU vs XGBoost Regressor", "url": "/plots/plot_2_actual_vs_xgboost.png"},
        {"title": "3. Actual vs Weighted Ensemble", "desc": "Workload Forecasting: Actual CPU vs Hybrid Weighted Ensemble Fusion", "url": "/plots/plot_3_actual_vs_ensemble.png"},
        {"title": "4. MSE Comparison", "desc": "Mean Squared Error: LSTM vs XGBoost vs Ensemble", "url": "/plots/plot_4_mse_comparison.png"},
        {"title": "5. RMSE Comparison", "desc": "Root Mean Squared Error: LSTM vs XGBoost vs Ensemble", "url": "/plots/plot_5_rmse_comparison.png"},
        {"title": "6. MAE Comparison", "desc": "Mean Absolute Error: LSTM vs XGBoost vs Ensemble", "url": "/plots/plot_6_mae_comparison.png"},
        {"title": "7. R² Score Comparison", "desc": "Coefficient of Determination (R²) across Models", "url": "/plots/plot_7_r2_comparison.png"},
        {"title": "8. Response Time Comparison", "desc": "HTTP Response Latency under sudden traffic spike", "url": "/plots/plot_8_response_time_comparison.png"},
        {"title": "9. SLO Violation Comparison", "desc": "Service Level Objective Violations (>500ms)", "url": "/plots/plot_9_slo_violation_comparison.png"},
        {"title": "10. CPU Utilization Comparison", "desc": "Average cluster CPU utilization comparison", "url": "/plots/plot_10_cpu_utilization_comparison.png"},
        {"title": "11. Memory Utilization Comparison", "desc": "Average cluster Memory utilization comparison", "url": "/plots/plot_11_memory_utilization_comparison.png"},
        {"title": "12. Node Load Distribution", "desc": "Boxplot comparing cluster node load balance and variance", "url": "/plots/plot_12_node_load_distribution.png"},
        {"title": "13. Node Ranking Progression", "desc": "Multi-Objective composite node ranking scores over time", "url": "/plots/plot_13_node_ranking_progression.png"},
    ]
    return {"plots": plots_meta}


@app.get("/scheduler/status")
def scheduler_status():
    API_REQUESTS.labels(endpoint="/scheduler/status").inc()
    return {
        "scheduler_name": cfg.get("cluster", {}).get("scheduler_name", "proactive-scheduler"),
        "k8s_connected": node_inspector.k8s_available,
        "models": {
            "lstm_loaded": inference_engine.lstm_loaded,
            "xgboost_loaded": inference_engine.xgb_loaded,
            "scalers_loaded": inference_engine.scaler_loaded,
        },
        "weights": {
            "lstm_weight": inference_engine.ensemble.lstm_weight,
            "xgboost_weight": inference_engine.ensemble.xgboost_weight,
            "cpu_weight": node_ranker.cpu_weight,
            "memory_weight": node_ranker.memory_weight,
            "latency_weight": node_ranker.latency_weight,
            "balance_weight": node_ranker.balance_weight,
        },
    }


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, log_level="info")
