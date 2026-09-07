import time
import logging
from typing import Dict, Any, Optional
from prometheus_client import (
    start_http_server,
    Gauge,
    Counter,
    Histogram,
    REGISTRY,
)

logger = logging.getLogger("module6.metrics")

# Workload Forecast Predictions
FORECAST_CPU = Gauge(
    "workload_prediction_cpu_ratio",
    "Predicted CPU utilization ratio (0-1)",
    ["model"],
)
FORECAST_MEMORY = Gauge(
    "workload_prediction_memory_ratio",
    "Predicted Memory utilization ratio (0-1)",
    ["model"],
)
FORECAST_REQUEST_RATE = Gauge(
    "workload_prediction_request_rate",
    "Predicted Request Rate in requests/second",
    ["model"],
)

# Forecast Residual / Error
FORECAST_ERROR = Gauge(
    "workload_forecast_absolute_error",
    "Absolute forecast error (|Actual - Forecast|)",
    ["metric", "model"],
)

# Node Ranking Metrics
NODE_RANKING_SCORE = Gauge(
    "node_ranking_score",
    "Composite multi-objective ranking score for Kubernetes worker node",
    ["node_name"],
)
NODE_SELECTED_COUNT = Counter(
    "proactive_scheduler_node_selected_total",
    "Total count of scheduling placements on node",
    ["node_name"],
)

# Cluster Actual Telemetry
ACTUAL_CPU = Gauge("cluster_actual_cpu_percent", "Actual measured cluster CPU usage %")
ACTUAL_MEMORY = Gauge("cluster_actual_memory_percent", "Actual measured cluster Memory usage %")
ACTUAL_REQUEST_RATE = Gauge("cluster_actual_request_rate", "Actual measured cluster request rate (req/s)")
ACTUAL_LATENCY = Gauge("cluster_actual_p95_latency_seconds", "Actual cluster P95 latency in seconds")

# SLO Telemetry
SLO_VIOLATIONS = Counter(
    "cluster_slo_violations_total",
    "Total requests exceeding SLO threshold (500ms)",
)
TOTAL_EVALUATED_REQUESTS = Counter(
    "cluster_requests_evaluated_total",
    "Total requests evaluated for SLO compliance",
)


class ProactiveMetricsExporter:
    """
    Prometheus metrics exporter for proactive scheduling telemetry.
    Can run as an embedded server or publish to existing registry.
    """

    def __init__(self, port: int = 8001):
        self.port = port
        self.started = False

    def start_server(self):
        if not self.started:
            try:
                start_http_server(self.port)
                self.started = True
                logger.info(f"Proactive Prometheus metrics exporter listening on port {self.port}")
            except Exception as exc:
                logger.warning(f"Metrics server already running or port in use: {exc}")

    def update_forecasts(
        self,
        lstm_fc: Dict[str, float],
        xgb_fc: Dict[str, float],
        ens_fc: Dict[str, float],
        actual: Optional[Dict[str, float]] = None,
    ):
        """Publish updated model forecasts and calculate real-time forecast errors."""
        for model_name, data in [("lstm", lstm_fc), ("xgboost", xgb_fc), ("ensemble", ens_fc)]:
            if "cpu" in data:
                FORECAST_CPU.labels(model=model_name).set(data["cpu"])
            if "memory" in data:
                FORECAST_MEMORY.labels(model=model_name).set(data["memory"])
            if "request_rate" in data:
                FORECAST_REQUEST_RATE.labels(model=model_name).set(data["request_rate"])

        # Calculate forecast error if actuals are available
        if actual:
            act_cpu = actual.get("cpu", 0.0)
            act_mem = actual.get("memory", 0.0)
            for model_name, data in [("lstm", lstm_fc), ("xgboost", xgb_fc), ("ensemble", ens_fc)]:
                if "cpu" in data:
                    err = abs(act_cpu - data["cpu"])
                    FORECAST_ERROR.labels(metric="cpu", model=model_name).set(err)
                if "memory" in data:
                    err = abs(act_mem - data["memory"])
                    FORECAST_ERROR.labels(metric="memory", model=model_name).set(err)

    def update_node_ranking(self, ranked_nodes: list):
        """Update node ranking gauges."""
        for node in ranked_nodes:
            name = node.get("name", "unknown")
            score = float(node.get("final_score", 0.0))
            NODE_RANKING_SCORE.labels(node_name=name).set(score)

    def record_pod_placement(self, node_name: str):
        """Increment placement count for selected node."""
        NODE_SELECTED_COUNT.labels(node_name=node_name).inc()

    def record_slo_check(self, latency_seconds: float, slo_threshold_seconds: float = 0.5):
        """Track SLO compliance."""
        TOTAL_EVALUATED_REQUESTS.inc()
        if latency_seconds > slo_threshold_seconds:
            SLO_VIOLATIONS.inc()
