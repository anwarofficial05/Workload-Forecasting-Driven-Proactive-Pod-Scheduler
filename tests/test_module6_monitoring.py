import os
import json
import pytest
from module6_monitoring.metrics import ProactiveMetricsExporter


def test_metrics_exporter_updates():
    exporter = ProactiveMetricsExporter()
    lstm_fc = {"cpu": 0.65, "memory": 0.55, "request_rate": 1400.0}
    xgb_fc = {"cpu": 0.70, "memory": 0.58, "request_rate": 1500.0}
    ens_fc = {"cpu": 0.67, "memory": 0.56, "request_rate": 1440.0}
    actual = {"cpu": 0.68, "memory": 0.57}

    exporter.update_forecasts(lstm_fc, xgb_fc, ens_fc, actual=actual)
    exporter.record_pod_placement("worker-2")
    exporter.record_slo_check(latency_seconds=0.650, slo_threshold_seconds=0.500)
    exporter.record_slo_check(latency_seconds=0.120, slo_threshold_seconds=0.500)


def test_grafana_dashboard_json_structure():
    dashboard_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "module6_monitoring",
        "grafana",
        "dashboards",
        "proactive_scheduler_dashboard.json",
    )
    assert os.path.exists(dashboard_path)
    with open(dashboard_path, "r") as f:
        dash = json.load(f)

    assert "panels" in dash
    panels = dash["panels"]
    assert len(panels) == 14 # Exact 14 panels required

    # Verify panel titles match required topics
    titles = [p["title"] for p in panels]
    assert any("CPU Usage" in t for t in titles)
    assert any("Memory Usage" in t for t in titles)
    assert any("Request Rate" in t for t in titles)
    assert any("Response Latency" in t for t in titles)
    assert any("Pod Count" in t for t in titles)
    assert any("Node Utilization" in t for t in titles)
    assert any("Actual vs LSTM" in t for t in titles)
    assert any("Actual vs XGBoost" in t for t in titles)
    assert any("Actual vs Ensemble" in t for t in titles)
    assert any("Node Ranking" in t for t in titles)
    assert any("Selected Node" in t for t in titles)
    assert any("SLO Violations" in t for t in titles)
    assert any("Resource Utilization" in t for t in titles)
    assert any("Forecast Absolute Error" in t for t in titles)
