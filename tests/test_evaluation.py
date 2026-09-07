import pytest
import numpy as np
from evaluation.metrics import calculate_forecast_metrics, calculate_system_metrics


def test_calculate_forecast_metrics():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([11.0, 19.0, 31.0, 39.0])

    m = calculate_forecast_metrics(y_true, y_pred)
    assert m["MSE"] == 1.0
    assert m["RMSE"] == 1.0
    assert m["MAE"] == 1.0
    assert m["R2"] > 0.95


def test_calculate_system_metrics_and_slo():
    # 10 requests: 2 requests exceed 500ms -> SLO violation = (2/10)*100 = 20.0%
    latencies = [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 600.0, 750.0]
    node_utils = {
        "w1": [45.0, 50.0],
        "w2": [55.0, 60.0],
    }

    sm = calculate_system_metrics(
        response_times_ms=latencies,
        node_utilizations_percent=node_utils,
        slo_latency_ms=500.0,
    )

    assert sm["total_requests"] == 10
    assert sm["slo_violations"] == 2
    assert sm["slo_violation_percentage"] == 20.0
    assert sm["avg_response_time_ms"] > 0
    assert sm["p95_response_time_ms"] > 500.0
