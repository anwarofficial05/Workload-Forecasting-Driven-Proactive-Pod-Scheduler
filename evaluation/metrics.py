import logging
from typing import Dict, Any, List, Union, Optional
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger("evaluation.metrics")


def calculate_forecast_metrics(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
) -> Dict[str, float]:
    """
    Calculate core ML forecasting metrics:
    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R² (Coefficient of Determination)
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    if len(y_t) == 0 or len(y_p) == 0:
        return {"MSE": 0.0, "RMSE": 0.0, "MAE": 0.0, "R2": 0.0}

    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_t, y_p))
    try:
        r2 = float(r2_score(y_t, y_p))
    except Exception:
        r2 = 0.0

    return {
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
    }


def calculate_system_metrics(
    response_times_ms: List[float],
    node_utilizations_percent: Dict[str, List[float]],
    slo_latency_ms: float = 500.0,
    cpu_utilizations: Optional[List[float]] = None,
    memory_utilizations: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Calculate Kubernetes cluster system performance metrics:
    - Average response time
    - P95 response time
    - SLO violation percentage: (Violating Requests / Total Requests) * 100
    - Cluster-wide load balance (variance & standard deviation across nodes)
    - Average CPU and Memory utilization
    """
    latencies = np.asarray(response_times_ms, dtype=np.float64)
    total_requests = len(latencies)

    if total_requests == 0:
        return {
            "total_requests": 0,
            "avg_response_time_ms": 0.0,
            "p95_response_time_ms": 0.0,
            "slo_violations": 0,
            "slo_violation_percentage": 0.0,
            "load_balance_std": 0.0,
            "avg_cpu_percent": 0.0,
            "avg_memory_percent": 0.0,
        }

    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    # Formula from specification:
    # SLO Violation Percentage = (Violating Requests / Total Requests) * 100
    violating_requests = int(np.sum(latencies > slo_latency_ms))
    slo_violation_pct = float((violating_requests / total_requests) * 100.0)

    # Node load balance across nodes
    node_means = [float(np.mean(vals)) for vals in node_utilizations_percent.values() if len(vals) > 0]
    load_balance_std = float(np.std(node_means)) if len(node_means) > 1 else 0.0
    load_balance_var = float(np.var(node_means)) if len(node_means) > 1 else 0.0

    avg_cpu = float(np.mean(cpu_utilizations)) if cpu_utilizations else float(np.mean(node_means) if node_means else 0.0)
    avg_mem = float(np.mean(memory_utilizations)) if memory_utilizations else 50.0

    return {
        "total_requests": total_requests,
        "avg_response_time_ms": round(avg_latency, 2),
        "p95_response_time_ms": round(p95_latency, 2),
        "slo_violations": violating_requests,
        "slo_violation_percentage": round(slo_violation_pct, 2),
        "load_balance_std": round(load_balance_std, 4),
        "load_balance_variance": round(load_balance_var, 4),
        "avg_cpu_percent": round(avg_cpu, 2),
        "avg_memory_percent": round(avg_mem, 2),
    }
