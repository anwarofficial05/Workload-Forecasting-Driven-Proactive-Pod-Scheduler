"""
Evaluation Package
Calculates ML forecasting metrics (MSE, RMSE, MAE, R²) and Kubernetes system metrics
(latency, SLO violations, load balance variance); produces publication-ready Matplotlib comparison figures.
"""

from .metrics import calculate_forecast_metrics, calculate_system_metrics
from .evaluate import run_comparative_evaluation
from .plots import generate_all_evaluation_plots

__all__ = [
    "calculate_forecast_metrics",
    "calculate_system_metrics",
    "run_comparative_evaluation",
    "generate_all_evaluation_plots",
]
