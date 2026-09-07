"""
Module 3: Workload Prediction
Implements LSTM neural network, XGBoost regressors, and Weighted Ensemble Fusion.
"""

from .lstm_model import LSTMWorkloadModel
from .xgboost_model import XGBoostWorkloadModel
from .ensemble import WeightedEnsembleFusion
from .inference import WorkloadInferenceEngine

__all__ = [
    "LSTMWorkloadModel",
    "XGBoostWorkloadModel",
    "WeightedEnsembleFusion",
    "WorkloadInferenceEngine",
]
