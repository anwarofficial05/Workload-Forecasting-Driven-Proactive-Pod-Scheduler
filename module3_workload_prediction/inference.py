import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from .lstm_model import LSTMWorkloadModel
from .xgboost_model import XGBoostWorkloadModel
from .ensemble import WeightedEnsembleFusion
from module2_data_preprocessing.scalers import ScalerManager
from module2_data_preprocessing.feature_engineering import FeatureEngineer

logger = logging.getLogger("module3.inference")


class WorkloadInferenceEngine:
    """
    Online inference engine for real-time Kubernetes workload forecasting.
    Loads persisted LSTM, XGBoost, and Scaler artifacts and performs
    Weighted Ensemble Fusion without retraining during inference.
    """

    def __init__(
        self,
        lstm_model_dir: str = "models/lstm",
        xgboost_model_dir: str = "models/xgboost",
        scalers_dir: str = "models/scalers",
        lstm_weight: float = 0.60,
        xgboost_weight: float = 0.40,
        sequence_length: int = 5,
        target_columns: Optional[List[str]] = None,
        feature_columns: Optional[List[str]] = None,
    ):
        self.sequence_length = sequence_length
        self.target_columns = target_columns or ["cpu_usage", "memory_usage", "request_rate"]
        self.feature_columns = feature_columns or ["cpu_usage", "memory_usage", "request_rate", "response_latency"]

        self.scaler_mgr = ScalerManager(scalers_dir=scalers_dir)
        self.scaler_loaded = self.scaler_mgr.load()

        self.lstm = LSTMWorkloadModel(
            sequence_length=sequence_length,
            n_features=len(self.feature_columns),
            n_targets=len(self.target_columns),
            model_dir=lstm_model_dir,
        )
        self.lstm_loaded = self.lstm.load()

        self.xgb = XGBoostWorkloadModel(
            model_dir=xgboost_model_dir,
            target_names=self.target_columns,
        )
        self.xgb_loaded = self.xgb.load()

        self.ensemble = WeightedEnsembleFusion(
            lstm_weight=lstm_weight,
            xgboost_weight=xgboost_weight,
        )
        self.fe = FeatureEngineer()

    def is_ready(self) -> bool:
        """Check if models and scalers are loaded and ready for inference."""
        return self.scaler_loaded and (self.lstm_loaded or self.xgb_loaded)

    def forecast_from_history(self, history_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Produce future workload forecast given the most recent history records.
        Requires at least sequence_length (5) recent rows.
        """
        if history_df.empty or len(history_df) < self.sequence_length:
            logger.warning("Insufficient history points for forecasting, using documented safe fallback")
            return self._safe_fallback_forecast(history_df)

        # Take last N points
        recent = history_df.iloc[-max(self.sequence_length + 5, len(history_df)) :].copy()

        # 1. Feature Engineering for XGBoost
        engineered = self.fe.create_features(recent)
        xgb_cols = self.fe.get_xgboost_feature_columns()
        latest_xgb_features = engineered.iloc[[-1]][xgb_cols].values

        # 2. Sequence preparation for LSTM
        raw_feat_window = recent.iloc[-self.sequence_length :][self.feature_columns].values
        if self.scaler_mgr.fitted:
            scaled_window = self.scaler_mgr.transform_features(raw_feat_window)
        else:
            scaled_window = raw_feat_window / 100.0 # simple fallback normalization

        # 3. Model Inference
        # LSTM Prediction
        if self.lstm_loaded:
            lstm_pred_scaled = self.lstm.predict(scaled_window)
            if self.scaler_mgr.fitted:
                lstm_pred = self.scaler_mgr.inverse_transform_targets(lstm_pred_scaled).ravel()
            else:
                lstm_pred = lstm_pred_scaled.ravel() * 100.0
        else:
            # Fallback to recent mean
            lstm_pred = raw_feat_window[:, :len(self.target_columns)].mean(axis=0)

        # XGBoost Prediction
        if self.xgb_loaded:
            xgb_pred = self.xgb.predict(latest_xgb_features).ravel()
        else:
            xgb_pred = raw_feat_window[:, :len(self.target_columns)].mean(axis=0)

        # 4. Weighted Ensemble Fusion
        ensemble_pred = self.ensemble.fuse(lstm_pred, xgb_pred).ravel()

        # Format and bound values
        cpu_val = max(0.0, min(100.0, float(ensemble_pred[0])))
        mem_val = max(0.0, min(100.0, float(ensemble_pred[1])))
        req_val = max(0.0, float(ensemble_pred[2]))

        l_cpu = round(float(lstm_pred[0]) / 100.0, 2)
        x_cpu = round(float(xgb_pred[0]) / 100.0, 2)
        e_cpu = round(cpu_val / 100.0, 2)

        # Mandatory logging format per project specification:
        # [FORECAST] LSTM CPU=0.65 XGBoost CPU=0.71 Ensemble CPU=0.67
        log_line = f"[FORECAST] LSTM CPU={l_cpu} XGBoost CPU={x_cpu} Ensemble CPU={e_cpu}"
        print(log_line)
        logger.info(log_line)

        return {
            "timestamp": float(recent["timestamp"].iloc[-1]) if "timestamp" in recent.columns else 0.0,
            "cpu": round(cpu_val / 100.0, 4), # Normalized 0-1
            "memory": round(mem_val / 100.0, 4), # Normalized 0-1
            "request_rate": round(req_val, 2),
            "raw_cpu_percent": round(cpu_val, 2),
            "raw_memory_percent": round(mem_val, 2),
            "lstm": {
                "cpu": round(float(lstm_pred[0]) / 100.0, 4),
                "memory": round(float(lstm_pred[1]) / 100.0, 4),
                "request_rate": round(float(lstm_pred[2]), 2),
            },
            "xgboost": {
                "cpu": round(float(xgb_pred[0]) / 100.0, 4),
                "memory": round(float(xgb_pred[1]) / 100.0, 4),
                "request_rate": round(float(xgb_pred[2]), 2),
            },
            "ensemble": {
                "cpu": round(cpu_val / 100.0, 4),
                "memory": round(mem_val / 100.0, 4),
                "request_rate": round(req_val, 2),
            },
        }

    def _safe_fallback_forecast(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Documented safe fallback when data history is warming up."""
        base_cpu = float(df["cpu_usage"].mean()) if not df.empty and "cpu_usage" in df.columns else 40.0
        base_mem = float(df["memory_usage"].mean()) if not df.empty and "memory_usage" in df.columns else 50.0
        base_req = float(df["request_rate"].mean()) if not df.empty and "request_rate" in df.columns else 500.0

        return {
            "timestamp": 0.0,
            "cpu": round(base_cpu / 100.0, 4),
            "memory": round(base_mem / 100.0, 4),
            "request_rate": round(base_req, 2),
            "raw_cpu_percent": round(base_cpu, 2),
            "raw_memory_percent": round(base_mem, 2),
            "lstm": {"cpu": round(base_cpu / 100.0, 4), "memory": round(base_mem / 100.0, 4), "request_rate": base_req},
            "xgboost": {"cpu": round(base_cpu / 100.0, 4), "memory": round(base_mem / 100.0, 4), "request_rate": base_req},
            "ensemble": {"cpu": round(base_cpu / 100.0, 4), "memory": round(base_mem / 100.0, 4), "request_rate": base_req},
        }
