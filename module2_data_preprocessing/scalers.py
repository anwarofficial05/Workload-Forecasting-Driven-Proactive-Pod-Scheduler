import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger("module2.scalers")


class ScalerManager:
    """
    Manages normalization scalers for features and target variables.
    Saves and loads scalers so that inference uses the exact same scaling parameters.
    """

    def __init__(self, scalers_dir: str = "models/scalers", scaler_type: str = "minmax"):
        self.scalers_dir = scalers_dir
        os.makedirs(self.scalers_dir, exist_ok=True)
        self.scaler_type = scaler_type.lower()
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1)) if self.scaler_type == "minmax" else StandardScaler()
        self.target_scaler = MinMaxScaler(feature_range=(0, 1)) if self.scaler_type == "minmax" else StandardScaler()
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit scalers on training set only (preventing leakage)."""
        self.feature_scaler.fit(X)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        self.target_scaler.fit(y)
        self.fitted = True
        logger.info(f"Fitted {self.scaler_type} scalers for X shape {X.shape} and y shape {y.shape}")

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        return self.feature_scaler.transform(X)

    def transform_targets(self, y: np.ndarray) -> np.ndarray:
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        transformed = self.target_scaler.transform(y)
        return transformed.squeeze() if transformed.shape[1] == 1 else transformed

    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        if len(y_scaled.shape) == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        inverted = self.target_scaler.inverse_transform(y_scaled)
        return inverted.squeeze() if inverted.shape[1] == 1 else inverted

    def save(self):
        """Persist fitted scalers to disk."""
        feat_path = os.path.join(self.scalers_dir, "feature_scaler.joblib")
        target_path = os.path.join(self.scalers_dir, "target_scaler.joblib")
        joblib.dump(self.feature_scaler, feat_path)
        joblib.dump(self.target_scaler, target_path)
        logger.info(f"Saved scalers to {self.scalers_dir}")

    def load(self) -> bool:
        """Load persisted scalers from disk."""
        feat_path = os.path.join(self.scalers_dir, "feature_scaler.joblib")
        target_path = os.path.join(self.scalers_dir, "target_scaler.joblib")
        if os.path.exists(feat_path) and os.path.exists(target_path):
            self.feature_scaler = joblib.load(feat_path)
            self.target_scaler = joblib.load(target_path)
            self.fitted = True
            logger.info(f"Loaded scalers from {self.scalers_dir}")
            return True
        logger.warning(f"No saved scalers found in {self.scalers_dir}")
        return False
