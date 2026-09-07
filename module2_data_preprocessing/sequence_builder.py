import logging
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("module2.sequence_builder")


class SequenceBuilder:
    """
    Constructs sliding time-window sequences for LSTM and tabular datasets for XGBoost.
    Default sequence length is 5: (t-4, t-3, t-2, t-1, t) -> (t+1 future targets).
    """

    def __init__(
        self,
        sequence_length: int = 5,
        forecast_horizon: int = 1,
        feature_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
    ):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.feature_columns = feature_columns or ["cpu_usage", "memory_usage", "request_rate", "response_latency"]
        self.target_columns = target_columns or ["cpu_usage", "memory_usage", "request_rate"]

    def create_lstm_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build 3D tensor for LSTM: (N, sequence_length, num_features)
        along with target tensor: (N, num_targets) representing step t + forecast_horizon.
        """
        X, y = [], []
        num_samples = len(features)

        for i in range(self.sequence_length, num_samples - self.forecast_horizon + 1):
            X.append(features[i - self.sequence_length : i])
            y.append(targets[i + self.forecast_horizon - 1])

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.float32)
        logger.info(f"Created LSTM sequences: X shape {X_arr.shape}, y shape {y_arr.shape}")
        return X_arr, y_arr

    def create_xgboost_dataset(
        self,
        df_engineered: pd.DataFrame,
        feature_cols: List[str],
        target_cols: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build tabular feature matrix X and target matrix y shifted by forecast_horizon
        for training XGBoost models.
        """
        targets = target_cols or self.target_columns
        # Shift target forward by forecast_horizon so current row predicts future
        y_df = df_engineered[targets].shift(-self.forecast_horizon)

        # Drop invalid rows at the tail resulting from shift
        valid_idx = ~y_df.isna().any(axis=1)
        X = df_engineered.loc[valid_idx, feature_cols].values.astype(np.float32)
        y = y_df.loc[valid_idx].values.astype(np.float32)

        # If single target, flatten to 1D
        if y.shape[1] == 1:
            y = y.ravel()

        logger.info(f"Created XGBoost dataset: X shape {X.shape}, y shape {y.shape}")
        return X, y
