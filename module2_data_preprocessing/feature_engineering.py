import logging
from typing import List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger("module2.feature_engineering")


class FeatureEngineer:
    """
    Creates rich engineered temporal and tabular features for XGBoost and LSTM:
    - Lag features (t-1, t-2, t-3)
    - Rolling window statistics (mean, std, min, max)
    - First-difference rates of change
    - Cyclical and calendar time features
    - Workload spike and sudden change indicators
    """

    def __init__(
        self,
        lag_steps: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
    ):
        self.lag_steps = lag_steps or [1, 2, 3]
        self.rolling_windows = rolling_windows or [3, 5]

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        df_feat = df.copy()

        # Ensure timestamp is datetime or numeric
        if "timestamp" in df_feat.columns:
            ts = pd.to_datetime(df_feat["timestamp"], unit="s", errors="coerce")
            df_feat["hour"] = ts.dt.hour.fillna(0).astype(int)
            df_feat["minute"] = ts.dt.minute.fillna(0).astype(int)
            # Cyclical time encodings
            df_feat["time_sin"] = np.sin(2 * np.pi * (df_feat["hour"] * 60 + df_feat["minute"]) / 1440.0)
            df_feat["time_cos"] = np.cos(2 * np.pi * (df_feat["hour"] * 60 + df_feat["minute"]) / 1440.0)

        # 1. CPU Features
        if "cpu_usage" in df_feat.columns:
            for lag in self.lag_steps:
                df_feat[f"cpu_lag_{lag}"] = df_feat["cpu_usage"].shift(lag)
            for w in self.rolling_windows:
                df_feat[f"cpu_rolling_mean_{w}"] = df_feat["cpu_usage"].rolling(window=w, min_periods=1).mean()
                df_feat[f"cpu_rolling_std_{w}"] = df_feat["cpu_usage"].rolling(window=w, min_periods=1).std().fillna(0.0)
            # Rate of change (velocity)
            df_feat["cpu_rate_of_change"] = df_feat["cpu_usage"].diff().fillna(0.0)

        # 2. Memory Features
        if "memory_usage" in df_feat.columns:
            for lag in [1, 2]:
                df_feat[f"mem_lag_{lag}"] = df_feat["memory_usage"].shift(lag)
            for w in self.rolling_windows:
                df_feat[f"mem_rolling_mean_{w}"] = df_feat["memory_usage"].rolling(window=w, min_periods=1).mean()
            df_feat["mem_rate_of_change"] = df_feat["memory_usage"].diff().fillna(0.0)

        # 3. Request Rate Features
        if "request_rate" in df_feat.columns:
            for lag in [1, 2]:
                df_feat[f"req_lag_{lag}"] = df_feat["request_rate"].shift(lag)
            for w in self.rolling_windows:
                df_feat[f"req_rolling_mean_{w}"] = df_feat["request_rate"].rolling(window=w, min_periods=1).mean()
            df_feat["req_rate_of_change"] = df_feat["request_rate"].diff().fillna(0.0)

        # 4. Latency Features
        if "response_latency" in df_feat.columns:
            df_feat["lat_lag_1"] = df_feat["response_latency"].shift(1)
            df_feat["lat_rolling_mean_3"] = df_feat["response_latency"].rolling(window=3, min_periods=1).mean()

        # 5. Workload Trend & Sudden Spike Indicators
        if "cpu_usage" in df_feat.columns and "cpu_rolling_mean_5" in df_feat.columns:
            df_feat["cpu_trend"] = df_feat["cpu_usage"] - df_feat["cpu_rolling_mean_5"]
            # Spike indicator: when current CPU is significantly higher than rolling mean
            std = df_feat["cpu_rolling_std_5"] if "cpu_rolling_std_5" in df_feat.columns else 5.0
            df_feat["is_spike_indicator"] = (df_feat["cpu_trend"] > (1.5 * std.replace(0, 5.0))).astype(int)

        # Backfill initial lag NaNs using first available value to prevent data loss
        df_feat = df_feat.bfill().ffill().fillna(0.0)
        logger.info(f"Feature engineering generated {df_feat.shape[1]} total columns")
        return df_feat

    @staticmethod
    def get_xgboost_feature_columns() -> List[str]:
        """Return canonical ordered list of feature names for XGBoost training and inference."""
        return [
            "cpu_usage",
            "cpu_lag_1",
            "cpu_lag_2",
            "cpu_lag_3",
            "cpu_rolling_mean_3",
            "cpu_rolling_mean_5",
            "cpu_rate_of_change",
            "memory_usage",
            "mem_lag_1",
            "mem_lag_2",
            "mem_rolling_mean_3",
            "mem_rate_of_change",
            "request_rate",
            "req_lag_1",
            "req_lag_2",
            "req_rolling_mean_3",
            "req_rate_of_change",
            "time_sin",
            "time_cos",
            "cpu_trend",
            "is_spike_indicator",
        ]
