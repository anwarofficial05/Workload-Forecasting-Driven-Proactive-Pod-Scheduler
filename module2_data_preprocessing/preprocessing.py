import logging
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger("module2.preprocessing")


class DataCleaner:
    """
    Cleans raw Kubernetes time-series workload data:
    1. Removes duplicate records based on (timestamp, node_name).
    2. Sorts data chronologically.
    3. Handles missing values via interpolation and forward/backward filling.
    4. Filters invalid out-of-bound values (negative usage or >100% percentages).
    """

    def __init__(self, key_columns: Optional[List[str]] = None):
        self.key_columns = key_columns or ["cpu_usage", "memory_usage", "request_rate", "response_latency"]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        cleaned = df.copy()

        # 1. Deduplicate by timestamp and node if node_name present, else timestamp
        subset = ["timestamp", "node_name"] if "node_name" in cleaned.columns else ["timestamp"]
        cleaned = cleaned.drop_duplicates(subset=subset, keep="last")

        # 2. Sort chronologically
        cleaned = cleaned.sort_values(by="timestamp").reset_index(drop=True)

        # 3. Clip invalid out-of-range numerical values
        if "cpu_usage" in cleaned.columns:
            cleaned["cpu_usage"] = pd.to_numeric(cleaned["cpu_usage"], errors="coerce")
            cleaned["cpu_usage"] = cleaned["cpu_usage"].clip(lower=0.0, upper=100.0)

        if "memory_usage" in cleaned.columns:
            cleaned["memory_usage"] = pd.to_numeric(cleaned["memory_usage"], errors="coerce")
            cleaned["memory_usage"] = cleaned["memory_usage"].clip(lower=0.0, upper=100.0)

        if "request_rate" in cleaned.columns:
            cleaned["request_rate"] = pd.to_numeric(cleaned["request_rate"], errors="coerce")
            cleaned["request_rate"] = cleaned["request_rate"].clip(lower=0.0)

        if "response_latency" in cleaned.columns:
            cleaned["response_latency"] = pd.to_numeric(cleaned["response_latency"], errors="coerce")
            cleaned["response_latency"] = cleaned["response_latency"].clip(lower=0.0)

        # 4. Handle missing values: linear interpolation + ffill + bfill
        for col in self.key_columns:
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].interpolate(method="linear").ffill().bfill().fillna(0.0)

        logger.info(f"Data cleaning completed: {len(df)} -> {len(cleaned)} records")
        return cleaned


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time-series data chronologically strictly without random shuffling
    to prevent data leakage between past and future time steps.
    """
    if not (0.99 <= (train_ratio + val_ratio + test_ratio) <= 1.01):
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    logger.info(
        f"Chronological split: Train={len(train_df)} ({train_ratio*100:.0f}%), "
        f"Val={len(val_df)} ({val_ratio*100:.0f}%), Test={len(test_df)} ({test_ratio*100:.0f}%)"
    )
    return train_df, val_df, test_df
