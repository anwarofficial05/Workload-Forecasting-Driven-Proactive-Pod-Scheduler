import os
import shutil
import tempfile
import pytest
import pandas as pd
import numpy as np

from module2_data_preprocessing.preprocessing import DataCleaner, chronological_split
from module2_data_preprocessing.feature_engineering import FeatureEngineer
from module2_data_preprocessing.sequence_builder import SequenceBuilder
from module2_data_preprocessing.scalers import ScalerManager


def test_data_cleaner_and_deduplication():
    cleaner = DataCleaner()
    raw = pd.DataFrame({
        "timestamp": [10.0, 10.0, 20.0, 30.0],
        "node_name": ["w1", "w1", "w1", "w1"],
        "cpu_usage": [40.0, 45.0, -10.0, 150.0], # Includes duplicates and out-of-bound
        "memory_usage": [50.0, 50.0, np.nan, 60.0],
        "request_rate": [100.0, 100.0, 200.0, 300.0],
        "response_latency": [0.02, 0.02, 0.05, 0.08],
    })
    cleaned = cleaner.clean(raw)
    assert len(cleaned) == 3 # Duplicate removed
    assert cleaned["cpu_usage"].min() >= 0.0 # Clamped negative value
    assert cleaned["cpu_usage"].max() <= 100.0 # Clamped >100 value
    assert not cleaned["memory_usage"].isna().any() # Interpolated NaN


def test_chronological_split_no_leakage():
    df = pd.DataFrame({
        "timestamp": list(range(100)),
        "cpu_usage": np.linspace(10, 90, 100),
    })
    train, val, test = chronological_split(df, 0.7, 0.15, 0.15)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    # Strict temporal ordering: max train time < min val time < min test time
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()


def test_feature_engineering_lags_and_spikes():
    fe = FeatureEngineer(lag_steps=[1, 2, 3], rolling_windows=[3, 5])
    df = pd.DataFrame({
        "timestamp": [1700000000 + i*10 for i in range(20)],
        "cpu_usage": [30.0 + (i*2.0) for i in range(20)],
        "memory_usage": [40.0 + (i*1.0) for i in range(20)],
        "request_rate": [500.0 + (i*50.0) for i in range(20)],
        "response_latency": [0.02 + (i*0.001) for i in range(20)],
    })
    feat_df = fe.create_features(df)
    assert "cpu_lag_1" in feat_df.columns
    assert "cpu_lag_2" in feat_df.columns
    assert "cpu_lag_3" in feat_df.columns
    assert "cpu_rolling_mean_3" in feat_df.columns
    assert "cpu_rolling_mean_5" in feat_df.columns
    assert "cpu_rate_of_change" in feat_df.columns
    assert "time_sin" in feat_df.columns
    assert "time_cos" in feat_df.columns
    assert "is_spike_indicator" in feat_df.columns
    assert not feat_df.isna().any().any()


def test_sequence_builder_lstm_and_xgboost():
    sb = SequenceBuilder(sequence_length=5, forecast_horizon=1)
    N = 30
    features = np.ones((N, 4)) * np.arange(N)[:, None]
    targets = np.ones((N, 3)) * np.arange(N)[:, None]

    X_lstm, y_lstm = sb.create_lstm_sequences(features, targets)
    # Expected samples = N - 5 - 1 + 1 = 25
    assert X_lstm.shape == (25, 5, 4)
    assert y_lstm.shape == (25, 3)

    # First target should be the value at index 5 (which is 5.0)
    assert y_lstm[0, 0] == 5.0


def test_scaler_manager_fit_transform_inverse():
    temp_dir = tempfile.mkdtemp()
    try:
        mgr = ScalerManager(scalers_dir=temp_dir)
        X = np.array([[10.0, 100.0], [50.0, 500.0], [90.0, 900.0]])
        y = np.array([[10.0], [50.0], [90.0]])

        mgr.fit(X, y)
        X_scaled = mgr.transform_features(X)
        y_scaled = mgr.transform_targets(y)

        assert np.allclose(X_scaled.min(axis=0), [0.0, 0.0])
        assert np.allclose(X_scaled.max(axis=0), [1.0, 1.0])

        y_inv = mgr.inverse_transform_targets(y_scaled)
        assert np.allclose(y.ravel(), y_inv.ravel())

        mgr.save()
        mgr2 = ScalerManager(scalers_dir=temp_dir)
        assert mgr2.load()
        assert np.allclose(mgr2.transform_features(X), X_scaled)
    finally:
        shutil.rmtree(temp_dir)
