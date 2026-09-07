import os
import shutil
import tempfile
import pytest
import numpy as np

from module3_workload_prediction.lstm_model import LSTMWorkloadModel
from module3_workload_prediction.xgboost_model import XGBoostWorkloadModel
from module3_workload_prediction.ensemble import WeightedEnsembleFusion
from module3_workload_prediction.inference import WorkloadInferenceEngine


def test_lstm_model_build_train_predict():
    temp_dir = tempfile.mkdtemp()
    try:
        model = LSTMWorkloadModel(
            sequence_length=5,
            n_features=4,
            n_targets=3,
            units=16,
            dense_units=8,
            model_dir=temp_dir,
        )
        model.build()

        X_train = np.random.uniform(0, 1, (30, 5, 4)).astype(np.float32)
        y_train = np.random.uniform(0, 1, (30, 3)).astype(np.float32)

        history = model.train(X_train, y_train, epochs=2, batch_size=16)
        assert "loss" in history

        preds = model.predict(X_train[:5])
        assert preds.shape == (5, 3)

        model.save()
        model2 = LSTMWorkloadModel(
            sequence_length=5,
            n_features=4,
            n_targets=3,
            units=16,
            dense_units=8,
            model_dir=temp_dir,
        )
        assert model2.load()
        preds2 = model2.predict(X_train[:5])
        assert np.allclose(preds, preds2, atol=1e-3)
    finally:
        shutil.rmtree(temp_dir)


def test_xgboost_model_train_predict_importance():
    temp_dir = tempfile.mkdtemp()
    try:
        xgb_m = XGBoostWorkloadModel(
            n_estimators=10,
            max_depth=3,
            model_dir=temp_dir,
            target_names=["cpu_usage", "memory_usage", "request_rate"],
        )
        feat_names = [f"f_{i}" for i in range(10)]
        X = np.random.uniform(0, 100, (50, 10))
        y = np.column_stack([X[:, 0]*0.5, X[:, 1]*0.8, X[:, 2]*10.0])

        metrics = xgb_m.train(X, y, feature_names=feat_names)
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert "request_rate" in metrics

        preds = xgb_m.predict(X[:4])
        assert preds.shape == (4, 3)

        xgb_m.save()
        xgb_m2 = XGBoostWorkloadModel(model_dir=temp_dir)
        assert xgb_m2.load()
        preds2 = xgb_m2.predict(X[:4])
        assert np.allclose(preds, preds2, atol=1e-3)
    finally:
        shutil.rmtree(temp_dir)


def test_weighted_ensemble_fusion():
    ens = WeightedEnsembleFusion(lstm_weight=0.60, xgboost_weight=0.40)
    lstm_preds = np.array([50.0, 60.0, 1000.0])
    xgb_preds = np.array([70.0, 80.0, 1200.0])

    fused = ens.fuse(lstm_preds, xgb_preds)
    expected_cpu = (0.60 * 50.0) + (0.40 * 70.0) # 30 + 28 = 58.0
    expected_mem = (0.60 * 60.0) + (0.40 * 80.0) # 36 + 32 = 68.0
    expected_req = (0.60 * 1000.0) + (0.40 * 1200.0) # 600 + 480 = 1080.0

    assert np.isclose(fused[0], expected_cpu)
    assert np.isclose(fused[1], expected_mem)
    assert np.isclose(fused[2], expected_req)

    details = ens.predict_detailed(lstm_preds, xgb_preds)
    assert details["ensemble"]["cpu_usage"] == round(expected_cpu, 4)
    assert details["weights"]["lstm_weight"] == 0.60
    assert details["weights"]["xgboost_weight"] == 0.40
