import os
import sys
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from module1_data_collection.data_storage import MetricsStorage
from module2_data_preprocessing.preprocessing import DataCleaner, chronological_split
from module2_data_preprocessing.feature_engineering import FeatureEngineer
from module2_data_preprocessing.sequence_builder import SequenceBuilder
from module2_data_preprocessing.scalers import ScalerManager
from module3_workload_prediction.lstm_model import LSTMWorkloadModel
from module3_workload_prediction.xgboost_model import XGBoostWorkloadModel
from module3_workload_prediction.ensemble import WeightedEnsembleFusion

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("train_pipeline")


def load_config(config_path: str = "config/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def run_training_pipeline(
    config_path: str = "config/config.yaml",
    epochs: int = 30,
    batch_size: int = 32,
    num_samples: int = 2000,
):
    cfg = load_config(config_path)
    pred_cfg = cfg.get("prediction", {})
    prep_cfg = cfg.get("preprocessing", {})
    coll_cfg = cfg.get("collection", {})

    storage_dir = coll_cfg.get("storage_dir", "data")
    storage = MetricsStorage(storage_dir=storage_dir)

    # 1. Load or Generate Historical Dataset
    df_raw = storage.load_data()
    if df_raw.empty or len(df_raw) < 100:
        logger.info(f"Dataset in {storage_dir} has insufficient samples. Generating {num_samples} realistic samples...")
        df_raw = storage.generate_synthetic_workload_history(num_samples=num_samples)

    logger.info(f"Loaded raw dataset with {len(df_raw)} records")

    # 2. Data Cleaning
    cleaner = DataCleaner()
    df_cleaned = cleaner.clean(df_raw)

    # 3. Chronological Split (Train: 70%, Val: 15%, Test: 15%)
    train_df, val_df, test_df = chronological_split(
        df_cleaned,
        train_ratio=prep_cfg.get("train_split_ratio", 0.70),
        val_ratio=prep_cfg.get("val_split_ratio", 0.15),
        test_ratio=prep_cfg.get("test_split_ratio", 0.15),
    )

    feature_cols = prep_cfg.get("feature_columns", ["cpu_usage", "memory_usage", "request_rate", "response_latency"])
    target_cols = pred_cfg.get("target_columns", ["cpu_usage", "memory_usage", "request_rate"])

    # 4. Fit Scalers on Training Data ONLY
    scalers_dir = prep_cfg.get("scalers_dir", "models/scalers")
    scaler_mgr = ScalerManager(scalers_dir=scalers_dir)
    scaler_mgr.fit(train_df[feature_cols].values, train_df[target_cols].values)
    scaler_mgr.save()

    # Normalize datasets
    train_feat_scaled = scaler_mgr.transform_features(train_df[feature_cols].values)
    train_targ_scaled = scaler_mgr.transform_targets(train_df[target_cols].values)

    val_feat_scaled = scaler_mgr.transform_features(val_df[feature_cols].values)
    val_targ_scaled = scaler_mgr.transform_targets(val_df[target_cols].values)

    test_feat_scaled = scaler_mgr.transform_features(test_df[feature_cols].values)
    test_targ_scaled = scaler_mgr.transform_targets(test_df[target_cols].values)

    # 5. Build Sequences for LSTM
    seq_len = prep_cfg.get("sequence_length", 5)
    horizon = prep_cfg.get("forecast_horizon", 1)
    seq_builder = SequenceBuilder(
        sequence_length=seq_len,
        forecast_horizon=horizon,
        feature_columns=feature_cols,
        target_columns=target_cols,
    )

    X_train_lstm, y_train_lstm = seq_builder.create_lstm_sequences(train_feat_scaled, train_targ_scaled)
    X_val_lstm, y_val_lstm = seq_builder.create_lstm_sequences(val_feat_scaled, val_targ_scaled)
    X_test_lstm, y_test_lstm = seq_builder.create_lstm_sequences(test_feat_scaled, test_targ_scaled)

    # 6. Train LSTM Model
    lstm_cfg = pred_cfg.get("lstm", {})
    lstm_model = LSTMWorkloadModel(
        sequence_length=seq_len,
        n_features=len(feature_cols),
        n_targets=len(target_cols),
        units=lstm_cfg.get("units", 50),
        dense_units=lstm_cfg.get("dense_units", 25),
        dropout=lstm_cfg.get("dropout", 0.2),
        learning_rate=lstm_cfg.get("learning_rate", 0.001),
        model_dir=lstm_cfg.get("model_dir", "models/lstm"),
    )
    logger.info("Training LSTM Neural Network...")
    lstm_model.train(
        X_train_lstm,
        y_train_lstm,
        X_val=X_val_lstm,
        y_val=y_val_lstm,
        epochs=epochs,
        batch_size=batch_size,
    )
    lstm_model.save()

    # 7. Feature Engineering for XGBoost
    fe = FeatureEngineer(lag_steps=prep_cfg.get("lag_steps", [1, 2, 3]))
    train_eng = fe.create_features(train_df)
    val_eng = fe.create_features(val_df)
    test_eng = fe.create_features(test_df)

    xgb_feat_cols = fe.get_xgboost_feature_columns()
    X_train_xgb, y_train_xgb = seq_builder.create_xgboost_dataset(train_eng, xgb_feat_cols, target_cols)
    X_val_xgb, y_val_xgb = seq_builder.create_xgboost_dataset(val_eng, xgb_feat_cols, target_cols)
    X_test_xgb, y_test_xgb = seq_builder.create_xgboost_dataset(test_eng, xgb_feat_cols, target_cols)

    # 8. Train XGBoost Models
    xgb_cfg = pred_cfg.get("xgboost", {})
    xgb_model = XGBoostWorkloadModel(
        n_estimators=xgb_cfg.get("n_estimators", 100),
        max_depth=xgb_cfg.get("max_depth", 6),
        learning_rate=xgb_cfg.get("learning_rate", 0.05),
        model_dir=xgb_cfg.get("model_dir", "models/xgboost"),
        target_names=target_cols,
    )
    logger.info("Training XGBoost Regressors...")
    xgb_model.train(
        X_train_xgb,
        y_train_xgb,
        feature_names=xgb_feat_cols,
        X_val=X_val_xgb,
        y_val=y_val_xgb,
    )
    xgb_model.save()

    # 9. Test Set Evaluation & Weighted Ensemble Fusion
    ens_cfg = pred_cfg.get("ensemble", {})
    ensemble = WeightedEnsembleFusion(
        lstm_weight=ens_cfg.get("lstm_weight", 0.60),
        xgboost_weight=ens_cfg.get("xgboost_weight", 0.40),
    )

    # Predictions in original physical units
    lstm_preds_scaled = lstm_model.predict(X_test_lstm)
    lstm_preds = scaler_mgr.inverse_transform_targets(lstm_preds_scaled)

    xgb_preds = xgb_model.predict(X_test_xgb)

    # Align sequence evaluation length
    eval_len = min(len(lstm_preds), len(xgb_preds))
    lstm_eval = lstm_preds[:eval_len]
    xgb_eval = xgb_preds[:eval_len]
    ens_eval = ensemble.fuse(lstm_eval, xgb_eval)
    y_actual = test_df[target_cols].iloc[-eval_len:].values

    # Compute Comparative Metrics
    results = {}
    for idx, col in enumerate(target_cols):
        act = y_actual[:, idx]
        l_p = lstm_eval[:, idx]
        x_p = xgb_eval[:, idx]
        e_p = ens_eval[:, idx]

        results[col] = {
            "LSTM": {
                "MSE": round(float(mean_squared_error(act, l_p)), 4),
                "RMSE": round(float(np.sqrt(mean_squared_error(act, l_p))), 4),
                "MAE": round(float(mean_absolute_error(act, l_p)), 4),
                "R2": round(float(r2_score(act, l_p)), 4),
            },
            "XGBoost": {
                "MSE": round(float(mean_squared_error(act, x_p)), 4),
                "RMSE": round(float(np.sqrt(mean_squared_error(act, x_p))), 4),
                "MAE": round(float(mean_absolute_error(act, x_p)), 4),
                "R2": round(float(r2_score(act, x_p)), 4),
            },
            "Ensemble": {
                "MSE": round(float(mean_squared_error(act, e_p)), 4),
                "RMSE": round(float(np.sqrt(mean_squared_error(act, e_p))), 4),
                "MAE": round(float(mean_absolute_error(act, e_p)), 4),
                "R2": round(float(r2_score(act, e_p)), 4),
            },
        }

    print("\n============================================================")
    print("        WORKLOAD FORECASTING EVALUATION RESULTS             ")
    print("============================================================")
    for col, models in results.items():
        print(f"\n--- Target: {col.upper()} ---")
        print(f"{'Model':<12} | {'MSE':<8} | {'RMSE':<8} | {'MAE':<8} | {'R²':<8}")
        print("-" * 50)
        for m_name, m_metrics in models.items():
            print(f"{m_name:<12} | {m_metrics['MSE']:<8.4f} | {m_metrics['RMSE']:<8.4f} | {m_metrics['MAE']:<8.4f} | {m_metrics['R2']:<8.4f}")

    print("============================================================\n")
    logger.info("Training pipeline finished successfully! Models saved.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM and XGBoost Workload Models")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--samples", type=int, default=2000, help="Number of synthetic samples if data file empty")
    args = parser.parse_args()

    run_training_pipeline(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.samples,
    )
