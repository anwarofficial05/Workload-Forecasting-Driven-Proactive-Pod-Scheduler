import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import xgboost as xgb
import joblib

logger = logging.getLogger("module3.xgboost")


class XGBoostWorkloadModel:
    """
    XGBoost Gradient Boosted Decision Tree models for non-linear workload forecasting.
    Trains specialized estimators for CPU, Memory, and Request Rate targets
    using engineered tabular features (lags, rolling averages, rates of change, spikes).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        model_dir: str = "models/xgboost",
        target_names: Optional[List[str]] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.model_dir = model_dir
        self.target_names = target_names or ["cpu_usage", "memory_usage", "request_rate"]
        os.makedirs(self.model_dir, exist_ok=True)

        self.models: Dict[str, xgb.XGBRegressor] = {}
        self.feature_names: List[str] = []

    def _build_regressor(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            objective="reg:squarederror",
            n_jobs=-1,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Train independent regressors for each target."""
        if feature_names:
            self.feature_names = feature_names

        # Ensure y_train is 2D
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)

        metrics = {}
        for idx, target in enumerate(self.target_names):
            if idx >= y_train.shape[1]:
                break
            y_col = y_train[:, idx]
            reg = self._build_regressor()

            eval_set = None
            if X_val is not None and y_val is not None:
                if len(y_val.shape) == 1:
                    y_val = y_val.reshape(-1, 1)
                eval_set = [(X_val, y_val[:, idx])]

            reg.fit(
                X_train,
                y_col,
                eval_set=eval_set,
                verbose=False,
            )
            self.models[target] = reg

            # Calculate train metric
            train_preds = reg.predict(X_train)
            mse = float(np.mean((train_preds - y_col) ** 2))
            metrics[target] = {"mse": mse, "rmse": float(np.sqrt(mse))}
            logger.info(f"Trained XGBoost for {target}: RMSE={metrics[target]['rmse']:.4f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for all targets.
        Output shape: (batch_size, n_targets) or (n_targets,) for 1D input.
        """
        if not self.models:
            raise RuntimeError("XGBoost models are not trained or loaded yet.")

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        preds = []
        for target in self.target_names:
            if target in self.models:
                p = self.models[target].predict(X)
                preds.append(p)
            else:
                preds.append(np.zeros(len(X)))

        result = np.column_stack(preds)
        return result

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Return feature importance scores for each target."""
        importance_dict = {}
        for target, model in self.models.items():
            booster = model.get_booster()
            scores = booster.get_score(importance_type="weight")
            # Map back to feature names if available
            named_scores = {}
            for k, v in scores.items():
                if k.startswith("f") and k[1:].isdigit():
                    idx = int(k[1:])
                    name = self.feature_names[idx] if idx < len(self.feature_names) else k
                else:
                    name = k
                named_scores[name] = float(v)
            importance_dict[target] = named_scores
        return importance_dict

    def save(self):
        """Save models and metadata."""
        for target, model in self.models.items():
            path = os.path.join(self.model_dir, f"xgb_{target}.json")
            model.save_model(path)
        meta_path = os.path.join(self.model_dir, "metadata.joblib")
        joblib.dump(
            {"target_names": self.target_names, "feature_names": self.feature_names},
            meta_path,
        )
        logger.info(f"Saved XGBoost models to {self.model_dir}")

    def load(self) -> bool:
        """Load models and metadata."""
        meta_path = os.path.join(self.model_dir, "metadata.joblib")
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            self.target_names = meta.get("target_names", self.target_names)
            self.feature_names = meta.get("feature_names", self.feature_names)

        loaded_count = 0
        for target in self.target_names:
            path = os.path.join(self.model_dir, f"xgb_{target}.json")
            if os.path.exists(path):
                reg = self._build_regressor()
                reg.load_model(path)
                self.models[target] = reg
                loaded_count += 1

        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} XGBoost target models from {self.model_dir}")
            return True
        return False
