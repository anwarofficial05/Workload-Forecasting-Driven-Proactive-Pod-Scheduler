import logging
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger("module3.ensemble")


class WeightedEnsembleFusion:
    """
    Weighted Ensemble Fusion combining temporal predictions from LSTM
    and tabular non-linear predictions from XGBoost.
    Formula:
      ENSEMBLE_PREDICTION = (LSTM_WEIGHT * LSTM_PREDICTION) + (XGBOOST_WEIGHT * XGBOOST_PREDICTION)
    Default weights:
      LSTM_WEIGHT = 0.60
      XGBOOST_WEIGHT = 0.40
    """

    def __init__(self, lstm_weight: float = 0.60, xgboost_weight: float = 0.40):
        total = lstm_weight + xgboost_weight
        if abs(total - 1.0) > 1e-4:
            # Normalize to 1.0
            self.lstm_weight = lstm_weight / total
            self.xgboost_weight = xgboost_weight / total
        else:
            self.lstm_weight = lstm_weight
            self.xgboost_weight = xgboost_weight

        logger.info(
            f"Initialized Weighted Ensemble with LSTM_WEIGHT={self.lstm_weight:.2f}, "
            f"XGBOOST_WEIGHT={self.xgboost_weight:.2f}"
        )

    def fuse(
        self,
        lstm_preds: np.ndarray,
        xgboost_preds: np.ndarray,
    ) -> np.ndarray:
        """
        Compute weighted linear combination of LSTM and XGBoost predictions.
        Both inputs must have matching dimensions.
        """
        lstm_arr = np.asarray(lstm_preds, dtype=np.float32)
        xgb_arr = np.asarray(xgboost_preds, dtype=np.float32)

        if lstm_arr.shape != xgb_arr.shape:
            # Broadcast or squeeze if single-sample
            lstm_arr = np.squeeze(lstm_arr)
            xgb_arr = np.squeeze(xgb_arr)

        ensemble_preds = (self.lstm_weight * lstm_arr) + (self.xgboost_weight * xgb_arr)
        return ensemble_preds

    def predict_detailed(
        self,
        lstm_preds: np.ndarray,
        xgboost_preds: np.ndarray,
        actual: Optional[np.ndarray] = None,
        target_names: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Return structured prediction dictionary showing:
        - LSTM prediction
        - XGBoost prediction
        - Ensemble prediction
        - Actual workload (if provided)
        """
        targets = target_names or ["cpu_usage", "memory_usage", "request_rate"]
        ensemble_preds = self.fuse(lstm_preds, xgboost_preds)

        l_flat = np.atleast_1d(np.squeeze(lstm_preds))
        x_flat = np.atleast_1d(np.squeeze(xgboost_preds))
        e_flat = np.atleast_1d(np.squeeze(ensemble_preds))
        a_flat = np.atleast_1d(np.squeeze(actual)) if actual is not None else None

        output = {
            "lstm": {},
            "xgboost": {},
            "ensemble": {},
            "weights": {
                "lstm_weight": self.lstm_weight,
                "xgboost_weight": self.xgboost_weight,
            },
        }

        for idx, name in enumerate(targets):
            if idx < len(l_flat):
                output["lstm"][name] = round(float(l_flat[idx]), 4)
                output["xgboost"][name] = round(float(x_flat[idx]), 4)
                output["ensemble"][name] = round(float(e_flat[idx]), 4)
                if a_flat is not None and idx < len(a_flat):
                    output.setdefault("actual", {})[name] = round(float(a_flat[idx]), 4)

        return output
