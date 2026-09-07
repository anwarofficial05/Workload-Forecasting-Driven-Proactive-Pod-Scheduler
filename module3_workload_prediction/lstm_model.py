import os
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger("module3.lstm")


class LSTMWorkloadModel:
    """
    Long Short-Term Memory (LSTM) Recurrent Neural Network for temporal workload forecasting.
    Forecasts future CPU, Memory, and Request Rate based on fixed-length sequences (default: 5 steps).
    Architecture:
      Input (sequence_length, n_features)
        ↓
      LSTM(units=50)
        ↓
      Dense(units=25, relu)
        ↓
      Dense(units=n_targets, linear)
    """

    def __init__(
        self,
        sequence_length: int = 5,
        n_features: int = 4,
        n_targets: int = 3,
        units: int = 50,
        dense_units: int = 25,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        model_dir: str = "models/lstm",
    ):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_targets = n_targets
        self.units = units
        self.dense_units = dense_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.backend = "unknown"
        self.model = None
        self._init_backend()

    def _init_backend(self):
        """Detect and initialize TensorFlow/Keras or PyTorch backend."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            self.backend = "tensorflow"
            logger.info("Using TensorFlow/Keras backend for LSTM")
        except Exception:
            try:
                import torch
                import torch.nn as nn
                self.backend = "torch"
                logger.info("Using PyTorch backend for LSTM")
            except Exception as exc:
                self.backend = "numpy_fallback"
                logger.warning(f"No deep learning backend found ({exc}), using analytical recurrent fallback")

    def build(self):
        """Construct the neural network layers."""
        if self.backend == "tensorflow":
            import tensorflow as tf
            from tensorflow.keras import layers, models, optimizers

            model = models.Sequential([
                layers.Input(shape=(self.sequence_length, self.n_features)),
                layers.LSTM(self.units, return_sequences=False),
                layers.Dropout(self.dropout),
                layers.Dense(self.dense_units, activation="relu"),
                layers.Dense(self.n_targets, activation="linear"),
            ])
            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                loss="mse",
                metrics=["mae"],
            )
            self.model = model
        elif self.backend == "torch":
            import torch
            import torch.nn as nn

            class TorchLSTM(nn.Module):
                def __init__(self, seq_len, in_feat, units, d_units, out_feat, dropout):
                    super().__init__()
                    self.lstm = nn.LSTM(in_feat, units, batch_first=True)
                    self.dropout = nn.Dropout(dropout)
                    self.fc1 = nn.Linear(units, d_units)
                    self.relu = nn.ReLU()
                    self.fc2 = nn.Linear(d_units, out_feat)

                def forward(self, x):
                    out, _ = self.lstm(x)
                    # Use last time-step
                    last_step = out[:, -1, :]
                    d1 = self.relu(self.fc1(self.dropout(last_step)))
                    return self.fc2(d1)

            self.model = TorchLSTM(
                self.sequence_length,
                self.n_features,
                self.units,
                self.dense_units,
                self.n_targets,
                self.dropout,
            )
        else:
            # Analytical fallback weights
            self.model = {
                "W_rec": np.random.normal(0, 0.05, (self.n_features, self.n_targets)),
                "bias": np.zeros(self.n_targets),
            }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 30,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Train the LSTM model on training sequences."""
        if self.model is None:
            self.build()

        if self.backend == "tensorflow":
            val_data = (X_val, y_val) if X_val is not None and y_val is not None else None
            history = self.model.fit(
                X_train,
                y_train,
                validation_data=val_data,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
            )
            return history.history

        elif self.backend == "torch":
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader

            dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            history = {"loss": [], "val_loss": []}

            self.model.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                for bx, by in loader:
                    optimizer.zero_grad()
                    pred = self.model(bx)
                    loss = criterion(pred, by)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * len(bx)
                epoch_loss /= len(X_train)
                history["loss"].append(epoch_loss)

                if X_val is not None and y_val is not None:
                    self.model.eval()
                    with torch.no_grad():
                        vx = torch.tensor(X_val, dtype=torch.float32)
                        vy = torch.tensor(y_val, dtype=torch.float32)
                        v_loss = criterion(self.model(vx), vy).item()
                        history["val_loss"].append(v_loss)
                    self.model.train()

            return history

        else:
            # Fallback ridge/least-squares fit on sequence mean
            X_flat = X_train.mean(axis=1)
            # pseudo-inverse
            W, residuals, rank, s = np.linalg.lstsq(X_flat, y_train, rcond=None)
            self.model["W_rec"] = W
            return {"loss": [float(np.mean(residuals)) if len(residuals) > 0 else 0.01]}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input sequences.
        Input shape: (batch_size, sequence_length, n_features) or (sequence_length, n_features).
        Output shape: (batch_size, n_targets).
        """
        if self.model is None:
            self.build()

        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=0)

        if self.backend == "tensorflow":
            return self.model.predict(X, verbose=0)
        elif self.backend == "torch":
            import torch
            self.model.eval()
            with torch.no_grad():
                tensor_x = torch.tensor(X, dtype=torch.float32)
                return self.model(tensor_x).cpu().numpy()
        else:
            X_flat = X.mean(axis=1)
            return np.dot(X_flat, self.model["W_rec"]) + self.model["bias"]

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Compute MSE, RMSE, and MAE on test sequences."""
        preds = self.predict(X_test)
        mse = float(np.mean((preds - y_test) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(preds - y_test)))
        return {"mse": mse, "rmse": rmse, "mae": mae}

    def save(self, filepath: Optional[str] = None):
        """Save trained model to disk."""
        path = filepath or os.path.join(self.model_dir, "lstm_model")
        if self.backend == "tensorflow":
            self.model.save(f"{path}.keras")
            logger.info(f"Saved Keras LSTM model to {path}.keras")
        elif self.backend == "torch":
            import torch
            torch.save(self.model.state_dict(), f"{path}.pt")
            logger.info(f"Saved PyTorch LSTM model to {path}.pt")
        else:
            import joblib
            joblib.dump(self.model, f"{path}.joblib")

    def load(self, filepath: Optional[str] = None) -> bool:
        """Load trained model from disk."""
        path = filepath or os.path.join(self.model_dir, "lstm_model")
        keras_path = f"{path}.keras"
        torch_path = f"{path}.pt"
        joblib_path = f"{path}.joblib"

        if os.path.exists(keras_path):
            try:
                import tensorflow as tf
                from tensorflow import keras
                self.model = keras.models.load_model(keras_path)
                self.backend = "tensorflow"
                logger.info(f"Loaded Keras model from {keras_path}")
                return True
            except Exception as exc:
                logger.warning(f"Could not load Keras model: {exc}")

        if os.path.exists(torch_path):
            try:
                import torch
                self.build()
                self.model.load_state_dict(torch.load(torch_path, map_location="cpu"))
                self.model.eval()
                self.backend = "torch"
                logger.info(f"Loaded PyTorch model from {torch_path}")
                return True
            except Exception as exc:
                logger.warning(f"Could not load PyTorch model: {exc}")

        if os.path.exists(joblib_path):
            import joblib
            self.model = joblib.load(joblib_path)
            self.backend = "numpy_fallback"
            return True

        logger.warning(f"No LSTM model found at {path}")
        return False
