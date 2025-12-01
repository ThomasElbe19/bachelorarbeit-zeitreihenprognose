"""
model_ffnn.py

Feedforward Neural Network (FFNN) für Zeitreihenprognosen.

Verwendet:
- Dense Layers mit ReLU
- Dropout zur Regularisierung
- Adam als Optimierer
- EarlyStopping zur Overfitting-Kontrolle

Unterstützt:
- In-Domain Training
- Cross-Domain Prediction (Parametertransfer via predict_ffnn)
"""

import numpy as np
from typing import Dict, Any

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from data_pipeline import prepare_dataset_for_ticker
from metrics import mae, rmse


def build_ffnn_model(
    input_dim: int,
    hidden_units: int = 64,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
) -> Sequential:
    """Erstellt ein Feedforward Neural Network (FFNN)."""
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(hidden_units, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_units // 2, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # Regression

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def train_ffnn_for_ticker(
    ticker: str,
    epochs: int = 40,
    batch_size: int = 32,
    verbose: int = 1,
) -> Dict[str, Any]:
    """In-Domain-Training FFNN für einen Ticker."""
    prep = prepare_dataset_for_ticker(ticker)
    X_train_scaled = prep["X_train_scaled"]
    X_val_scaled   = prep["X_val_scaled"]
    X_test_scaled  = prep["X_test_scaled"]

    y_train = prep["y_train"]
    y_val   = prep["y_val"]
    y_test  = prep["y_test"]

    input_dim = X_train_scaled.shape[1]
    model = build_ffnn_model(input_dim=input_dim)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose,
    )

    pred_train = model.predict(X_train_scaled).flatten()
    pred_val   = model.predict(X_val_scaled).flatten()
    pred_test  = model.predict(X_test_scaled).flatten()

    return {
        "ticker": ticker,
        "model": model,
        "scaler": prep["scaler"],
        "history": history.history,

        # Zielgrößen (für weitere Analysen, z.B. vol-abhängige Fehler)
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,

        # Vorhersagen
        "pred_train": pred_train,
        "pred_val":   pred_val,
        "pred_test":  pred_test,

        # Kernmetriken
        "mae_val": mae(y_val, pred_val),
        "rmse_val": rmse(y_val, pred_val),
        "mae_test": mae(y_test, pred_test),
        "rmse_test": rmse(y_test, pred_test),
    }


def predict_ffnn(model, scaler, X_raw: np.ndarray) -> np.ndarray:
    """
    Wendet ein trainiertes FFNN auf neue (fremde) Daten an.

    Wichtig:
        - scaler MUSS derselbe sein wie beim Training auf der Quelle
        - X_raw: UNskalierte Features des Ziel-Tickers
    """
    X_scaled = scaler.transform(X_raw)
    preds = model.predict(X_scaled).flatten()
    return preds


if __name__ == "__main__":
    print("Starte FFNN-Testtraining für IBM...")
    res = train_ffnn_for_ticker("IBM")
    print("MAE Test:", res["mae_test"])
    print("RMSE Test:", res["rmse_test"])
