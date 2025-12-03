"""
LSTM-Modell für die Prognose der täglichen log-Rendite r_{t+1}.

Features:
- Sequenzen (z.B. 30 Tage)
- LSTM + Dropout + Dense(1)
- EarlyStopping (Overfitting-Kontrolle)

Unterstützt:
- In-Domain Training
- Cross-Domain Prediction (Transferlernen über Parametertransfer)
"""

from typing import Dict, Any, Tuple
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from data_pipeline import prepare_dataset_for_ticker
from metrics import mae, rmse


def build_lstm_sequences(
    X_2d: np.ndarray,
    y_1d: np.ndarray,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Erstellt aus 2D-Features (Samples × Features) LSTM-Sequenzen:
        X[t-seq+1 : t] → y[t]
    """
    X_seq, y_seq = [], []
    for t in range(sequence_length - 1, len(X_2d)):
        X_seq.append(X_2d[t - sequence_length + 1 : t + 1, :])
        y_seq.append(y_1d[t])
    return np.array(X_seq, dtype="float32"), np.array(y_seq, dtype="float32")


def build_lstm_model(
    sequence_length: int,
    n_features: int,
    lstm_units: int = 64,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
) -> Sequential:
    """Erstellt ein LSTM-Modell: Input → LSTM → Dropout → Dense(1)."""
    model = Sequential()
    model.add(Input(shape=(sequence_length, n_features)))
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def train_lstm_for_ticker(
    ticker: str,
    sequence_length: int = 30,
    epochs: int = 40,
    batch_size: int = 32,
    verbose: int = 1,
) -> Dict[str, Any]:
    """In-Domain-Training LSTM für einen Ticker."""
    prep = prepare_dataset_for_ticker(ticker)

    # 2D-Features und Zielgrößen aus der Pipeline
    X_train_2d = prep["X_train_scaled"]
    X_val_2d   = prep["X_val_scaled"]
    X_test_2d  = prep["X_test_scaled"]

    y_train_1d = prep["y_train"]
    y_val_1d   = prep["y_val"]
    y_test_1d  = prep["y_test"]

    n_features = X_train_2d.shape[1]

    # Sequenzen für LSTM bauen
    X_train, y_train = build_lstm_sequences(X_train_2d, y_train_1d, sequence_length)
    X_val,   y_val   = build_lstm_sequences(X_val_2d,   y_val_1d,   sequence_length)
    X_test,  y_test  = build_lstm_sequences(X_test_2d,  y_test_1d,  sequence_length)

    model = build_lstm_model(sequence_length, n_features)

    es = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=verbose,
    )

    # Vorhersagen
    pred_train = model.predict(X_train).flatten()
    pred_val   = model.predict(X_val).flatten()
    pred_test  = model.predict(X_test).flatten()

    results = {
        "ticker": ticker,
        "model": model,
        "scaler": prep["scaler"],  
        "sequence_length": sequence_length,
        "history": history.history,

        "X_train_seq": X_train,
        "X_val_seq":   X_val,
        "X_test_seq":  X_test,
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,

        "pred_train": pred_train,
        "pred_val":   pred_val,
        "pred_test":  pred_test,

        "mae_val":  mae(y_val, pred_val),
        "rmse_val": rmse(y_val, pred_val),
        "mae_test": mae(y_test, pred_test),
        "rmse_test": rmse(y_test, pred_test),
    }

    return results


def predict_lstm(model, scaler, X_seq_raw: np.ndarray) -> np.ndarray:
    """
    Verwendet ein trainiertes LSTM-Modell, um Sequenzdaten eines anderen
    Tickers vorherzusagen.

    Erwartet:
        X_seq_raw: 3D-Array (Samples × seq_len × n_features) mit
                   UNskalierten Features; Skalierung erfolgt mit dem
                   Source-Scaler innerhalb dieser Funktion.
    """
    X_scaled = X_seq_raw.copy()
    for i in range(len(X_seq_raw)):
        X_scaled[i] = scaler.transform(X_seq_raw[i])
    preds = model.predict(X_scaled).flatten()
    return preds


if __name__ == "__main__":
    print("Starte LSTM-Testtraining für IBM...")
    res = train_lstm_for_ticker("IBM", sequence_length=30)
    print("MAE Test :", res["mae_test"])
    print("RMSE Test:", res["rmse_test"])
