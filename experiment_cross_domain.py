"""
experiment_cross_domain.py

Cross-Domain-Experimente:
    - FFNN & LSTM: Parametertransfer (Transferlernen)
    - ARIMA & Prophet: Strukturtransfer

Ergebnis:
    - CSV: results/model_comparison_cross_domain.csv
      mit Kernmetriken (MAE, RMSE) auf Val und Test,
      Δ-Metriken (delta_mae, delta_rmse) im Sinne von dL_fcst (Deng et al.),
      sowie volatilitätsabhängigen RMSE-Werten (low/med/high Volatilität)
      auf dem Test-Set der Zieldomäne.
    - Plots: für alle 6 Source→Target-Kombinationen (IBM, NVDA, Nike)
             je EIN Test-Plot, in dem alle vier Modelle
             (ARIMA, Prophet, FFNN, LSTM) gemeinsam mit den echten
             Testwerten des Ziel-Tickers liegen.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from model_ffnn import train_ffnn_for_ticker, predict_ffnn
from model_lstm import train_lstm_for_ticker, predict_lstm, build_lstm_sequences
from model_arima import cross_domain_arima
from model_prophet import cross_domain_prophet

from data_pipeline import prepare_dataset_for_ticker
from metrics import mae, rmse, rmse_by_vol_quantiles

from plot_results import plot_test_set_all_models

TICKERS = ["IBM", "NVDA", "Nike"]
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

PLOT_ZOOM_LAST = 250


# =====================================================================
# FFNN CROSS-DOMAIN
# =====================================================================

def cross_domain_ffnn(source: str, target: str):
    """
    FFNN-Transfer:
        - FFNN auf Source trainieren (In-Domain)
        - Source-Scaler & Modell auf Target-Rohfeatures anwenden
        - In-Domain-Testfehler der Quelle (MAE/RMSE) mitgeben
    """
    prep_s = prepare_dataset_for_ticker(source)
    prep_t = prepare_dataset_for_ticker(target)

    trained = train_ffnn_for_ticker(source)
    model = trained["model"]
    scaler = trained["scaler"]

    mae_source_test = trained["mae_test"]
    rmse_source_test = trained["rmse_test"]

    pred_val = predict_ffnn(model, scaler, prep_t["X_val"])
    pred_test = predict_ffnn(model, scaler, prep_t["X_test"])

    return {
        "pred_val": pred_val.astype("float32"),
        "pred_test": pred_test.astype("float32"),
        "y_val": prep_t["y_val"].astype("float32"),
        "y_test": prep_t["y_test"].astype("float32"),
        "y_train": prep_s["y_train"].astype("float32"),
        "best_cfg": None,
        "mae_source_test": mae_source_test,
        "rmse_source_test": rmse_source_test,
    }


# =====================================================================
# LSTM CROSS-DOMAIN
# =====================================================================

def cross_domain_lstm(source: str, target: str):
    """
    LSTM-Transfer:
        - LSTM auf Source trainieren (mit Sequenzen auf skalierten Features)
        - trainiertes Modell + Source-Scaler auf Target-Sequenzen anwenden,
          die aus UNskalierten Features gebaut werden
    """
    prep_s = prepare_dataset_for_ticker(source)
    prep_t = prepare_dataset_for_ticker(target)

    trained = train_lstm_for_ticker(source, sequence_length=30)
    model = trained["model"]
    scaler = trained["scaler"]

    mae_source_test = trained["mae_test"]
    rmse_source_test = trained["rmse_test"]

    # Sequenzen aus UNskalierten Target-Features bauen
    X_val_seq_raw, y_val_seq = build_lstm_sequences(
        prep_t["X_val"], prep_t["y_val"], sequence_length=30
    )
    X_test_seq_raw, y_test_seq = build_lstm_sequences(
        prep_t["X_test"], prep_t["y_test"], sequence_length=30
    )

    pred_val = predict_lstm(model, scaler, X_val_seq_raw)
    pred_test = predict_lstm(model, scaler, X_test_seq_raw)

    return {
        "pred_val": pred_val.astype("float32"),
        "pred_test": pred_test.astype("float32"),
        "y_val": y_val_seq.astype("float32"),
        "y_test": y_test_seq.astype("float32"),
        "y_train": prep_s["y_train"].astype("float32"),
        "best_cfg": None,
        "mae_source_test": mae_source_test,
        "rmse_source_test": rmse_source_test,
    }


# =====================================================================
# PROPHET & ARIMA CROSS-DOMAIN
# =====================================================================

def cross_domain_prophet_wrapper(source: str, target: str):
    return cross_domain_prophet(source, target)


def cross_domain_arima_wrapper(source: str, target: str):
    return cross_domain_arima(source, target)


# =====================================================================
# Hilfsfunktionen
# =====================================================================

def make_row(
    model: str,
    source: str,
    target: str,
    res: dict,
    vol_test_full: np.ndarray,
) -> dict:
    """
    Baut eine Ergebniszeile für die Ergebnis-CSV.

    Kernmetriken:
        - MAE/RMSE auf Val und Test

    Cross-Domain-Metriken:
        - delta_mae, delta_rmse = |Loss_target - Loss_source|

    Volatilitätsanalyse (nur Target-Test-Set):
        - rmse_test_low_vol, rmse_test_mid_vol, rmse_test_high_vol
    """
    mae_val = mae(res["y_val"], res["pred_val"])
    rmse_val = rmse(res["y_val"], res["pred_val"])

    mae_test = mae(res["y_test"], res["pred_test"])
    rmse_test = rmse(res["y_test"], res["pred_test"])

    mae_source_test = res.get("mae_source_test", np.nan)
    rmse_source_test = res.get("rmse_source_test", np.nan)

    delta_mae = (
        abs(mae_test - mae_source_test)
        if not np.isnan(mae_source_test) else np.nan
    )
    delta_rmse = (
        abs(rmse_test - rmse_source_test)
        if not np.isnan(rmse_source_test) else np.nan
    )

    # Volatilitätsabhängige RMSE auf dem Target-Test-Set
    y_test = np.asarray(res["y_test"], dtype=float)
    preds_test = np.asarray(res["pred_test"], dtype=float)

    n_points = len(y_test)
    vol_aligned = vol_test_full[-n_points:]

    rmse_low, rmse_mid, rmse_high = rmse_by_vol_quantiles(
        y_true=y_test,
        y_pred=preds_test,
        vol=vol_aligned,
    )

    return {
        "scenario": "cross_domain",
        "source": source,
        "target": target,
        "model": model,

        # Validierung
        "mae_val": mae_val,
        "rmse_val": rmse_val,

        # Test gesamt
        "mae_test": mae_test,
        "rmse_test": rmse_test,

        # Cross-Domain-Delta (dL_fcst)
        "delta_mae": delta_mae,
        "delta_rmse": delta_rmse,

        # Volatilitätsabhängige Test-RMSE auf Target
        "rmse_test_low_vol": rmse_low,
        "rmse_test_mid_vol": rmse_mid,
        "rmse_test_high_vol": rmse_high,

        "best_cfg": res.get("best_cfg", None),
    }


# =====================================================================
# Hauptfunktion
# =====================================================================

def run_cross_domain():
    """
    Führt alle Cross-Domain-Kombinationen aus:

        - für jedes (source, target) mit source != target:
            ARIMA, Prophet, FFNN, LSTM

    und erzeugt:
        - vollständige Ergebnis-CSV inkl. Δ-Metriken und
          volatilitätsabhängiger Test-RMSE
        - pro Source→Target-Kombination einen gemeinsamen Test-Plot
    """
    rows = []

    for source in TICKERS:
        for target in TICKERS:
            if source == target:
                continue

            print(f"\n===== Cross-Domain: {source} → {target} =====")

            # Target-Daten (für Volatilität & Plot nur einmal laden)
            prep_target = prepare_dataset_for_ticker(target)
            test_df_target = prep_target["test_df"]

            if "roll_std_20" not in test_df_target.columns:
                raise KeyError(
                    "Spalte 'roll_std_20' wird für die Volatilitätsanalyse "
                    f"auf dem Target-Test-Set benötigt (Ticker: {target})."
                )

            vol_test_full = test_df_target["roll_std_20"].values.astype(float)

            # FFNN
            res_ffnn = cross_domain_ffnn(source, target)
            rows.append(make_row("FFNN", source, target, res_ffnn, vol_test_full))

            # LSTM
            res_lstm = cross_domain_lstm(source, target)
            rows.append(make_row("LSTM", source, target, res_lstm, vol_test_full))

            # ARIMA
            res_arima = cross_domain_arima_wrapper(source, target)
            rows.append(make_row("ARIMA", source, target, res_arima, vol_test_full))

            # Prophet
            res_prophet = cross_domain_prophet_wrapper(source, target)
            rows.append(make_row("Prophet", source, target, res_prophet, vol_test_full))

            # Gemeinsamer Test-Plot für dieses Source→Target-Paar
            model_results_for_pair = {
                "FFNN": res_ffnn,
                "LSTM": res_lstm,
                "ARIMA": res_arima,
                "Prophet": res_prophet,
            }
            ticker_label = f"{source}_to_{target}_cross"

            plot_test_set_all_models(
                ticker=ticker_label,
                prep=prep_target,
                model_results=model_results_for_pair,
                save_dir=PLOTS_DIR,
                zoom_last=PLOT_ZOOM_LAST,
            )

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "model_comparison_cross_domain.csv"
    df.to_csv(out, index=False)
    print(f"\nCross-Domain Ergebnisse gespeichert unter: {out}")
    return df


if __name__ == "__main__":
    df = run_cross_domain()
    print(df)
