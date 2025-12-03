"""
experiment_execution.py

Zentrale Ausführungspipeline für die Bachelorarbeit.

Funktion:
    - führt alle Modelle (ARIMA, Prophet, FFNN, LSTM)
      für alle drei Aktien (IBM, NVDA, Nike) aus
    - sammelt die wichtigsten Kennzahlen (MAE, RMSE)
      für Validierungs- und Testdaten
    - ergänzt für das Test-Set eine volatilitätsabhängige Fehleranalyse:
      RMSE in drei Volatilitätsregimen (niedrig, mittel, hoch)
    - speichert die Ergebnisse als CSV-Datei im Ordner 'results'
    - erzeugt pro Ticker genau EINEN Plot auf dem Test-Set, in dem
      alle Modellvorhersagen gemeinsam mit den echten Werten liegen.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from model_arima import train_arima_for_ticker
from model_prophet import train_prophet_for_ticker
from model_ffnn import train_ffnn_for_ticker
from model_lstm import train_lstm_for_ticker

from data_pipeline import prepare_dataset_for_ticker
from plot_results import plot_test_set_all_models

from metrics import rmse_by_vol_quantiles

try:
    import tensorflow as tf
    np.random.seed(42)
    tf.random.set_seed(42)
except ImportError:
    pass


# =============================================================================
# 1. Konfiguration
# =============================================================================

TICKERS: List[str] = ["IBM", "NVDA", "Nike"]
MODELS: List[str] = ["ARIMA", "Prophet", "FFNN", "LSTM"]

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

PLOT_ZOOM_LAST: int | None = 250


# =============================================================================
# 2. Hilfsfunktion: ein Modell für einen Ticker ausführen
# =============================================================================

def run_model_for_ticker(model_name: str, ticker: str) -> Dict[str, Any]:
    """
    Führt das angegebene Modell für einen Ticker aus und
    gibt das vollständige Ergebnis-Dictionary zurück.

    Erwartet:
        - Jede train_*_for_ticker-Funktion liefert mindestens:
          'mae_val', 'rmse_val',
          'mae_test', 'rmse_test',
          'pred_val', 'pred_test',
          'y_val', 'y_test'
    """
    if model_name == "ARIMA":
        res = train_arima_for_ticker(ticker)
    elif model_name == "Prophet":
        res = train_prophet_for_ticker(ticker)
    elif model_name == "FFNN":
        res = train_ffnn_for_ticker(ticker)
    elif model_name == "LSTM":
        res = train_lstm_for_ticker(ticker, sequence_length=30)
    else:
        raise ValueError(f"Unbekannter Modellname: {model_name}")

    return res


# =============================================================================
# 3. Hauptfunktion: alle Experimente ausführen
# =============================================================================

def run_all_experiments() -> pd.DataFrame:
    """
    Führt alle Modell-Ticker-Kombinationen aus, erzeugt pro Ticker
    EINEN Test-Plot (alle Modelle gemeinsam) und speichert die
    Ergebnisse als CSV. Gibt zusätzlich das DataFrame zurück.

    Zusätzlich werden für das Test-Set volatilitätsabhängige RMSE-Werte
    (niedrige, mittlere, hohe Volatilität) mitgespeichert.
    """
    rows: List[Dict[str, Any]] = []

    for ticker in TICKERS:
        print(f"\n============================")
        print(f"Starte Experimente für {ticker}")
        print(f"============================")

        prep = prepare_dataset_for_ticker(ticker)
        test_df = prep["test_df"]

        # Rolling-Volatilität, die als σ_t in der Volatilitätsanalyse dienen soll
        # (hier: 20-Tage-Rolling-Std der log_ret, siehe data_pipeline.py)
        if "roll_std_20" not in test_df.columns:
            raise KeyError("Spalte 'roll_std_20' wird für die Volatilitätsanalyse benötigt.")
        vol_test_full = test_df["roll_std_20"].values.astype(float)

        model_results_for_ticker: Dict[str, Dict[str, Any]] = {}

        for model_name in MODELS:
            print(f"\n>>> Modell: {model_name} | Ticker: {ticker}")
            try:
                res = run_model_for_ticker(model_name, ticker)

                model_results_for_ticker[model_name] = res

                mae_val = res.get("mae_val", np.nan)
                rmse_val = res.get("rmse_val", np.nan)
                mae_test = res.get("mae_test", np.nan)
                rmse_test = res.get("rmse_test", np.nan)

                # Volatilitätsabhängige RMSE auf dem Test-Set
                # y_test / pred_test können je nach Modell unterschiedliche Längen haben
                y_test = res.get("y_test", None)
                pred_test = res.get("pred_test", None)

                if y_test is not None and pred_test is not None:
                    y_test_arr = np.asarray(y_test, dtype=float)
                    pred_test_arr = np.asarray(pred_test, dtype=float)

                    # Volatilität auf dieselbe Länge zuschneiden (Suffix)
                    n_points = len(y_test_arr)
                    vol_aligned = vol_test_full[-n_points:]

                    rmse_low, rmse_mid, rmse_high = rmse_by_vol_quantiles(
                        y_true=y_test_arr,
                        y_pred=pred_test_arr,
                        vol=vol_aligned,
                    )
                else:
                    rmse_low = rmse_mid = rmse_high = np.nan

                # Zeile für Ergebnis-Tabelle bauen
                row = {
                    "scenario": "in_domain",
                    "ticker": ticker,
                    "model": model_name,

                    # Validierungsmetriken
                    "mae_val": mae_val,
                    "rmse_val": rmse_val,

                    # Testmetriken (gesamt)
                    "mae_test": mae_test,
                    "rmse_test": rmse_test,

                    # Testmetriken nach Volatilitätsregimen
                    "rmse_test_low_vol": rmse_low,
                    "rmse_test_mid_vol": rmse_mid,
                    "rmse_test_high_vol": rmse_high,
                }
                rows.append(row)

                print(
                    f"Fertig: {model_name} auf {ticker} | "
                    f"MAE_test={row['mae_test']:.6f}, "
                    f"RMSE_test={row['rmse_test']:.6f}, "
                    f"RMSE_low/med/high="
                    f"{row['rmse_test_low_vol']:.6f}/"
                    f"{row['rmse_test_mid_vol']:.6f}/"
                    f"{row['rmse_test_high_vol']:.6f}"
                )

            except Exception as e:
                print(f"Fehler bei {model_name} auf {ticker}: {e}")
                rows.append({
                    "scenario": "in_domain",
                    "ticker": ticker,
                    "model": model_name,
                    "mae_val": np.nan,
                    "rmse_val": np.nan,
                    "mae_test": np.nan,
                    "rmse_test": np.nan,
                    "rmse_test_low_vol": np.nan,
                    "rmse_test_mid_vol": np.nan,
                    "rmse_test_high_vol": np.nan,
                })

        # EIN gemeinsamer Test-Plot mit allen vier Modellvorhersagen
        try:
            if model_results_for_ticker:
                plot_test_set_all_models(
                    ticker=ticker,
                    prep=prep,
                    model_results=model_results_for_ticker,
                    save_dir=PLOTS_DIR,
                    zoom_last=PLOT_ZOOM_LAST,
                )
            else:
                print(f"Keine Modellresultate für {ticker}, kein Plot erzeugt.")
        except Exception as pe:
            print(f"Warnung: Gemeinsamer Test-Plot für {ticker} fehlgeschlagen: {pe}")

    df_results = pd.DataFrame(rows)

    out_file = RESULTS_DIR / "model_comparison_in_domain.csv"
    df_results.to_csv(out_file, index=False)
    print(f"\nErgebnisse gespeichert unter: {out_file}")

    return df_results


# =============================================================================
# 4. Skript-Einstiegspunkt
# =============================================================================

if __name__ == "__main__":
    df = run_all_experiments()
    print("\nZusammenfassung:")
    print(df)
