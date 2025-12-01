"""
model_prophet.py

Prophet-Modell für die Prognose der täglichen logarithmierten Rendite r_{t+1}.

In-Domain:
    - Prophet auf Trainingszeitraum fitten
    - Prognosen für Val- und Testzeitraum
    - Kernmetriken: MAE, RMSE

Cross-Domain (Strukturtransfer):
    - dieselbe Prophet-Struktur (Saisonalitäten, Trend-Settings) für alle Ticker
    - auf jedem Ziel-Ticker neu geschätzt (keine Parameterübertragung)
    - In-Domain-Testfehler der Quelle werden mitgegeben (für dL_fcst)
"""

from __future__ import annotations
from typing import Dict, Any

import numpy as np
import pandas as pd

from prophet import Prophet

from data_pipeline import prepare_dataset_for_ticker
from metrics import mae, rmse


# =============================================================================
# 1. Prophet-Modell definieren
# =============================================================================

def build_prophet_model(
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False,
    changepoint_prior_scale: float = 0.05,
) -> Prophet:
    """Erstellt ein Prophet-Modell mit Standardparametern für tägliche Finanzreihen."""
    m = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
    )
    return m


# =============================================================================
# 2. In-Domain: Training + Evaluation
# =============================================================================

def train_prophet_for_ticker(ticker: str) -> Dict[str, Any]:
    """Trainiert Prophet In-Domain und wertet auf Val- und Testdaten aus."""
    prep = prepare_dataset_for_ticker(ticker)
    train_df = prep["train_df"]
    val_df   = prep["val_df"]
    test_df  = prep["test_df"]

    y_train = train_df["log_ret"].values.astype("float32")
    y_val   = val_df["log_ret"].values.astype("float32")
    y_test  = test_df["log_ret"].values.astype("float32")

    df_train = pd.DataFrame({"ds": train_df.index, "y": y_train})

    model = build_prophet_model()
    model.fit(df_train)

    full_index = train_df.index.append(val_df.index).append(test_df.index)
    df_future = pd.DataFrame({"ds": full_index})

    forecast = model.predict(df_future).set_index("ds")
    yhat_series = forecast["yhat"]

    pred_train = yhat_series.loc[train_df.index].values.astype("float32")
    pred_val   = yhat_series.loc[val_df.index].values.astype("float32")
    pred_test  = yhat_series.loc[test_df.index].values.astype("float32")

    return {
        "ticker": ticker,
        "model": model,
        "forecast": forecast,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "pred_train": pred_train,
        "pred_val": pred_val,
        "pred_test": pred_test,

        # Kernmetriken
        "mae_val": mae(y_val, pred_val),
        "rmse_val": rmse(y_val, pred_val),
        "mae_test": mae(y_test, pred_test),
        "rmse_test": rmse(y_test, pred_test),
    }


# =============================================================================
# 3. Cross-Domain: Strukturtransfer Prophet
# =============================================================================

def cross_domain_prophet(source: str, target: str) -> Dict[str, Any]:
    """
    Cross-Domain-Variante für Prophet (Strukturtransfer):

        - In-Domain Prophet auf Source (für Baseline-Fehler)
        - dasselbe Modelldesign (Saisonalitäten etc.) für Target,
          dort neu geschätzt
        - gibt In-Domain-Testfehler der Quelle (MAE/RMSE) mit zurück
    """
    # In-Domain-Baseline auf Quelle
    src_res = train_prophet_for_ticker(source)
    mae_source_test = src_res["mae_test"]
    rmse_source_test = src_res["rmse_test"]

    # Target-Daten
    prep_t = prepare_dataset_for_ticker(target)
    train_t = prep_t["train_df"]
    val_t   = prep_t["val_df"]
    test_t  = prep_t["test_df"]

    y_train = train_t["log_ret"].values.astype("float32")
    y_val   = val_t["log_ret"].values.astype("float32")
    y_test  = test_t["log_ret"].values.astype("float32")

    df_train = pd.DataFrame({"ds": train_t.index, "y": y_train})

    model = build_prophet_model()
    model.fit(df_train)

    # Val
    df_val = pd.DataFrame({"ds": val_t.index})
    pred_val = model.predict(df_val)["yhat"].values.astype("float32")

    # Test
    df_test = pd.DataFrame({"ds": test_t.index})
    pred_test = model.predict(df_test)["yhat"].values.astype("float32")

    return {
        "pred_val": pred_val,
        "pred_test": pred_test,
        "y_val": y_val,
        "y_test": y_test,
        "y_train": y_train,
        "best_cfg": None,  # zur Kompatibilität mit ARIMA-Interface

        "mae_source_test": mae_source_test,
        "rmse_source_test": rmse_source_test,
    }


if __name__ == "__main__":
    print("Starte Prophet-In-Domain-Testtraining für IBM...")
    res = train_prophet_for_ticker("IBM")
    print("MAE Test :", res["mae_test"])
    print("RMSE Test:", res["rmse_test"])

    from metrics import mae as _mae
    print("\nStarte Prophet-Cross-Domain (IBM → NVDA)...")
    cd = cross_domain_prophet("IBM", "NVDA")
    print("Cross-Domain MAE Test:", _mae(cd["y_test"], cd["pred_test"]))
