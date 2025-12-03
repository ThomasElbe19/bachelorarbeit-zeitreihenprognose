"""
ARIMA-Modell für:
    - In-Domain-Prognosen (AutoARIMA)
    - Cross-Domain-Prognosen (Strukturtransfer)

In-Domain:
    - AutoARIMA wählt (p,d,q) auf Trainingsdaten
    - Fit auf Train → Prognose für Validation
    - Fit auf Train+Val → Prognose für Test

Cross-Domain (Strukturtransfer):
    - Ordnung (p,d,q) wird auf Source bestimmt
    - dieselbe Ordnung wird auf Target-Daten neu gefittet
    - keine Parameterübertragung, nur Modellstruktur
"""

from __future__ import annotations
import warnings
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from pmdarima import auto_arima

from data_pipeline import prepare_dataset_for_ticker
from metrics import mae, rmse

warnings.simplefilter("ignore", ConvergenceWarning)


# =====================================================================
# 1) AutoARIMA – Ordnung bestimmen
# =====================================================================

def select_arima_order(train_series: np.ndarray) -> Tuple[int, int, int]:
    """Bestimmt die optimale ARIMA-Ordnung (p,d,q) mit AutoARIMA auf dem Quell-Datensatz."""
    model = auto_arima(
        y=train_series,
        seasonal=False,
        max_p=2,
        max_q=2,
        max_d=1,
        start_p=0,
        start_q=0,
        start_d=0,
        stepwise=True,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        information_criterion="aic",
    )
    return model.order  # (p, d, q)


# =====================================================================
# 2) In-Domain ARIMA
# =====================================================================

def train_arima_for_ticker(ticker: str) -> Dict[str, Any]:
    """In-Domain ARIMA für einen Ticker."""
    prep = prepare_dataset_for_ticker(ticker)

    train = prep["train_df"]["log_ret"].values.astype("float32")
    val   = prep["val_df"]["log_ret"].values.astype("float32")
    test  = prep["test_df"]["log_ret"].values.astype("float32")

    best_order = select_arima_order(train)

    # Validation
    model_train = ARIMA(train, order=best_order).fit()
    pred_val = np.asarray(model_train.forecast(len(val)), dtype="float32")

    # Test (Train+Val refitten)
    trainval = np.concatenate([train, val])
    model_tv = ARIMA(trainval, order=best_order).fit()
    pred_test = np.asarray(model_tv.forecast(len(test)), dtype="float32")

    return {
        "ticker": ticker,
        "best_cfg": best_order,

        "pred_val": pred_val,
        "pred_test": pred_test,

        "y_val": val,
        "y_test": test,

        # Kernmetriken
        "mae_val": mae(val, pred_val),
        "rmse_val": rmse(val, pred_val),
        "mae_test": mae(test, pred_test),
        "rmse_test": rmse(test, pred_test),
    }


# =====================================================================
# 3) Cross-Domain ARIMA (Strukturtransfer + dL_fcst-Infos)
# =====================================================================

def cross_domain_arima(source: str, target: str) -> Dict[str, Any]:
    """
    Cross-Domain-Variante für ARIMA:

        - AutoARIMA bestimmt (p,d,q) auf Source (In-Domain)
        - dieselbe Ordnung wird auf Target-Train / Train+Val gefittet
        - zusätzlich werden die In-Domain-Testfehler der Source
          (MAE/RMSE) zurückgegeben, um dL_fcst (Δ-Fehler) zu berechnen.
    """
    # In-Domain-Baseline auf der Quelle (für Δ-Metriken)
    src_res = train_arima_for_ticker(source)
    mae_source_test = src_res["mae_test"]
    rmse_source_test = src_res["rmse_test"]

    prep_t = prepare_dataset_for_ticker(target)
    train_t = prep_t["train_df"]["log_ret"].values.astype("float32")
    val_t   = prep_t["val_df"]["log_ret"].values.astype("float32")
    test_t  = prep_t["test_df"]["log_ret"].values.astype("float32")

    best_order = src_res["best_cfg"]

    # Fit auf Target-Train → Val
    model_val = ARIMA(train_t, order=best_order).fit()
    pred_val = np.asarray(model_val.forecast(len(val_t)), dtype="float32")

    # Fit auf Target-Train+Val → Test
    trainval_t = np.concatenate([train_t, val_t])
    model_test = ARIMA(trainval_t, order=best_order).fit()
    pred_test = np.asarray(model_test.forecast(len(test_t)), dtype="float32")

    return {
        "pred_val": pred_val,
        "pred_test": pred_test,
        "y_val": val_t,
        "y_test": test_t,
        "y_train": train_t,
        "best_cfg": best_order,

        "mae_source_test": mae_source_test,
        "rmse_source_test": rmse_source_test,
    }


if __name__ == "__main__":
    print("Starte ARIMA In-Domain Testtraining für IBM...")
    res = train_arima_for_ticker("IBM")
    print("Ordnung:", res["best_cfg"])
    print("MAE Test:", res["mae_test"])

    print("\nStarte ARIMA Cross-Domain Test (IBM → NVDA)...")
    res_cd = cross_domain_arima("IBM", "NVDA")
    from metrics import mae as _mae
    print("Cross-Domain MAE Test:", _mae(res_cd["y_test"], res_cd["pred_test"]))
