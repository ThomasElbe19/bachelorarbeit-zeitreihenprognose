"""
plot_results.py

Erstellt Plots für tatsächliche Daten (y_true) und Modellvorhersagen (y_pred).

In-Domain:
    - plot_predictions: Einzelner Plot für ein Modell und einen Split (Val/Test)
    - plot_test_set_all_models: EIN Plot pro Ticker mit allen Modellen (nur Test-Set)

Wird von experiment_execution.py und experiment_cross_domain.py verwendet.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def plot_predictions(
    ticker: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index: pd.Index,
    save_dir: Path = Path("plots"),
    split_name: str = "test",
    zoom_last: int | None = None,
):
    """
    Erstellt einen Plot:
        - y_true vs. y_pred im angegebenen Zeitraum (Val/Test)
        - speichert ihn automatisch als PNG

    Parameter:
        ticker: Aktienkürzel (IBM, NVDA, Nike)
        model_name: ARIMA / Prophet / FFNN / LSTM
        y_true: echte Werte
        y_pred: Vorhersagen
        index: Datumsindex des jeweiligen Splits
        save_dir: Ordner zum Speichern der Plots
        split_name: 'val' / 'test'
        zoom_last: Wenn gesetzt, nur die letzten N Punkte anzeigen
    """

    save_dir.mkdir(exist_ok=True)

    # Falls Zoom gewünscht: nur letzte N Punkte anzeigen
    if zoom_last is not None and zoom_last < len(y_true):
        y_true = y_true[-zoom_last:]
        y_pred = y_pred[-zoom_last:]
        index = index[-zoom_last:]

    plt.figure(figsize=(12, 5))
    plt.plot(index, y_true, label="Echte Werte", linewidth=1.5)
    plt.plot(index, y_pred, label="Vorhersage", linewidth=1.2)

    plt.title(f"{model_name} – {ticker} – {split_name.capitalize()}-Set")
    plt.xlabel("Datum")
    plt.ylabel("Log-Rendite")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_path = save_dir / f"{ticker}_{model_name}_{split_name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    print(f"Plot gespeichert unter: {out_path}")


def plot_test_set_all_models(
    ticker: str,
    prep: dict,
    model_results: dict,
    save_dir: Path = Path("plots"),
    zoom_last: int | None = None,
):
    """
    Erstellt EINEN Plot pro Ticker für das Test-Set, in dem:
        - die echten Testwerte
        - die Test-Vorhersagen von ARIMA, Prophet, FFNN, LSTM

    gemeinsam dargestellt werden.

    Parameter:
        ticker        : Aktienkürzel (IBM, NVDA, Nike)
        prep          : Output von prepare_dataset_for_ticker(...)
                        (wird für y_test und den Test-Index genutzt)
        model_results : Dict[model_name -> results_dict]
                        results_dict muss 'pred_test' enthalten
        save_dir      : Ordner zum Speichern der Plots
        zoom_last     : Wenn gesetzt, nur die letzten N Punkte anzeigen
    """

    save_dir.mkdir(exist_ok=True)

    # Voller Testindex & echte Testwerte
    test_idx_full: pd.Index = prep["test_df"].index
    y_true_full: np.ndarray = prep["y_test"]

    # Gemeinsame Länge über alle vorhandenen Modelle bestimmen
    lengths = [len(y_true_full)]
    for name in ["ARIMA", "Prophet", "FFNN", "LSTM"]:
        if name in model_results and "pred_test" in model_results[name]:
            lengths.append(len(model_results[name]["pred_test"]))

    if len(lengths) == 1:
        raise ValueError("Keine Test-Vorhersagen in model_results gefunden.")

    n_points = min(lengths)

    # Optional zusätzlich zoomen (nur letzte N Punkte)
    if zoom_last is not None:
        n_points = min(n_points, zoom_last)

    # Gemeinsamer Index & y_true auf die letzten n_points beschränken
    index = test_idx_full[-n_points:]
    y_true = y_true_full[-n_points:]

    plt.figure(figsize=(12, 5))

    # Wahre Werte (Test-Set) – schwarze Linie, etwas dicker
    plt.plot(
        index,
        y_true,
        label="Wahre Werte (Test)",
        linewidth=1.4,
        color="black",
    )

    # Farben und Reihenfolge der Modelle
    model_colors = {
        "ARIMA": "tab:blue",
        "Prophet": "tab:orange",
        "FFNN": "tab:green",
        "LSTM": "tab:red",
    }

    # Modelle als gestrichelte Linien hinzufügen
    for name in ["ARIMA", "Prophet", "FFNN", "LSTM"]:
        if name not in model_results:
            continue
        res = model_results[name]
        if "pred_test" not in res:
            continue

        preds_full = res["pred_test"]
        preds = preds_full[-n_points:]

        plt.plot(
            index,
            preds,
            linestyle="--",        # gestrichelt
            linewidth=1.0,         # eher dünn
            label=name,
            color=model_colors.get(name, None),
        )

    plt.title(f"Test-Set: {ticker} – Vorhersagen aller Modelle")
    plt.xlabel("Datum")
    plt.ylabel("Log-Rendite")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = save_dir / f"{ticker}_all_models_test.png"
    plt.savefig(out_path, dpi=220)
    plt.close()

    print(f"Test-Plot (alle Modelle) gespeichert unter: {out_path}")


def plot_all_splits(
    ticker: str,
    model_name: str,
    prep: dict,
    results: dict,
    save_dir: Path = Path("plots"),
    zoom_last: int | None = None,
):
    """
    (Alt / optional – wird in der aktuellen In-Domain-Pipeline nicht mehr verwendet)

    Erstellt Plots für:
        - Validation (y_val vs pred_val)
        - Test (y_test vs pred_test)

    Parameter:
        ticker     : Aktienkürzel
        model_name : Modellname (ARIMA, Prophet, FFNN, LSTM)
        prep       : Output von prepare_dataset_for_ticker(...)
        results    : Output von train_*_for_ticker(...), enthält pred_val/pred_test
    """

    # Datumsindizes aus der Pipeline
    val_idx = prep["val_df"].index
    test_idx = prep["test_df"].index

    # VALIDATION
    plot_predictions(
        ticker=ticker,
        model_name=model_name,
        y_true=prep["y_val"],
        y_pred=results["pred_val"],
        index=val_idx,
        save_dir=save_dir,
        split_name="val",
        zoom_last=zoom_last,
    )

    # TEST
    plot_predictions(
        ticker=ticker,
        model_name=model_name,
        y_true=prep["y_test"],
        y_pred=results["pred_test"],
        index=test_idx,
        save_dir=save_dir,
        split_name="test",
        zoom_last=zoom_last,
    )
