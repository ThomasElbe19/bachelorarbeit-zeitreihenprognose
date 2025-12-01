"""
data_pipeline.py

Zentrale Datenpipeline für die Bachelorarbeit:
- Laden der Kursdaten (IBM, NVIDIA, Nike)
- Berechnung der logarithmierten Renditen
- Feature Engineering (Lags, Rolling-Statistiken, Volumenänderung, Kalenderdummies)
- Zeitbasierter Train/Validierung/Test-Split
- Standardisierung der Features (für FFNN und LSTM)

Alle Modelle (ARIMA, Prophet, FFNN, LSTM) sollen auf dieser Pipeline aufbauen,
damit der Vergleich konsistent und wissenschaftlich sauber ist.
"""

from __future__ import annotations

import numpy as np 
import pandas as pd 
from pathlib import Path
from typing import Dict, Tuple, List

from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1. Globale Einstellungen / Pfade
# =============================================================================

# Basisverzeichnis, in dem deine CSV-Dateien liegen
BASE_DATA_DIR = Path(
    r"C:\Users\thoma\Desktop\Wirtschaftsinformatik\Bachelorarbeit\Daten"
)

# Dateinamen der drei Aktien
DATA_FILES = {
    "IBM":  BASE_DATA_DIR / "IBM_Stock_1980_2025.csv",
    "NVDA": BASE_DATA_DIR / "NVDA.csv",
    "Nike": BASE_DATA_DIR / "Nike_historical_data.csv",
}

# Zeitliche Aufteilung (wie in der Arbeit definiert)
TRAIN_END = pd.Timestamp("2019-12-31")
VAL_END   = pd.Timestamp("2021-12-31")
# Test: alles > VAL_END bis 2025

# Feature-Parameter
N_LAGS = 30
ROLL_WINDOWS = (5, 20)


# =============================================================================
# 2. Hilfsfunktionen: Datum & Laden
# =============================================================================

def parse_date_col(date_series: pd.Series) -> pd.Series:
    """
    Wandelt eine Datums-Spalte robust in naive pandas-Timestamps um.
    Funktioniert auch, wenn Zeitzoneninformationen in der CSV enthalten sind.
    """
    dt = pd.to_datetime(date_series.astype(str), utc=True, errors="coerce")
    return dt.dt.tz_convert(None)


def load_stock(path: Path, ticker_name: str) -> pd.DataFrame:
    """
    Lädt einen Aktien-Datensatz von 'path' und gibt ein DataFrame mit
    Index = Datum und Spalten ['Close', 'Volume', 'ticker'] zurück.
    """
    df = pd.read_csv(path)

    # Datums-Spalte vereinheitlichen
    if "Date" not in df.columns:
        raise ValueError(f"Erwarte eine Spalte 'Date' in {path}, gefunden: {df.columns}")

    df["Date"] = parse_date_col(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Volume ggf. von Objekt mit Tausendertrennzeichen in Integer umwandeln
    if df["Volume"].dtype == object:
        df["Volume"] = (
            df["Volume"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype("int64")
        )

    if "Close" not in df.columns:
        raise ValueError(f"Erwarte eine Spalte 'Close' in {path}, gefunden: {df.columns}")

    df = df[["Date", "Close", "Volume"]].copy()
    df["ticker"] = ticker_name
    df.set_index("Date", inplace=True)

    return df


def load_all_stocks() -> Dict[str, pd.DataFrame]:
    """
    Lädt alle drei Aktien (IBM, NVIDIA, Nike) und gibt ein Dict
    { 'IBM': df_IBM, 'NVDA': df_NVDA, 'Nike': df_Nike } zurück.
    """
    stocks = {}
    for name, path in DATA_FILES.items():
        stocks[name] = load_stock(path, name)
    return stocks


# =============================================================================
# 3. Feature Engineering
# =============================================================================

def make_feature_table(
    price_df: pd.DataFrame,
    n_lags: int = N_LAGS,
    roll_windows: Tuple[int, int] = ROLL_WINDOWS,
) -> pd.DataFrame:
    """
    Erzeugt den Feature-Table für eine Aktie.

    Input:
        price_df:
            DataFrame mit Index = Datum, Spalten ['Close', 'Volume', 'ticker']

    Output:
        DataFrame mit:
        - Zielvariable: 'target_ret' (r_{t+1})
        - Features:
          * log_ret_lag_1 ... log_ret_lag_n
          * roll_mean_5/20, roll_std_5/20
          * log_vol_chg (logarithmierte Volumenänderung)
          * Kalender-Dummies (Wochentag, Monat)
          * Originalspalten 'Close', 'Volume', 'ticker' bleiben enthalten
            (können für Analysen genutzt werden, werden aber später aus X entfernt).
    """
    df = price_df.copy()

    # --- Log-Renditen r_t = ln(Close_t) - ln(Close_{t-1})
    df["log_ret"] = np.log(df["Close"]).diff()

    # --- Volumenänderung (logarithmiert, +1 verhindert log(0))
    df["log_vol_chg"] = np.log(df["Volume"].astype(float) + 1).diff()

    # --- Kalender-Features
    df["weekday"] = df.index.dayofweek  # 0=Montag, 6=Sonntag
    df["month"] = df.index.month

    # --- Lags der Renditen
    for lag in range(1, n_lags + 1):
        df[f"log_ret_lag_{lag}"] = df["log_ret"].shift(lag)

    # --- Rolling-Kennzahlen über log_ret
    for win in roll_windows:
        df[f"roll_mean_{win}"] = df["log_ret"].rolling(win).mean()
        df[f"roll_std_{win}"]  = df["log_ret"].rolling(win).std()

    # --- Zielvariable: r_{t+1}
    df["target_ret"] = df["log_ret"].shift(-1)

    # --- Dummy-Variablen für Wochentag & Monat
    dummies = pd.get_dummies(
        df[["weekday", "month"]].astype(int),
        columns=["weekday", "month"],
        prefix=["wd", "m"],
        drop_first=False,  # alle Dummies behalten, einfacher für Auswertung
    )
    df = pd.concat([df, dummies], axis=1)

    # Reihen mit NaN (durch diff, rolling, shift) entfernen
    df = df.dropna().copy()

    return df


# =============================================================================
# 4. Train/Validation/Test-Split & Feature-Matrizen
# =============================================================================

def split_by_date(
    feature_df: pd.DataFrame,
    train_end: pd.Timestamp = TRAIN_END,
    val_end: pd.Timestamp = VAL_END,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Teilt einen Feature-DataFrame in Train / Validation / Test nach Datum auf.
    """
    idx = feature_df.index

    train_df = feature_df[idx <= train_end]
    val_df   = feature_df[(idx > train_end) & (idx <= val_end)]
    test_df  = feature_df[idx > val_end]

    return train_df, val_df, test_df


def build_xy(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Erzeugt (X, y, feature_names) aus einem Feature-DataFrame.

    - y: 'target_ret'
    - X: alle numerischen Features (inkl. Lags, Rolling-Stats, Dummies),
         aber ohne 'target_ret', 'log_ret', 'Close', 'Volume', 'ticker'.
    """
    drop_cols = {"ticker", "Close", "Volume", "log_ret", "target_ret"}

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values.astype("float32")
    y = df["target_ret"].values.astype("float32")

    return X, y, feature_cols


def standardize_train_val_test(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardisiert Features (µ=0, σ=1) anhand der Trainingsdaten.
    Wird für FFNN und LSTM verwendet.

    Gibt zurück:
        X_train_scaled, X_val_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# =============================================================================
# 5. Convenience-Funktion: komplette Vorbereitung für einen Ticker
# =============================================================================

def prepare_dataset_for_ticker(
    ticker: str,
    start: str = "1999-01-01",
    end: str = "2025-12-31",
    n_lags: int = N_LAGS,
    roll_windows: Tuple[int, int] = ROLL_WINDOWS,
):
    """
    Komplettpaket für einen einzelnen Ticker:
      - Daten laden
      - Zeitraum filtern
      - Features erzeugen
      - Train/Val/Test-Split
      - X, y bauen
      - Standardisierung

    Rückgabe:
        dict mit Schlüsseln:
            'train_df', 'val_df', 'test_df',
            'X_train', 'X_val', 'X_test',
            'y_train', 'y_val', 'y_test',
            'X_train_scaled', 'X_val_scaled', 'X_test_scaled',
            'feature_names', 'scaler'
    """
    all_stocks = load_all_stocks()
    if ticker not in all_stocks:
        raise ValueError(f"Ticker '{ticker}' ist nicht in {list(all_stocks.keys())} enthalten.")

    price_df = all_stocks[ticker]
    price_df = price_df.loc[start:end].copy()

    feature_df = make_feature_table(price_df, n_lags=n_lags, roll_windows=roll_windows)

    train_df, val_df, test_df = split_by_date(feature_df)

    X_train, y_train, feature_names = build_xy(train_df)
    X_val,   y_val,   _            = build_xy(val_df)
    X_test,  y_test,  _            = build_xy(test_df)

    X_train_s, X_val_s, X_test_s, scaler = standardize_train_val_test(
        X_train, X_val, X_test
    )

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_scaled": X_train_s,
        "X_val_scaled": X_val_s,
        "X_test_scaled": X_test_s,
        "feature_names": feature_names,
        "scaler": scaler,
    }


# =============================================================================
# 6. Optionaler Selbsttest (kann bleiben oder entfernt werden)
# =============================================================================

if __name__ == "__main__":
    # Einfacher Testlauf für IBM, um zu prüfen, ob alles läuft
    prep = prepare_dataset_for_ticker("IBM")

    print("IBM – Anzahl Beobachtungen:")
    print("Train:", len(prep["y_train"]))
    print("Val  :", len(prep["y_val"]))
    print("Test :", len(prep["y_test"]))

    print("\nFeature-Namen (erste 10):")
    print(prep["feature_names"][:10])
