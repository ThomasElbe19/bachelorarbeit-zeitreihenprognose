import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

path_ibm   = r"C:\Users\thoma\Desktop\Wirtschaftsinformatik\Bachelorarbeit\Daten\IBM_Stock_1980_2025.csv"
path_nike  = r"C:\Users\thoma\Desktop\Wirtschaftsinformatik\Bachelorarbeit\Daten\Nike_historical_data.csv"
path_nvda  = r"C:\Users\thoma\Desktop\Wirtschaftsinformatik\Bachelorarbeit\Daten\NVDA.csv"

save_path = r"C:\Users\thoma\Desktop\Wirtschaftsinformatik\Bachelorarbeit\Python\plots\rolling_volatility.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

def load_rolling_vol(path, ticker_name,
                     date_col="Date",
                     price_cols=("Adj Close", "Adj_Close", "Close", "close"),
                     window=60):
    df = pd.read_csv(path)

    # Datum parsen + nur kalendarisches Datum behalten
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df["Date_only"] = df[date_col].dt.date

    # nach Datum sortieren & index setzen
    df = df.sort_values("Date_only").set_index("Date_only")

    # passende Preisspalte finden
    price_col = None
    for c in price_cols:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        raise ValueError(f"Keine Preisspalte in {ticker_name} gefunden. Spalten: {df.columns.tolist()}")

    prices = df[price_col].astype(float)

    # Log-Renditen und Rolling-Volatilität
    log_returns = np.log(prices).diff()
    rolling_vol = log_returns.rolling(window=window).std() * np.sqrt(252)
    rolling_vol.name = ticker_name
    return rolling_vol

vol_ibm  = load_rolling_vol(path_ibm,  "IBM")
vol_nvda = load_rolling_vol(path_nvda, "NVIDIA")
vol_nike = load_rolling_vol(path_nike, "Nike")

# outer join über Kalendertage
vol_df = pd.concat([vol_ibm, vol_nvda, vol_nike], axis=1, join="outer")
vol_df.index = pd.to_datetime(vol_df.index)          # Index wieder als DatetimeIndex
vol_df = vol_df.loc["1999-01-01":"2025-12-31"]
vol_df = vol_df.dropna(how="all")

print("Shape nach Aufbereitung:", vol_df.shape)
print(vol_df.head())

plt.figure(figsize=(14, 6))
plt.plot(vol_df.index, vol_df["IBM"],    label="IBM",    linewidth=1.2)
plt.plot(vol_df.index, vol_df["NVIDIA"], label="NVIDIA", linewidth=1.2)
plt.plot(vol_df.index, vol_df["Nike"],   label="Nike",   linewidth=1.2)

plt.title("Rolling-Volatilität (60-Tage, annualisiert) von IBM, NVIDIA und Nike")
plt.xlabel("Datum")
plt.ylabel("Rolling-Volatilität (σ, annualisiert)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Plot erfolgreich gespeichert unter:\n{save_path}")
