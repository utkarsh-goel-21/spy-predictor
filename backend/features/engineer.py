import pandas as pd
import numpy as np
import ta
import yfinance as yf
from pathlib import Path

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_external(ticker: str, period: str = "5y", name: str = "") -> pd.Series:
    """Fetch daily close for any ticker and return as a named Series."""
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df["Close"].rename(name)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add only high-signal, low-redundancy technical indicators."""
    df = df.copy()

    # Momentum
    df["rsi_14"] = ta.momentum.rsi(df["Close"], window=14)
    df["roc_10"] = ta.momentum.roc(df["Close"], window=10)

    # Trend
    df["macd_diff"] = ta.trend.macd_diff(df["Close"])

    # Volatility
    df["atr_14"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"])
    df["bb_width"] = ta.volatility.bollinger_wband(df["Close"])

    # Price-derived
    df["daily_return"] = df["Close"].pct_change()
    df["hl_spread"] = (df["High"] - df["Low"]) / df["Close"]

    # Lagged returns
    df["return_lag1"] = df["daily_return"].shift(1)
    df["return_lag2"] = df["daily_return"].shift(2)
    df["return_lag3"] = df["daily_return"].shift(3)
    df["return_lag5"] = df["daily_return"].shift(5)

    # Day of week
    df["day_of_week"] = df.index.dayofweek

    return df


def add_external_context(df: pd.DataFrame, period: str = "5y") -> pd.DataFrame:
    """
    Add lagged returns from European and Asian markets.
    We use lag-1 (previous day close) because these markets
    close before the US open — no lookahead leakage.
    """
    df = df.copy()

    external = {
        "dax_return": "^GDAXI",
        "ftse_return": "^FTSE",
        "nikkei_return": "^N225",
    }

    for name, ticker in external.items():
        print(f"Fetching {name} ({ticker})...")
        series = fetch_external(ticker, period=period, name=ticker)
        pct = series.pct_change().rename(name)
        pct = pct.shift(1)
        df = df.join(pct, how="left")

    # VIX
    print("Fetching VIX...")
    vix = fetch_external("^VIX", period=period, name="vix_close")
    df = df.join(vix, how="left")
    df["vix_change"] = df["vix_close"].pct_change()

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Binary target: 1 if next day Close > today Close, else 0."""
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.iloc[:-1]  # drop last row, no known target
    return df


def build_features(df: pd.DataFrame, period: str = "5y") -> pd.DataFrame:
    """Full pipeline: indicators + external context + target + clean."""
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = add_technical_indicators(df)
    df = add_external_context(df, period=period)
    df = add_target(df)
    df = df.dropna()
    return df


def save_processed(df: pd.DataFrame, filename: str = "spy_features.csv") -> Path:
    path = PROCESSED_DATA_DIR / filename
    df.to_csv(path)
    print(f"Saved {len(df)} rows, {len(df.columns)} columns to {path}")
    return path


if __name__ == "__main__":
    from backend.data.fetch import fetch_spy_data

    raw = fetch_spy_data(period="5y")
    featured = build_features(raw, period="5y")
    print(featured.tail())
    print(f"\nFeatures: {list(featured.columns)}")
    print(f"Target distribution:\n{featured['target'].value_counts()}")
    save_processed(featured)