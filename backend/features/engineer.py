import pandas as pd
import ta
from pathlib import Path

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to raw OHLCV data.
    All indicators use only past data — no lookahead.
    """
    df = df.copy()

    # Trend
    df["ema_10"] = ta.trend.ema_indicator(df["Close"], window=10)
    df["ema_20"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["ema_50"] = ta.trend.ema_indicator(df["Close"], window=50)
    df["macd"] = ta.trend.macd(df["Close"])
    df["macd_signal"] = ta.trend.macd_signal(df["Close"])
    df["macd_diff"] = ta.trend.macd_diff(df["Close"])

    # Momentum
    df["rsi_14"] = ta.momentum.rsi(df["Close"], window=14)
    df["stoch_k"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"])
    df["stoch_d"] = ta.momentum.stoch_signal(df["High"], df["Low"], df["Close"])
    df["roc_10"] = ta.momentum.roc(df["Close"], window=10)

    # Volatility
    df["bb_high"] = ta.volatility.bollinger_hband(df["Close"])
    df["bb_low"] = ta.volatility.bollinger_lband(df["Close"])
    df["bb_width"] = ta.volatility.bollinger_wband(df["Close"])
    df["atr_14"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"])

    # Volume
    df["obv"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
    df["vwap"] = ta.volume.volume_weighted_average_price(
        df["High"], df["Low"], df["Close"], df["Volume"]
    )

    # Price-derived
    df["daily_return"] = df["Close"].pct_change()
    df["hl_spread"] = (df["High"] - df["Low"]) / df["Close"]
    df["close_vs_ema20"] = (df["Close"] - df["ema_20"]) / df["ema_20"]

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary target: 1 if next day's Close > today's Close, else 0.
    This is what we are predicting.
    """
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: indicators + target + drop NaNs."""
    df = add_technical_indicators(df)
    df = add_target(df)
    df = df.dropna()
    # Drop last row — target is unknown (no next day yet)
    df = df.iloc[:-1]
    return df


def save_processed(df: pd.DataFrame, filename: str = "spy_features.csv") -> Path:
    path = PROCESSED_DATA_DIR / filename
    df.to_csv(path)
    print(f"Saved {len(df)} rows, {len(df.columns)} columns to {path}")
    return path


if __name__ == "__main__":
    from backend.data.fetch import fetch_spy_data

    raw = fetch_spy_data(period="5y")
    featured = build_features(raw)
    print(featured.tail())
    print(f"\nFeatures: {list(featured.columns)}")
    print(f"Target distribution:\n{featured['target'].value_counts()}")
    save_processed(featured)