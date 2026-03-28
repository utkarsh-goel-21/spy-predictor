import yfinance as yf
import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_spy_data(period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical SPY OHLCV data from Yahoo Finance.

    Args:
        period: How far back to fetch. E.g. '5y', '2y', '1y'.
        interval: Bar size. We use '1d' (daily) for this project.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    print(f"Fetching SPY data: period={period}, interval={interval}")
    ticker = yf.Ticker("SPY")
    df = ticker.history(period=period, interval=interval, auto_adjust=True)

    if df.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # Keep only what we need
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.dropna()

    return df


def save_raw_data(df: pd.DataFrame, filename: str = "spy_raw.csv") -> Path:
    """Save raw DataFrame to CSV."""
    path = RAW_DATA_DIR / filename
    df.to_csv(path)
    print(f"Saved {len(df)} rows to {path}")
    return path


if __name__ == "__main__":
    df = fetch_spy_data(period="5y")
    print(df.tail())
    save_raw_data(df)