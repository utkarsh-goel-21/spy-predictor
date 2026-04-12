import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
ROOT_DIR = Path(__file__).resolve().parents[2]
ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"
ALPHAVANTAGE_CACHE_TTL_SECONDS = 300
LATEST_SOURCE_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
PERIOD_PATTERN = re.compile(r"^(?P<value>\d+)(?P<unit>d|mo|y)$")


def load_alphavantage_api_key() -> str | None:
    load_dotenv(ROOT_DIR / ".env")
    return os.getenv("ALPHAVANTAGE_API_KEY")


def trim_df_to_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    match = PERIOD_PATTERN.match(period)
    if not match or df.empty:
        return df

    value = int(match.group("value"))
    unit = match.group("unit")
    last_date = pd.Timestamp(df.index.max())

    if unit == "d":
        cutoff = last_date - pd.Timedelta(days=value)
    elif unit == "mo":
        cutoff = last_date - pd.DateOffset(months=value)
    else:
        cutoff = last_date - pd.DateOffset(years=value)

    trimmed = df[df.index >= cutoff].copy()
    return trimmed if not trimmed.empty else df


def fetch_spy_from_yahoo(period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    ticker = yf.Ticker("SPY")
    df = ticker.history(period=period, interval=interval, auto_adjust=True)

    if df.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.attrs["source"] = "yahoo"
    return df


def fetch_spy_from_alphavantage(period: str = "1y") -> pd.DataFrame:
    api_key = load_alphavantage_api_key()
    if not api_key:
        raise ValueError("ALPHAVANTAGE_API_KEY is not configured.")

    cache_key = "alphavantage_spy_daily_compact"
    cached = LATEST_SOURCE_CACHE.get(cache_key)
    if cached and time.time() - cached[0] < ALPHAVANTAGE_CACHE_TTL_SECONDS:
        return trim_df_to_period(cached[1].copy(), period)

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": "SPY",
        "outputsize": "compact",
        "apikey": api_key,
    }
    response = requests.get(ALPHAVANTAGE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
    if "Information" in data:
        raise ValueError(f"Alpha Vantage info response: {data['Information']}")
    if "Note" in data:
        raise ValueError(f"Alpha Vantage rate limit: {data['Note']}")

    series = data.get("Time Series (Daily)")
    meta = data.get("Meta Data", {})
    if not series:
        raise ValueError("No daily time series returned from Alpha Vantage.")

    df = pd.DataFrame.from_dict(series, orient="index", dtype=float)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.rename(
        columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume",
        }
    )
    df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index().dropna()
    df.attrs["source"] = "alphavantage"
    df.attrs["last_refreshed"] = meta.get("3. Last Refreshed")
    LATEST_SOURCE_CACHE[cache_key] = (time.time(), df.copy())
    return trim_df_to_period(df, period)


def fetch_spy_data(period: str = "5y", interval: str = "1d", source_preference: str = "yahoo") -> pd.DataFrame:
    """
    Fetch historical SPY OHLCV data.

    Args:
        period: How far back to fetch. E.g. '5y', '2y', '1y'.
        interval: Bar size. We use '1d' (daily) for this project.
        source_preference: "latest" uses Alpha Vantage first for live daily data
            and falls back to Yahoo. "yahoo" uses Yahoo directly.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    print(f"Fetching SPY data: period={period}, interval={interval}, source_preference={source_preference}")

    if source_preference == "latest" and interval == "1d":
        try:
            return fetch_spy_from_alphavantage(period=period)
        except Exception as exc:
            print(f"Alpha Vantage primary source failed: {exc}. Falling back to Yahoo Finance.")

    return fetch_spy_from_yahoo(period=period, interval=interval)


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
