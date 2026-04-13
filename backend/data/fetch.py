import os
import re
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from backend.data.alphavantage import alphavantage_get_json, load_alphavantage_api_key

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
ALPHAVANTAGE_CACHE_TTL_SECONDS = 300
MARKET_DATA_SOURCE_CACHE_TTL_SECONDS = 900
LATEST_SOURCE_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
PERIOD_PATTERN = re.compile(r"^(?P<value>\d+)(?P<unit>d|mo|y)$")
TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"
FMP_EOD_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"


def load_optional_api_key(env_name: str) -> str | None:
    load_dotenv(ROOT_DIR / ".env")
    return os.getenv(env_name)


def load_twelve_data_api_key() -> str | None:
    return load_optional_api_key("TWELVE_DATA_API_KEY")


def load_fmp_api_key() -> str | None:
    return load_optional_api_key("FMP_API_KEY")


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


def expected_latest_daily_bar_date(now: pd.Timestamp | None = None) -> pd.Timestamp:
    ny_now = (now or pd.Timestamp.now(tz=ZoneInfo("America/New_York"))).tz_convert(ZoneInfo("America/New_York"))
    session_date = ny_now.normalize()

    if ny_now.weekday() >= 5:
        return (session_date - pd.offsets.BDay(1)).normalize()

    if ny_now.hour < 18:
        return (session_date - pd.offsets.BDay(1)).normalize()

    return session_date


def is_latest_daily_bar_fresh(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    latest_bar = pd.Timestamp(df.index.max()).normalize()
    return latest_bar >= expected_latest_daily_bar_date()


def latest_bar_date(df: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(df.index.max()).normalize()


def fetch_spy_from_yahoo(period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    ticker = yf.Ticker("SPY")
    df = ticker.history(period=period, interval=interval, auto_adjust=True)

    if df.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.attrs["source"] = "yahoo"
    df.attrs["last_refreshed"] = str(df.index.max().date())
    return df


def fetch_spy_from_twelve_data(period: str = "1y") -> pd.DataFrame:
    api_key = load_twelve_data_api_key()
    if not api_key:
        raise ValueError("TWELVE_DATA_API_KEY is not configured.")

    cache_key = "twelvedata_spy_daily"
    cached = LATEST_SOURCE_CACHE.get(cache_key)
    if cached and time.time() - cached[0] < MARKET_DATA_SOURCE_CACHE_TTL_SECONDS:
        return trim_df_to_period(cached[1].copy(), period)

    response = requests.get(
        TWELVE_DATA_URL,
        params={
            "symbol": "SPY",
            "interval": "1day",
            "outputsize": 500,
            "apikey": api_key,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("status") == "error":
        message = data.get("message") or data.get("code") or "unknown Twelve Data error"
        raise ValueError(f"Twelve Data error: {message}")

    values = data.get("values")
    if not values:
        raise ValueError("No daily time series returned from Twelve Data.")

    df = pd.DataFrame(values)
    df = df.rename(
        columns={
            "datetime": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    df["Date"] = pd.to_datetime(df["Date"])
    for column in ("Open", "High", "Low", "Close", "Volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df.set_index("Date").sort_index()
    df.index.name = "Date"
    df.attrs["source"] = "twelvedata"
    df.attrs["last_refreshed"] = str(df.index.max().date())
    LATEST_SOURCE_CACHE[cache_key] = (time.time(), df.copy())
    return trim_df_to_period(df, period)


def fetch_spy_from_fmp(period: str = "1y") -> pd.DataFrame:
    api_key = load_fmp_api_key()
    if not api_key:
        raise ValueError("FMP_API_KEY is not configured.")

    cache_key = "fmp_spy_daily"
    cached = LATEST_SOURCE_CACHE.get(cache_key)
    if cached and time.time() - cached[0] < MARKET_DATA_SOURCE_CACHE_TTL_SECONDS:
        return trim_df_to_period(cached[1].copy(), period)

    response = requests.get(
        FMP_EOD_URL,
        params={
            "symbol": "SPY",
            "apikey": api_key,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list) or not data:
        raise ValueError("No EOD series returned from FMP.")

    df = pd.DataFrame(data)
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError("FMP response missing OHLCV fields.")

    df = df.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    df["Date"] = pd.to_datetime(df["Date"])
    for column in ("Open", "High", "Low", "Close", "Volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df.set_index("Date").sort_index()
    df.index.name = "Date"
    df.attrs["source"] = "fmp"
    df.attrs["last_refreshed"] = str(df.index.max().date())
    LATEST_SOURCE_CACHE[cache_key] = (time.time(), df.copy())
    return trim_df_to_period(df, period)


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
    data = alphavantage_get_json(params=params, timeout=30)

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
        provider_chain = [
            ("alphavantage", fetch_spy_from_alphavantage),
            ("twelvedata", fetch_spy_from_twelve_data),
            ("fmp", fetch_spy_from_fmp),
            ("yahoo", lambda period_arg: fetch_spy_from_yahoo(period=period_arg, interval=interval)),
        ]
        successful_results: list[tuple[str, pd.DataFrame]] = []
        last_error: Exception | None = None

        for provider_name, fetcher in provider_chain:
            try:
                df = fetcher(period)
                last_date = latest_bar_date(df)
                print(f"{provider_name} latest SPY bar: {last_date.date()}")
                successful_results.append((provider_name, df))
                if is_latest_daily_bar_fresh(df):
                    return df
                print(
                    f"{provider_name} returned stale SPY daily data "
                    f"({last_date.date()}); trying next provider."
                )
            except Exception as exc:
                last_error = exc
                print(f"{provider_name} source failed: {exc}")

        if successful_results:
            freshest_provider, freshest_df = max(
                successful_results,
                key=lambda item: latest_bar_date(item[1]),
            )
            print(
                f"All latest-source providers were stale. Using freshest available data from "
                f"{freshest_provider} ({latest_bar_date(freshest_df).date()})."
            )
            return freshest_df

        if last_error is not None:
            raise RuntimeError("All latest-source SPY providers failed.") from last_error

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
