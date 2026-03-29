import requests
from datetime import datetime, timedelta
import os
import time
from dotenv import load_dotenv
from pathlib import Path


def _fetch_for_date(date: datetime) -> dict:
    """Fetch SPY sentiment for a single specific date."""
    load_dotenv()
    load_dotenv(Path(".env"))
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")

    time_from = date.strftime("%Y%m%dT0000")
    next_day = date + timedelta(days=1)
    time_to = next_day.strftime("%Y%m%dT0000")

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": "SPY",
        "time_from": time_from,
        "time_to": time_to,
        "limit": 50,
        "apikey": api_key,
    }

    response = requests.get("https://www.alphavantage.co/query", params=params)
    data = response.json()

    # Handle API errors
    if "Error Message" in data or "Information" in data:
        return None

    feed = data.get("feed", [])
    if not feed:
        return None

    total_weight = 0.0
    weighted_score = 0.0

    for article in feed:
        for ticker_data in article.get("ticker_sentiment", []):
            if ticker_data["ticker"] == "SPY":
                relevance = float(ticker_data["relevance_score"])
                score = float(ticker_data["ticker_sentiment_score"])
                weighted_score += score * relevance
                total_weight += relevance

    if total_weight == 0:
        return None

    final_score = weighted_score / total_weight

    if final_score <= -0.35:
        label = "Bearish"
    elif final_score <= -0.15:
        label = "Somewhat-Bearish"
    elif final_score < 0.15:
        label = "Neutral"
    elif final_score < 0.35:
        label = "Somewhat-Bullish"
    else:
        label = "Bullish"

    return {
        "sentiment_score": round(final_score, 4),
        "sentiment_label": label,
        "article_count": len(feed),
        "sentiment_date": date.strftime("%Y-%m-%d"),
        "available": True,
    }


def fetch_spy_sentiment(last_close_date: datetime) -> dict:
    """
    Try last close date first. If not available, try previous trading day.
    Always reports which date's sentiment was used.
    """
    last_close_normalized = last_close_date.replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    # Step 1: Try last close date
    result = _fetch_for_date(last_close_normalized)
    if result:
        result["sentiment_source"] = "last_close"
        return result

    # Step 2: Fall back to previous trading day
    time.sleep(1)
    prev_day = last_close_normalized - timedelta(days=1)
    # Skip weekends
    while prev_day.weekday() >= 5:
        prev_day -= timedelta(days=1)

    result = _fetch_for_date(prev_day)
    if result:
        result["sentiment_source"] = "previous_day"
        return result

    # Step 3: Nothing available
    return {
        "sentiment_score": 0.0,
        "sentiment_label": "Neutral",
        "article_count": 0,
        "sentiment_date": last_close_normalized.strftime("%Y-%m-%d"),
        "sentiment_source": "unavailable",
        "available": False,
    }


if __name__ == "__main__":
    result = fetch_spy_sentiment(datetime(2026, 3, 27))
    print(f"Result: {result}")