import os
import time
from datetime import datetime, timedelta
from pathlib import Path

from backend.data.alphavantage import alphavantage_get_json, load_alphavantage_api_key

NEWS_CACHE_TTL_SECONDS = 300
NEWS_CACHE: dict[str, tuple[float, list[dict]]] = {}


def fetch_recent_spy_feed(limit: int = 50, days_back: int = 7) -> list[dict] | None:
    api_key = load_alphavantage_api_key()
    if not api_key:
        return None

    cache_key = f"spy_feed_{limit}_{days_back}"
    cached = NEWS_CACHE.get(cache_key)
    if cached and time.time() - cached[0] < NEWS_CACHE_TTL_SECONDS:
        return cached[1]

    time_from = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y%m%dT0000")
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": "SPY",
        "time_from": time_from,
        "limit": limit,
        "apikey": api_key,
    }
    data = alphavantage_get_json(params=params, timeout=30)

    if "Error Message" in data or "Information" in data or "Note" in data:
        return None

    feed = data.get("feed", [])
    NEWS_CACHE[cache_key] = (time.time(), feed)
    return feed


def score_feed_for_date(feed: list[dict], date: datetime) -> dict | None:
    target_date = date.date()
    day_feed = []
    total_weight = 0.0
    weighted_score = 0.0

    for article in feed:
        published = article.get("time_published", "")
        try:
            published_dt = datetime.strptime(published, "%Y%m%dT%H%M%S")
        except ValueError:
            continue

        if published_dt.date() != target_date:
            continue

        spy_sentiment = next(
            (ticker_data for ticker_data in article.get("ticker_sentiment", []) if ticker_data["ticker"] == "SPY"),
            None,
        )
        if not spy_sentiment:
            continue

        relevance = float(spy_sentiment["relevance_score"])
        score = float(spy_sentiment["ticker_sentiment_score"])
        weighted_score += score * relevance
        total_weight += relevance
        day_feed.append(article)

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
        "article_count": len(day_feed),
        "sentiment_date": date.strftime("%Y-%m-%d"),
        "available": True,
    }


def _fetch_for_date(date: datetime) -> dict:
    """Fetch SPY sentiment for a single specific date from a cached recent feed."""
    feed = fetch_recent_spy_feed(limit=50, days_back=7)
    if not feed:
        return None
    return score_feed_for_date(feed, date)


def fetch_spy_sentiment(last_close_date: datetime) -> dict:
    """
    Try last close date first. If not available, try previous trading day.
    Always reports which date's sentiment was used.
    """
    last_close_normalized = last_close_date.replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    result = _fetch_for_date(last_close_normalized)
    if result:
        result["sentiment_source"] = "last_close"
        return result

    prev_day = last_close_normalized - timedelta(days=1)
    while prev_day.weekday() >= 5:
        prev_day -= timedelta(days=1)

    result = _fetch_for_date(prev_day)
    if result:
        result["sentiment_source"] = "previous_day"
        return result

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
