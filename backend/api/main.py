import logging
import threading
import time as time_module
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.api.predict import predict_next_day
from backend.data.fetch import fetch_spy_data
from backend.data.sentiment import fetch_spy_sentiment
from backend.features.engineer import build_features, build_features_for_inference

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
PROCESSED_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "spy_features.csv"
MARKET_TZ = ZoneInfo("America/New_York")
MARKET_CLOSE = time(hour=16, minute=0)
LOGGER = logging.getLogger(__name__)
SELF_PING_URL = "https://spy-predictor-api.onrender.com/health"
DASHBOARD_CACHE_TTL_SECONDS = 300
LIVE_DASHBOARD_CACHE: dict[str, tuple[float, dict]] = {}

app = FastAPI(title="SPY Direction Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_self_ping():
    def ping_loop():
        session = requests.Session()
        while True:
            time_module.sleep(600)
            try:
                response = session.get(SELF_PING_URL, timeout=30)
                response.raise_for_status()
            except Exception as exc:
                LOGGER.warning("backend self-ping failed for %s: %s", SELF_PING_URL, exc)

    thread = threading.Thread(target=ping_loop, daemon=True)
    thread.start()
    LOGGER.info("backend self-ping enabled: %s every 600s", SELF_PING_URL)


def keep_completed_daily_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop today's partial daily bar before the US market close."""
    if raw.empty or len(raw) == 1:
        return raw

    last_date = pd.Timestamp(raw.index[-1]).tz_localize(None)
    now_market = datetime.now(MARKET_TZ)

    if last_date.date() == now_market.date() and now_market.time() < MARKET_CLOSE:
        return raw.iloc[:-1]

    return raw


def next_trading_day_for(last_date: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(last_date) + pd.offsets.BDay(1)


def expected_latest_completed_bar_date(now_market: datetime | None = None) -> pd.Timestamp:
    now_market = now_market or datetime.now(MARKET_TZ)
    market_day = pd.Timestamp(now_market.date())
    if now_market.time() >= MARKET_CLOSE:
        return market_day
    return market_day - pd.offsets.BDay(1)


def load_training_dates() -> pd.DatetimeIndex:
    df = pd.read_csv(PROCESSED_DATA_PATH, usecols=["Date"], parse_dates=["Date"])
    return pd.DatetimeIndex(df["Date"]).tz_localize(None)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"status": "ok", "service": "spy-predictor-api", "docs": "/docs"}


def build_prediction_payload(raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    result = predict_next_day(df)
    latest_missing_external = df.attrs.get("latest_missing_external", [])

    raw_last_date = pd.Timestamp(raw.index[-1]).tz_localize(None)
    last_date = pd.Timestamp(df.index[-1])
    next_trading_day = next_trading_day_for(last_date)
    expected_last_date = expected_latest_completed_bar_date()

    result["market_data_source"] = raw.attrs.get("source", "unknown")
    result["market_data_last_refreshed"] = raw.attrs.get("last_refreshed")
    result["raw_data_last_date"] = str(raw_last_date.date())
    result["as_of_date"] = str(last_date.date())
    result["predicting_for"] = str(next_trading_day.date())
    result["expected_latest_bar_date"] = str(expected_last_date.date())

    if last_date < raw_last_date:
        result["market_data_status"] = "feature_lag"
        result["market_data_warning"] = (
            f"SPY data is available through {raw_last_date.date()}, but one or more external features "
            f"are only available through {last_date.date()}."
        )
    elif latest_missing_external:
        result["market_data_status"] = "feature_lag_filled"
        result["market_data_warning"] = (
            f"SPY data is available through {raw_last_date.date()}, but "
            f"{', '.join(latest_missing_external)} were carried forward from the last available day."
        )
    elif raw_last_date < expected_last_date:
        result["market_data_status"] = "source_lag"
        result["market_data_warning"] = (
            f"{result['market_data_source']} daily data is currently available only through {raw_last_date.date()}, "
            f"so the latest prediction is for {next_trading_day.date()}."
        )
    else:
        result["market_data_status"] = "current"
        result["market_data_warning"] = None

    sentiment = fetch_spy_sentiment(last_date.to_pydatetime().replace(tzinfo=None))
    result["sentiment"] = sentiment

    lstm_up = result["prediction"] == 1
    sent_score = sentiment["sentiment_score"]

    if lstm_up and sent_score >= 0.15:
        signal = "Strong UP"
    elif lstm_up and sent_score <= -0.15:
        signal = "Weak UP — sentiment disagrees"
    elif not lstm_up and sent_score <= -0.15:
        signal = "Strong DOWN"
    elif not lstm_up and sent_score >= 0.15:
        signal = "Weak DOWN — sentiment disagrees"
    else:
        signal = "Uncertain — sentiment neutral"

    result["combined_signal"] = signal
    return result


def build_model_info_payload() -> dict:
    feature_cols = joblib.load(MODELS_DIR / "lstm_feature_cols.pkl")
    config = joblib.load(MODELS_DIR / "lstm_config.pkl")
    return {
        "model": "LSTM",
        "sequence_length": config.get("sequence_length", 10),
        "features": feature_cols,
        "num_features": config["input_size"],
        "cv_accuracy": config.get("cv_accuracy", 0.5209),
    }


def build_chart_payload(raw: pd.DataFrame) -> dict:
    df = raw[["Close"]].copy()
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    cutoff = pd.Timestamp(df.index.max()) - pd.DateOffset(months=3)
    df = df[df.index >= cutoff]
    records = [
        {"date": str(idx.date()), "close": round(float(row["Close"]), 2)}
        for idx, row in df.iterrows()
    ]
    return {
        "data": records,
        "source": raw.attrs.get("source", "unknown"),
        "last_refreshed": raw.attrs.get("last_refreshed"),
    }


def build_news_payload() -> dict:
    from backend.data.sentiment import fetch_recent_spy_feed

    feed = fetch_recent_spy_feed(limit=10, days_back=7) or []
    articles = []
    for article in feed:
        spy_sentiment = next(
            (t for t in article.get("ticker_sentiment", []) if t["ticker"] == "SPY"),
            None
        )
        articles.append({
            "title": article.get("title", ""),
            "summary": article.get("summary", "")[:200] + "...",
            "source": article.get("source", ""),
            "url": article.get("url", ""),
            "time_published": article.get("time_published", ""),
            "sentiment_label": spy_sentiment["ticker_sentiment_label"] if spy_sentiment else "Neutral",
            "sentiment_score": float(spy_sentiment["ticker_sentiment_score"]) if spy_sentiment else 0.0,
        })
    return {"articles": articles}


def build_live_dashboard_payload() -> dict:
    cache_key = "live_dashboard"
    cached = LIVE_DASHBOARD_CACHE.get(cache_key)
    if cached and time_module.time() - cached[0] < DASHBOARD_CACHE_TTL_SECONDS:
        return cached[1]

    raw = fetch_spy_data(period="1y", source_preference="latest")
    raw = keep_completed_daily_bars(raw)
    df = build_features_for_inference(raw, period="1y")

    payload = {
        "prediction": build_prediction_payload(raw, df),
        "chart": build_chart_payload(raw),
        "news": build_news_payload(),
        "model_info": build_model_info_payload(),
    }
    LIVE_DASHBOARD_CACHE[cache_key] = (time_module.time(), payload)
    return payload


@app.get("/dashboard")
def dashboard():
    try:
        return build_live_dashboard_payload()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict")
def predict():
    try:
        return build_live_dashboard_payload()["prediction"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
def model_info():
    try:
        return build_model_info_payload()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chart-data")
def chart_data():
    try:
        return build_live_dashboard_payload()["chart"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/news")
def news():
    try:
        return build_live_dashboard_payload()["news"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest")
def backtest(start_date: str, end_date: str):
    """
    Backtest the LSTM model over a historical date range.

    Params:
        start_date: YYYY-MM-DD — first day to predict for
        end_date:   YYYY-MM-DD — last day to predict for (must be in the past)
    """
    try:
        from datetime import timedelta

        # ── 1. Parse and validate dates ──────────────────────────────
        try:
            start_dt = pd.Timestamp(start_date)
            end_dt   = pd.Timestamp(end_date)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

        today = pd.Timestamp(datetime.now().date())

        if end_dt >= today:
            raise HTTPException(status_code=400, detail="end_date must be before today — no actual data to compare against.")

        if start_dt >= end_dt:
            raise HTTPException(status_code=400, detail="start_date must be before end_date.")

        if (end_dt - start_dt).days > 366:
            raise HTTPException(status_code=400, detail="Date range cannot exceed 1 year.")

        # ── 2. Fetch data — start 60 calendar days early for sequence buffer ──
        fetch_start  = start_dt - timedelta(days=60)
        total_days   = (today - fetch_start).days
        period       = f"{total_days}d"

        raw = fetch_spy_data(period=period)
        df  = build_features_for_inference(raw, period=period)

        # ── 3. Keep raw close prices for actual direction calculation ──
        raw_close = raw[["Close"]].copy()
        raw_close.index = raw_close.index.tz_localize(None) if raw_close.index.tz else raw_close.index

        # ── 4. Get all trading days in the requested range ────────────
        all_dates    = df.index
        mask         = (all_dates >= start_dt) & (all_dates <= end_dt)
        target_dates = all_dates[mask]

        if len(target_dates) == 0:
            raise HTTPException(status_code=400, detail="No trading days found in the specified range.")

        # ── 5. Load model once, loop and predict ──────────────────────
        from backend.api.predict import load_model
        import torch
        import numpy as np
        from backend.models.lstm_train import SEQUENCE_LENGTH, DEVICE

        model, scaler, feature_cols = load_model()
        training_dates = load_training_dates()
        training_date_set = {d.date() for d in training_dates}
        training_start = pd.Timestamp(training_dates.min())
        training_end = pd.Timestamp(training_dates.max())

        results = []

        for date in target_dates:
            df_slice = df[df.index <= date]

            if len(df_slice) < SEQUENCE_LENGTH:
                continue

            X        = df_slice[feature_cols].values
            X_scaled = scaler.transform(X)
            sequence = X_scaled[-SEQUENCE_LENGTH:]
            tensor   = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(tensor)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred   = int(np.argmax(probs))

            predicted_direction = "UP" if pred == 1 else "DOWN"
            prob_up    = round(float(probs[1]), 4)
            prob_down  = round(float(probs[0]), 4)
            confidence = round(float(probs[pred]), 4)

            # Actual: compare next trading day close to this day's close
            future_closes = raw_close[raw_close.index > date]
            if len(future_closes) == 0:
                continue

            next_close    = float(future_closes.iloc[0]["Close"])
            current_close = float(raw_close[raw_close.index == date]["Close"].iloc[0])
            actual_direction = "UP" if next_close > current_close else "DOWN"

            correct = predicted_direction == actual_direction

            results.append({
                "date":                str(date.date()),
                "predicted_direction": predicted_direction,
                "actual_direction":    actual_direction,
                "correct":             correct,
                "in_sample":           date.date() in training_date_set,
                "confidence":          confidence,
                "prob_up":             prob_up,
                "prob_down":           prob_down,
            })

        if not results:
            raise HTTPException(status_code=400, detail="Could not generate predictions for the specified range.")

        # ── 6. Summary stats ──────────────────────────────────────────
        total         = len(results)
        correct_count = sum(r["correct"] for r in results)
        accuracy      = round(correct_count / total, 4)

        correct_confs = [r["confidence"] for r in results if r["correct"]]
        wrong_confs   = [r["confidence"] for r in results if not r["correct"]]

        up_preds   = [r for r in results if r["predicted_direction"] == "UP"]
        down_preds = [r for r in results if r["predicted_direction"] == "DOWN"]
        in_sample_results = [r for r in results if r["in_sample"]]
        out_sample_results = [r for r in results if not r["in_sample"]]

        up_accuracy   = round(sum(r["correct"] for r in up_preds)   / len(up_preds),   4) if up_preds   else None
        down_accuracy = round(sum(r["correct"] for r in down_preds) / len(down_preds), 4) if down_preds else None
        in_sample_accuracy = (
            round(sum(r["correct"] for r in in_sample_results) / len(in_sample_results), 4)
            if in_sample_results else None
        )
        out_sample_accuracy = (
            round(sum(r["correct"] for r in out_sample_results) / len(out_sample_results), 4)
            if out_sample_results else None
        )

        summary = {
            "total":                  total,
            "correct_count":          correct_count,
            "wrong_count":            total - correct_count,
            "accuracy":               accuracy,
            "avg_confidence_correct": round(sum(correct_confs) / len(correct_confs), 4) if correct_confs else None,
            "avg_confidence_wrong":   round(sum(wrong_confs)   / len(wrong_confs),   4) if wrong_confs   else None,
            "up_predictions":         len(up_preds),
            "down_predictions":       len(down_preds),
            "up_accuracy":            up_accuracy,
            "down_accuracy":          down_accuracy,
            "range_overlaps_training": bool(in_sample_results),
            "training_start_date":     str(training_start.date()),
            "training_end_date":       str(training_end.date()),
            "in_sample_count":         len(in_sample_results),
            "out_of_sample_count":     len(out_sample_results),
            "in_sample_accuracy":      in_sample_accuracy,
            "out_of_sample_accuracy":  out_sample_accuracy,
        }

        return {
            "start_date": start_date,
            "end_date":   end_date,
            "summary":    summary,
            "results":    results,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
