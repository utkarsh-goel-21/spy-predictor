from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.api.predict import predict_next_day
from backend.features.engineer import build_features, build_features_for_inference
from backend.data.fetch import fetch_spy_data
from backend.data.sentiment import fetch_spy_sentiment
from datetime import datetime
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

app = FastAPI(title="SPY Direction Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict():
    try:
        import pandas as pd
        raw = fetch_spy_data(period="1y")
        df = build_features_for_inference(raw, period="1y")
        result = predict_next_day(df)

        last_date = pd.Timestamp(df.index[-1])
        next_trading_day = last_date + pd.offsets.BDay(1)
        result["as_of_date"] = str(last_date.date())
        result["predicting_for"] = str(next_trading_day.date())

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
def model_info():
    try:
        feature_cols = joblib.load(MODELS_DIR / "lstm_feature_cols.pkl")
        config = joblib.load(MODELS_DIR / "lstm_config.pkl")
        return {
            "model": "LSTM",
            "sequence_length": 10,
            "features": feature_cols,
            "num_features": config["input_size"],
            "cv_accuracy": 0.5209,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chart-data")
def chart_data():
    try:
        import pandas as pd
        raw = fetch_spy_data(period="3mo")
        df = raw[["Close"]].copy()
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
        records = [
            {"date": str(idx.date()), "close": round(float(row["Close"]), 2)}
            for idx, row in df.iterrows()
        ]
        return {"data": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/news")
def news():
    try:
        from backend.data.sentiment import _fetch_for_date, load_dotenv, Path, os
        load_dotenv()
        load_dotenv(Path(".env"))
        import requests as req
        from datetime import datetime, timedelta
        import os

        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        date = datetime.now() - timedelta(days=3)
        time_from = date.strftime("%Y%m%dT0000")

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": "SPY",
            "time_from": time_from,
            "limit": 10,
            "apikey": api_key,
        }
        response = req.get("https://www.alphavantage.co/query", params=params)
        data = response.json()
        feed = data.get("feed", [])

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
        import pandas as pd
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

        up_accuracy   = round(sum(r["correct"] for r in up_preds)   / len(up_preds),   4) if up_preds   else None
        down_accuracy = round(sum(r["correct"] for r in down_preds) / len(down_preds), 4) if down_preds else None

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