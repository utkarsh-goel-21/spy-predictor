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

        # Fetch today's sentiment
        sentiment = fetch_spy_sentiment(last_date.to_pydatetime().replace(tzinfo=None))
        result["sentiment"] = sentiment

        # Combined signal
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