from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.api.predict import predict_next_day
from backend.features.engineer import build_features
from backend.data.fetch import fetch_spy_data
import joblib
from pathlib import Path
from backend.features.engineer import build_features, build_features_for_inference

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