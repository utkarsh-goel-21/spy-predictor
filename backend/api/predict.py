import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from backend.models.lstm_train import LSTMClassifier, SPYDataset, SEQUENCE_LENGTH, DEVICE

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def load_model():
    config = joblib.load(MODELS_DIR / "lstm_config.pkl")
    scaler = joblib.load(MODELS_DIR / "lstm_scaler.pkl")
    feature_cols = joblib.load(MODELS_DIR / "lstm_feature_cols.pkl")

    model = LSTMClassifier(input_size=config["input_size"]).to(DEVICE)
    model.load_state_dict(torch.load(MODELS_DIR / "lstm_model.pt", map_location=DEVICE))
    model.eval()

    return model, scaler, feature_cols


def predict_next_day(df: pd.DataFrame) -> dict:
    """
    Given a feature dataframe, predict next day SPY direction.
    Returns prediction (0=down, 1=up) and confidence score.
    """
    model, scaler, feature_cols = load_model()

    X = df[feature_cols].values
    X_scaled = scaler.transform(X)

    if len(X_scaled) < SEQUENCE_LENGTH:
        raise ValueError(f"Need at least {SEQUENCE_LENGTH} rows of data, got {len(X_scaled)}")

    # Use the last SEQUENCE_LENGTH rows as input sequence
    sequence = X_scaled[-SEQUENCE_LENGTH:]
    tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction])

    return {
        "prediction": prediction,
        "direction": "UP" if prediction == 1 else "DOWN",
        "confidence": round(confidence, 4),
        "prob_up": round(float(probs[1]), 4),
        "prob_down": round(float(probs[0]), 4),
    }