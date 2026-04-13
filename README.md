# SPY Direction Predictor

A full-stack ML application that predicts next-trading-day SPY (S&P 500 ETF) direction using an LSTM neural network, technical indicators, global market context, and real-time news sentiment analysis.

---

## What It Does

Get daily predictions for SPY's next-day movement:

- _"Will SPY go UP or DOWN tomorrow?"_
- _"What's the model's confidence level?"_
- _"Does news sentiment agree with the prediction?"_
- _"How has the model performed over the last 90 days?"_

The system combines:
1. **LSTM model** trained on 5 years of historical data
2. **20+ technical indicators** (RSI, MACD, ATR, Bollinger Bands, etc.)
3. **Global market context** (DAX, FTSE, Nikkei, VIX)
4. **Real-time news sentiment** via Alpha Vantage API

Returns a prediction with confidence score, sentiment analysis, and a combined signal strength.

---

## Architecture Overview

```
User clicks "Run Prediction"
              ↓
    ┌─────────────────┐
    │  Fetch SPY Data │  ← yfinance (1 year OHLCV)
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │ Feature Engineer│  ← Technical indicators + global markets
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │   LSTM Model    │  ← 10-day sequence → UP/DOWN + confidence
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │ News Sentiment  │  ← Alpha Vantage API
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │ Combined Signal │  ← Merge model + sentiment
    └─────────────────┘
             ↓
      Dashboard Display
```

### Signal Logic

| Model | Sentiment | Combined Signal |
|-------|-----------|-----------------|
| UP | Bullish (≥0.15) | **Strong UP** |
| UP | Bearish (≤-0.15) | Weak UP — sentiment disagrees |
| DOWN | Bearish (≤-0.15) | **Strong DOWN** |
| DOWN | Bullish (≥0.15) | Weak DOWN — sentiment disagrees |
| Any | Neutral | Uncertain — sentiment neutral |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI, Uvicorn |
| **ML Model** | PyTorch LSTM (2-layer, 64 hidden units) |
| **Feature Engineering** | pandas, ta (technical analysis library) |
| **Data Source** | yfinance (Yahoo Finance API) |
| **Sentiment** | Alpha Vantage News Sentiment API |
| **Frontend** | Streamlit, Plotly |
| **Serialization** | joblib (scalers, config), torch.save (model) |

---

## Features Used

The LSTM model uses **20 engineered features** in 10-day sequences:

| Category | Features |
|----------|----------|
| **Momentum** | RSI (14), ROC (10), MACD diff |
| **Volatility** | ATR (14), Bollinger Band width |
| **Price-derived** | Daily return, High-Low spread |
| **Lagged returns** | 1, 2, 3, 5-day lagged returns |
| **Calendar** | Day of week (0-4) |
| **Global markets** | DAX, FTSE, Nikkei (lagged 1-day returns) |
| **Fear index** | VIX close, VIX change |

---

## Project Structure

```
spy-predictor/
├── backend/
│   ├── api/
│   │   ├── main.py              # FastAPI app — /predict, /backtest, /chart-data, /news
│   │   └── predict.py           # Model loading and inference
│   │
│   ├── data/
│   │   ├── fetch.py             # yfinance data fetcher
│   │   └── sentiment.py         # Alpha Vantage news sentiment
│   │
│   ├── features/
│   │   └── engineer.py          # Technical indicators + global context
│   │
│   └── models/
│       ├── lstm_train.py        # LSTM model definition + training
│       ├── train.py             # RandomForest baseline (legacy)
│       ├── lstm_model.pt        # Trained LSTM weights
│       ├── lstm_scaler.pkl      # StandardScaler for features
│       ├── lstm_feature_cols.pkl # Feature column names
│       └── lstm_config.pkl      # Model config (input_size)
│
├── frontend/
│   ├── app.py                   # Streamlit dashboard
│   └── requirements.txt         # Frontend dependencies
│
├── data/                        # Local data cache (gitignored)
│   ├── raw/
│   └── processed/
│
├── requirements.txt             # Python dependencies
├── .env                         # API keys (gitignored)
└── .env.example                 # Example environment variables
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- An [Alpha Vantage API key](https://www.alphavantage.co/support/#api-key) (free)

---

### Backend Setup

#### 1. Clone the repository

```bash
git clone <your-repo-url>
cd spy-predictor
```

#### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Configure environment variables

Create a `.env` file in the root directory:

```env
ALPHAVANTAGE_API_KEY=your_api_key_here
TWELVE_DATA_API_KEY=your_api_key_here
FMP_API_KEY=your_api_key_here
```

For the live daily `SPY` bar, the backend now tries `Alpha Vantage -> Twelve Data -> FMP -> Yahoo Finance` and keeps moving down the chain if an earlier provider is unavailable or stale.

#### 5. Run the backend server

```bash
uvicorn backend.api.main:app --reload
```

The API starts at `http://127.0.0.1:8000`.

---

### Frontend Setup

#### 1. Navigate to the frontend directory

```bash
cd frontend
```

#### 2. Run the Streamlit app

```bash
streamlit run app.py
```

Frontend runs at `http://localhost:8501`. The backend must also be running.

---

## API Reference

### `GET /health`
Health check. Returns `{"status": "ok"}`.

### `GET /predict`
Main prediction endpoint. Fetches live SPY data, generates features, runs LSTM inference, and fetches news sentiment.

**Response:**
```json
{
  "prediction": 1,
  "direction": "UP",
  "confidence": 0.5842,
  "prob_up": 0.5842,
  "prob_down": 0.4158,
  "as_of_date": "2026-03-31",
  "predicting_for": "2026-04-01",
  "sentiment": {
    "sentiment_score": 0.1823,
    "sentiment_label": "Somewhat-Bullish",
    "article_count": 12,
    "sentiment_date": "2026-03-31",
    "available": true
  },
  "combined_signal": "Strong UP"
}
```

### `GET /model-info`
Returns model metadata.

```json
{
  "model": "LSTM",
  "sequence_length": 10,
  "features": ["rsi_14", "roc_10", "macd_diff", ...],
  "num_features": 20,
  "cv_accuracy": 0.5209
}
```

### `GET /chart-data`
Returns 3 months of SPY closing prices for charting.

```json
{
  "data": [
    {"date": "2026-01-02", "close": 485.23},
    {"date": "2026-01-03", "close": 487.45},
    ...
  ]
}
```

### `GET /news`
Returns recent SPY-related news with sentiment scores.

```json
{
  "articles": [
    {
      "title": "S&P 500 rallies on Fed comments",
      "summary": "Markets surged after...",
      "source": "Reuters",
      "url": "https://...",
      "time_published": "20260331T143000",
      "sentiment_label": "Bullish",
      "sentiment_score": 0.42
    }
  ]
}
```

### `GET /backtest`
Backtest the model over a historical date range.

**Parameters:**
- `start_date`: YYYY-MM-DD — first day to predict for
- `end_date`: YYYY-MM-DD — last day (must be in the past)

**Response:**
```json
{
  "start_date": "2026-01-01",
  "end_date": "2026-03-30",
  "summary": {
    "total": 62,
    "correct_count": 34,
    "wrong_count": 28,
    "accuracy": 0.5484,
    "avg_confidence_correct": 0.5621,
    "avg_confidence_wrong": 0.5512,
    "up_predictions": 45,
    "down_predictions": 17,
    "up_accuracy": 0.5556,
    "down_accuracy": 0.5294
  },
  "results": [
    {
      "date": "2026-01-02",
      "predicted_direction": "UP",
      "actual_direction": "UP",
      "correct": true,
      "confidence": 0.5723,
      "prob_up": 0.5723,
      "prob_down": 0.4277
    }
  ]
}
```

---

## Training the Model

To retrain the LSTM model on fresh data:

```bash
# 1. Fetch raw SPY data
python -m backend.data.fetch

# 2. Generate features
python -m backend.features.engineer

# 3. Train the LSTM model
python -m backend.models.lstm_train
```

Training uses:
- **5-fold TimeSeriesSplit** cross-validation
- **Early stopping** with patience=10
- **Adam optimizer** with weight decay (1e-4)
- **Cross-entropy loss** for binary classification

Model artifacts saved to `backend/models/`:
- `lstm_model.pt` — trained weights
- `lstm_scaler.pkl` — StandardScaler
- `lstm_feature_cols.pkl` — feature column names
- `lstm_config.pkl` — model configuration

---

## Model Architecture

```
LSTMClassifier(
  (lstm): LSTM(20, 64, num_layers=2, batch_first=True, dropout=0.3)
  (classifier): Sequential(
    Linear(64, 32)
    ReLU()
    Dropout(0.3)
    Linear(32, 2)
  )
)
```

- **Input**: 10-day sequences of 20 features
- **Output**: 2-class softmax (DOWN=0, UP=1)
- **Hidden size**: 64
- **Dropout**: 0.3
- **Device**: MPS (Apple Silicon) or CPU

---

## UI Theme

The dashboard uses a dark, minimal design:

| Element | Style |
|---------|-------|
| **Background** | `#111111` |
| **Cards** | `#161616` with `#1f1f1f` borders |
| **UP color** | `#4ade80` (green) |
| **DOWN color** | `#f87171` (red) |
| **Neutral** | `#71717a` (gray) |
| **Fonts** | Syne (headings), IBM Plex Mono (data) |

---

## Notes

- **No lookahead bias**: Global market features use lag-1 returns (previous day close) since European/Asian markets close before US open.
- **Sentiment fallback**: If no news for the latest trading day, falls back to previous day's sentiment.
- **Cross-validation accuracy**: ~52% — typical for financial time series. The model provides probabilistic guidance, not guaranteed predictions.
- **Backtest limitation**: Maximum 1-year date range to prevent excessive API calls.
- **Weekend handling**: Sentiment fetcher automatically skips weekend days.

---

## Disclaimer

This tool is for educational and research purposes only. Stock market predictions are inherently uncertain. Do not use this for financial decisions without proper due diligence. Past performance does not guarantee future results.
