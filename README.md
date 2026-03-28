# SPY Direction Predictor

Predicts next-trading-day SPY (S&P 500 ETF) up/down movement using historical market data, technical indicators, and market context.

## Structure
```
spy-predictor/
├── backend/         # FastAPI app + ML pipeline
│   ├── data/        # Data fetching and storage
│   ├── features/    # Feature engineering
│   ├── models/      # Model training and inference
│   └── api/         # API routes
├── frontend/        # Dashboard UI
├── notebooks/       # Exploration and experimentation
└── data/            # Local data (gitignored)
```

## Stack

- **Backend**: Python, FastAPI
- **ML**: scikit-learn, pandas, pandas-ta
- **Frontend**: React + TypeScript (Vite)
- **Data**: yfinance
