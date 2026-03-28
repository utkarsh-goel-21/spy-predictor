import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

NON_FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume", "target"]


def load_features() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "spy_features.csv"
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    return [col for col in df.columns if col not in NON_FEATURE_COLS]


def train_model(df: pd.DataFrame):
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["target"].values

    tscv = TimeSeriesSplit(n_splits=5)

    print(f"Training with {len(feature_cols)} features on {len(df)} samples")
    print(f"Features: {feature_cols}\n")

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        y_train, y_val = y[train_idx], y[val_idx]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = accuracy_score(y_val, preds)
        fold_scores.append(score)
        print(f"Fold {fold + 1} accuracy: {score:.4f}")

    print(f"\nMean CV accuracy: {np.mean(fold_scores):.4f}")
    print(f"Std: {np.std(fold_scores):.4f}")

    # Final model on all data
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)

    final_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_scaled, y)

    final_preds = final_model.predict(X_scaled)
    print(f"\nFull training set report:")
    print(classification_report(y, final_preds))

    return final_model, final_scaler, feature_cols


def save_artifacts(model, scaler, feature_cols):
    joblib.dump(model, MODELS_DIR / "rf_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")
    print("Saved model, scaler, and feature list.")


if __name__ == "__main__":
    df = load_features()
    model, scaler, feature_cols = train_model(df)
    save_artifacts(model, scaler, feature_cols)