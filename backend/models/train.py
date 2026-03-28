import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import joblib

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

NON_FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume", "target"]
SEQUENCE_LENGTH = 10
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class SPYDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]
        y_label = self.y[idx + self.seq_len]
        return x_seq, y_label


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.classifier(hidden[-1])
        return out


def load_features() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "spy_features.csv"
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    return [col for col in df.columns if col not in NON_FEATURE_COLS]


def get_class_weights(y: np.ndarray) -> torch.Tensor:
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    weights = torch.tensor([n_pos / n_neg, n_neg / n_pos], dtype=torch.float32)
    return weights.to(DEVICE)


def train_with_early_stopping(model, train_loader, val_loader, optimizer, criterion, patience=10, max_epochs=100):
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(X_batch), y_batch).backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                val_loss += criterion(model(X_batch), y_batch).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            preds = torch.argmax(model(X_batch), dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    return np.array(all_preds), np.array(all_labels)


def train_model(df: pd.DataFrame):
    feature_cols = get_feature_columns(df)
    X_raw = df[feature_cols].values
    y = df["target"].values

    print(f"Device: {DEVICE}")
    print(f"Training LSTM with {len(feature_cols)} features, sequence length={SEQUENCE_LENGTH}")
    print(f"Total samples: {len(df)}\n")

    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_raw)):
        scaler = StandardScaler()
        X_train_raw = scaler.fit_transform(X_raw[train_idx])
        X_val_raw = scaler.transform(X_raw[val_idx])

        y_train = y[train_idx]
        y_val = y[val_idx]

        train_ds = SPYDataset(X_train_raw, y_train, SEQUENCE_LENGTH)
        val_ds = SPYDataset(X_val_raw, y_val, SEQUENCE_LENGTH)

        if len(train_ds) == 0 or len(val_ds) == 0:
            print(f"Fold {fold + 1}: not enough data, skipping")
            continue

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        model = LSTMClassifier(input_size=len(feature_cols)).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=get_class_weights(y_train))

        model = train_with_early_stopping(
            model, train_loader, val_loader, optimizer, criterion,
            patience=10, max_epochs=100
        )

        preds, labels = evaluate(model, val_loader)
        score = accuracy_score(labels, preds)
        fold_scores.append(score)
        print(f"Fold {fold + 1} accuracy: {score:.4f}")

    print(f"\nMean CV accuracy: {np.mean(fold_scores):.4f}")
    print(f"Std: {np.std(fold_scores):.4f}")

    # Final model — train on 80%, use next 10% for early stopping only
    final_scaler = StandardScaler()
    n = len(X_raw)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)

    X_train_final = final_scaler.fit_transform(X_raw[:train_end])
    X_val_final = final_scaler.transform(X_raw[train_end:val_end])
    y_train_final = y[:train_end]
    y_val_final = y[train_end:val_end]

    final_ds = SPYDataset(X_train_final, y_train_final, SEQUENCE_LENGTH)
    final_val_ds = SPYDataset(X_val_final, y_val_final, SEQUENCE_LENGTH)
    final_loader = DataLoader(final_ds, batch_size=32, shuffle=False)
    final_val_loader = DataLoader(final_val_ds, batch_size=32, shuffle=False)

    final_model = LSTMClassifier(input_size=len(feature_cols)).to(DEVICE)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=get_class_weights(y_train_final))

    final_model = train_with_early_stopping(
        final_model, final_loader, final_val_loader, optimizer, criterion,
        patience=10, max_epochs=100
    )

    preds, labels = evaluate(final_model, final_val_loader)
    print(f"\nFinal model validation report:")
    print(classification_report(labels, preds))

    return final_model, final_scaler, feature_cols


def save_artifacts(model, scaler, feature_cols):
    torch.save(model.state_dict(), MODELS_DIR / "lstm_model.pt")
    joblib.dump(scaler, MODELS_DIR / "lstm_scaler.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "lstm_feature_cols.pkl")
    joblib.dump({"input_size": len(feature_cols)}, MODELS_DIR / "lstm_config.pkl")
    print("Saved LSTM model, scaler, and config.")


if __name__ == "__main__":
    df = load_features()
    model, scaler, feature_cols = train_model(df)
    save_artifacts(model, scaler, feature_cols)