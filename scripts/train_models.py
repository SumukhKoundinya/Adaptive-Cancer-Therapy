"""
scripts/train_models.py
========================
Train ResponseModel and SurvivalModel on clinical dataset and save checkpoints.

Usage:
    python scripts/train_models.py

Exports:
    train_response_model(df, epochs=10)
    train_survival_model(df, epochs=10)

"""
from pathlib import Path
import torch
import numpy as np
from data.loader import load_dataset
from data.preprocess import clean_dataset
from ml.feature_model import FeatureModel
from ml.response_model import ResponseTrainer
from ml.survival_model import SurvivalTrainer

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def train_response_model(df, epochs: int = 10, batch_size: int = 32):
    """Train ResponseModel if `response_encoded` exists in df."""
    if "response_encoded" not in df.columns:
        print("No response labels in dataframe; skipping ResponseModel training.")
        return None

    # Drop rows with missing labels
    df = df.dropna(subset=["response_encoded"]).copy()
    if df.empty:
        print("No non-missing response labels available; skipping ResponseModel training.")
        return None

    feat = FeatureModel()
    X = feat.fit_transform(df)
    y = df["response_encoded"].astype(int).values

    X_np = X.values.astype(np.float32)
    n_features = X_np.shape[1]

    trainer = ResponseTrainer(input_dim=n_features, lr=1e-3)
    trainer.model.train()

    n = len(X_np)
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        losses = []
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            X_batch = torch.from_numpy(X_np[idx])
            y_batch = torch.from_numpy(y[idx]).long()
            loss = trainer.train_step(X_batch, y_batch)
            losses.append(loss)
        print(f"ResponseModel epoch {epoch+1}/{epochs} - loss {np.mean(losses):.4f}")

    ckpt = CHECKPOINT_DIR / "response_model.pth"
    torch.save(trainer.model.state_dict(), ckpt)
    print(f"Saved ResponseModel checkpoint: {ckpt}")
    return str(ckpt)


def train_survival_model(df, epochs: int = 10, batch_size: int = 32):
    """Train SurvivalModel if `os_months` exists in df."""
    if "os_months" not in df.columns:
        print("No survival labels in dataframe; skipping SurvivalModel training.")
        return None
    # Drop rows with missing survival labels
    df = df.dropna(subset=["os_months"]).copy()
    if df.empty:
        print("No non-missing survival labels available; skipping SurvivalModel training.")
        return None
    feat = FeatureModel()
    X = feat.fit_transform(df)
    y = df["os_months"].astype(float).values

    X_np = X.values.astype(np.float32)
    n_features = X_np.shape[1]

    trainer = SurvivalTrainer(input_dim=n_features, lr=1e-3)
    trainer.model.train()

    n = len(X_np)
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        losses = []
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            X_batch = torch.from_numpy(X_np[idx])
            y_batch = torch.from_numpy(y[idx]).float()
            loss = trainer.train_step(X_batch, y_batch)
            losses.append(loss)
        print(f"SurvivalModel epoch {epoch+1}/{epochs} - loss {np.mean(losses):.4f}")

    ckpt = CHECKPOINT_DIR / "survival_model.pth"
    torch.save(trainer.model.state_dict(), ckpt)
    print(f"Saved SurvivalModel checkpoint: {ckpt}")
    return str(ckpt)


if __name__ == "__main__":
    data_path = "data/gbm_columbia_2019_clinical_data.tsv"
    if not Path(data_path).exists():
        print(f"Dataset not found at {data_path}. Cannot train models.")
    else:
        df = load_dataset(data_path)
        df = clean_dataset(df)
        train_response_model(df, epochs=5)
        train_survival_model(df, epochs=5)
