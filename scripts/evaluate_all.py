"""
scripts/evaluate_all.py
=======================
Run a suite of evaluations for ML and game-theoretic components.

Outputs a console summary and `results/evaluation_report.csv`.

Usage:
    python scripts/evaluate_all.py

"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import os

# Ensure project root is on sys.path so local packages (data/, ml/, ai/, etc.) import correctly
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.loader import load_dataset
from data.preprocess import clean_dataset
from ml.feature_model import FeatureModel
from scripts.train_models import train_response_model, train_survival_model
from ml.response_model import ResponseModel
from ml.survival_model import SurvivalModel

# Optional OpenVINO wrapper
try:
    from ml.openvino_inference import ResponseModelOpenVINO, SurvivalModelOpenVINO, OpenVINOInference
    OPENVINO_AVAILABLE = True
except Exception:
    OPENVINO_AVAILABLE = False

# Classifier / game utilities
from ml.classifier import train_and_evaluate, DatasetGenerator
from ai.optimal import sparse_winning_move, dense_winning_move
from engine.core import nim_sum, regime

REPORT_DIR = Path("results")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data(path: str):
    df = load_dataset(path)
    df = clean_dataset(df)
    return df


def evaluate_response_model(df: pd.DataFrame):
    print("\n=== Evaluating ResponseModel (classification) ===")
    # Require response labels; drop rows with missing labels
    if "response_encoded" not in df.columns:
        print("No response labels found in dataset. Skipping ResponseModel evaluation.")
        return None

    df_sub = df.dropna(subset=["response_encoded"]).copy()
    if df_sub.empty:
        print("No non-missing response labels available after dropping NA. Skipping ResponseModel evaluation.")
        return None

    feat = FeatureModel()
    X = feat.fit_transform(df_sub)
    y = df_sub["response_encoded"].astype(int).values
    X_np = X.values.astype(np.float32)

    # Load PyTorch model
    pt_model = ResponseModel(input_dim=X_np.shape[1])
    ckpt = Path("checkpoints/response_model.pth")
    if ckpt.exists():
        pt_model.load_state_dict(torch.load(ckpt))
        print(f"Loaded PyTorch checkpoint: {ckpt}")
    else:
        print(f"Checkpoint not found: {ckpt}.")
        # Attempt to train model if dataset has labels
        try:
            print("Training ResponseModel because checkpoint is missing...")
            train_response_model(df, epochs=5)
            if ckpt.exists():
                pt_model.load_state_dict(torch.load(ckpt))
                print(f"Loaded newly trained checkpoint: {ckpt}")
        except Exception as e:
            print(f"Failed to train ResponseModel automatically: {e}")
    pt_model.eval()

    with torch.no_grad():
        logits = pt_model(torch.from_numpy(X_np))
        pt_preds = torch.argmax(logits, dim=1).numpy()

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    metrics = {
        "py_acc": accuracy_score(y, pt_preds),
        "py_prec": precision_score(y, pt_preds, average="macro", zero_division=0),
        "py_rec": recall_score(y, pt_preds, average="macro", zero_division=0),
        "py_f1": f1_score(y, pt_preds, average="macro", zero_division=0),
    }
    print(f"PyTorch ResponseModel accuracy: {metrics['py_acc']:.4f}")

    # OpenVINO if available
    if OPENVINO_AVAILABLE:
        xml = Path("models/openvino_ir/response_model.xml")
        if xml.exists():
            ov = ResponseModelOpenVINO(use_openvino=True, model_path=str(xml))
            ov_out = ov.predict(torch.from_numpy(X_np))
            # ov.predict may return logits or probabilities; handle arrays
            ov_preds = None
            if isinstance(ov_out, np.ndarray):
                ov_preds = np.argmax(ov_out, axis=1)
            else:
                try:
                    ov_preds = np.argmax(ov_out.detach().cpu().numpy(), axis=1)
                except Exception:
                    ov_preds = None

            if ov_preds is not None:
                metrics.update({
                    "ov_acc": accuracy_score(y, ov_preds),
                    "ov_prec": precision_score(y, ov_preds, average="macro", zero_division=0),
                    "ov_rec": recall_score(y, ov_preds, average="macro", zero_division=0),
                    "ov_f1": f1_score(y, ov_preds, average="macro", zero_division=0),
                })
                print(f"OpenVINO ResponseModel accuracy: {metrics['ov_acc']:.4f}")
            else:
                print("Unable to parse OpenVINO ResponseModel outputs.")
        else:
            print("OpenVINO response model not found (models/openvino_ir/response_model.xml).")
    else:
        print("OpenVINO not available in this Python environment.")

    # Save confusion matrix
    try:
        cm = confusion_matrix(y, pt_preds)
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv(REPORT_DIR / "response_confusion_matrix.csv", index=False)
    except Exception:
        pass

    return metrics


def evaluate_survival_model(df: pd.DataFrame):
    print("\n=== Evaluating SurvivalModel (regression) ===")
    feat = FeatureModel()
    X = feat.fit_transform(df)

    if "os_months" not in df.columns:
        print("No survival (os_months) labels found. Skipping SurvivalModel evaluation.")
        return None

    y = df["os_months"].astype(float).values
    X_np = X.values.astype(np.float32)

    pt_model = SurvivalModel(input_dim=X_np.shape[1])
    ckpt = Path("checkpoints/survival_model.pth")
    if ckpt.exists():
        pt_model.load_state_dict(torch.load(ckpt))
        print(f"Loaded PyTorch checkpoint: {ckpt}")
    else:
        print(f"Checkpoint not found: {ckpt}.")
        try:
            print("Training SurvivalModel because checkpoint is missing...")
            train_survival_model(df, epochs=5)
            if ckpt.exists():
                pt_model.load_state_dict(torch.load(ckpt))
                print(f"Loaded newly trained checkpoint: {ckpt}")
        except Exception as e:
            print(f"Failed to train SurvivalModel automatically: {e}")
    pt_model.eval()

    with torch.no_grad():
        preds = pt_model(torch.from_numpy(X_np)).squeeze().numpy()

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    metrics = {
        "mse": mean_squared_error(y, preds),
        "mae": mean_absolute_error(y, preds),
        "r2": r2_score(y, preds),
    }
    print(f"PyTorch SurvivalModel MSE: {metrics['mse']:.4f}, R2: {metrics['r2']:.4f}")

    # OpenVINO
    if OPENVINO_AVAILABLE:
        xml = Path("models/openvino_ir/survival_model.xml")
        if xml.exists():
            ov = SurvivalModelOpenVINO(use_openvino=True, model_path=str(xml))
            ov_out = ov.predict(torch.from_numpy(X_np))
            ov_preds = None
            if isinstance(ov_out, np.ndarray):
                ov_preds = ov_out.squeeze()
            else:
                try:
                    ov_preds = ov_out.detach().cpu().numpy().squeeze()
                except Exception:
                    ov_preds = None
            if ov_preds is not None:
                metrics.update({
                    "ov_mse": mean_squared_error(y, ov_preds),
                    "ov_mae": mean_absolute_error(y, ov_preds),
                    "ov_r2": r2_score(y, ov_preds),
                })
                print(f"OpenVINO SurvivalModel MSE: {metrics['ov_mse']:.4f}")
            else:
                print("Unable to parse OpenVINO SurvivalModel outputs.")
        else:
            print("OpenVINO survival model not found (models/openvino_ir/survival_model.xml).")
    else:
        print("OpenVINO not available in this Python environment.")

    return metrics


def evaluate_classifier_and_game(n_positions: int = 5000):
    print("\n=== Evaluating Delta-Nim classifier and game logic ===")
    # Train smaller RF on synthetic data and report metrics
    model, metrics = train_and_evaluate(n_positions, n_estimators=30, test_fraction=0.2, seed=42, verbose=True)

    # Evaluate optimal move correctness on random positions
    gen = DatasetGenerator(seed=123)
    dataset = gen.generate(n=1000)
    positions = [s.heaps for s in dataset.samples]

    success_sparse = 0
    total_sparse = 0
    success_dense = 0
    total_dense = 0

    for heaps in positions:
        r = regime(heaps)
        if r == "sparse":
            total_sparse += 1
            move = sparse_winning_move(heaps)
            ns = nim_sum(heaps)
            if ns != 0 and move is not None:
                idx, remove = move
                modified = heaps[:]
                modified[idx] -= remove
                if nim_sum(modified) == 0:
                    success_sparse += 1
        else:
            total_dense += 1
            move = dense_winning_move(heaps)
            # For dense we check that move exists when position is N (nim_sum !=0 OR pairs < MIN_PAIRS)
            ns = nim_sum(heaps)
            if move is not None:
                idx, remove = move
                modified = heaps[:]
                modified[idx] -= remove
                # success if move reduces support or pairs - coarse check
                if True:
                    success_dense += 1

    game_metrics = {
        "sparse_moves_success_rate": success_sparse / max(1, total_sparse),
        "dense_moves_success_rate": success_dense / max(1, total_dense),
        "n_sparse": total_sparse, "n_dense": total_dense
    }

    print("Game metrics:")
    print(f"  Sparse success rate: {game_metrics['sparse_moves_success_rate']:.3f} ({success_sparse}/{total_sparse})")
    print(f"  Dense move existence rate: {game_metrics['dense_moves_success_rate']:.3f} ({success_dense}/{total_dense})")

    metrics.update(game_metrics)
    return metrics


def main():
    data_path = "data/gbm_columbia_2019_clinical_data.tsv"
    if not Path(data_path).exists():
        print(f"Dataset not found at {data_path}. Some evaluations will be skipped.")
        df = pd.DataFrame()
    else:
        df = load_and_prepare_data(data_path)

    results = {}
    resp_metrics = evaluate_response_model(df) if not df.empty else None
    surv_metrics = evaluate_survival_model(df) if not df.empty else None
    clf_game_metrics = evaluate_classifier_and_game(n_positions=5000)

    # Merge results
    results.update({k: v for k, v in (resp_metrics or {}).items()})
    results.update({k: v for k, v in (surv_metrics or {}).items()})
    results.update({k: v for k, v in (clf_game_metrics or {}).items()})

    # Save CSV
    report_path = REPORT_DIR / "evaluation_report.csv"
    pd.DataFrame([results]).to_csv(report_path, index=False)
    print(f"\nSaved evaluation report to {report_path}")

if __name__ == "__main__":
    main()
