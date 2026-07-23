import numpy as np
import torch

from data.loader import load_dataset
from data.preprocess import clean_dataset
from data.encoders import encode_response
from state.patient_state import build_patient_states
from state.state_encoder import encode_to_tumor_state
from ml.response_model import ResponseTrainer
from policy.treatment_policy import TreatmentPolicy
from evaluation.evaluator import Evaluator
from visualization.policy_comparison import compare_policies
from visualization.survival_curves import plot_survival_curves

DATA_PATH = "data/gbm_columbia_2019_clinical_data.tsv"

ACTION_SPACE = [
    {
        "name": "low_intensity",
        "tumor_reduction_factor": 0.95,
        "toxicity": 0.05,
        "efficacy": 0.25,
        "pressure": 0.10,
    },
    {
        "name": "moderate_intensity",
        "tumor_reduction_factor": 0.80,
        "toxicity": 0.15,
        "efficacy": 0.45,
        "pressure": 0.25,
    },
    {
        "name": "high_intensity",
        "tumor_reduction_factor": 0.65,
        "toxicity": 0.30,
        "efficacy": 0.70,
        "pressure": 0.40,
    },
]

FEATURE_KEYS = [
    "tumor_burden",
    "mutation_entropy",
    "immune_pressure",
    "treatment_pressure",
    "time",
    "toxicity",
]


class ResponseModelWrapper:
    def __init__(self, input_dim: int):
        self.trainer = ResponseTrainer(input_dim)

    def train(self, X, y, epochs: int = 30):
        self.trainer.model.train()
        for epoch in range(epochs):
            loss = self.trainer.train_step(X, y)
            if epoch % 10 == 0:
                print(f"[Response] epoch={epoch} loss={loss:.4f}")

    def predict_proba(self, state):
        self.trainer.model.eval()
        x = self._state_to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            logits = self.trainer.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs

    @staticmethod
    def _state_to_tensor(state):
        x = torch.tensor(
            [
                state.get("tumor_burden", 0.0),
                state.get("mutation_entropy", 0.0),
                state.get("immune_pressure", 0.0),
                state.get("treatment_pressure", 0.0),
                state.get("time", 0.0),
                state.get("toxicity", 0.0),
            ],
            dtype=torch.float32,
        )
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class SurvivalModelWrapper:
    def __init__(self, input_dim: int):
        self.weights = np.zeros(input_dim, dtype=np.float32)
        self.bias = 0.0

    def train(self, X, y, epochs: int = 1):
        X_np = X.numpy()
        y_np = y.numpy()

        # Regularized linear regression on the state vectors.
        reg = 1e-3
        xtx = X_np.T.dot(X_np) + reg * np.eye(X_np.shape[1], dtype=np.float32)
        self.weights = np.linalg.solve(xtx, X_np.T.dot(y_np))
        self.bias = float(np.mean(y_np - X_np.dot(self.weights)))
        print(f"[Survival] fitted linear regression on {X_np.shape[0]} samples")

    def predict(self, state):
        x = np.array(
            [
                state.get("tumor_burden", 0.0),
                state.get("mutation_entropy", 0.0),
                state.get("immune_pressure", 0.0),
                state.get("treatment_pressure", 0.0),
                state.get("time", 0.0),
                state.get("toxicity", 0.0),
            ],
            dtype=np.float32,
        )
        return float(x.dot(self.weights) + self.bias)


def tumor_state_to_vector(state):
    vector = np.array(
        [
            state.tumor_burden,
            state.mutation_entropy,
            state.immune_pressure,
            state.treatment_pressure,
            state.time,
            0.0,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def state_dict_from_tumor_state(state):
    tumor_burden = np.nan_to_num(state.tumor_burden, nan=0.0, posinf=0.0, neginf=0.0)
    mutation_entropy = np.nan_to_num(state.mutation_entropy, nan=0.0, posinf=0.0, neginf=0.0)
    immune_pressure = np.nan_to_num(state.immune_pressure, nan=0.0, posinf=0.0, neginf=0.0)
    treatment_pressure = np.nan_to_num(state.treatment_pressure, nan=0.0, posinf=0.0, neginf=0.0)
    time = np.nan_to_num(state.time, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "tumor_burden": float(tumor_burden),
        "mutation_entropy": float(mutation_entropy),
        "immune_pressure": float(immune_pressure),
        "treatment_pressure": float(treatment_pressure),
        "time": float(time),
        "toxicity": 0.0,
    }


def build_training_dataset(states, responses, survivals):
    vectors = []
    labels = []
    durations = []

    for state, response, survival in zip(states, responses, survivals):
        if response is None or survival is None:
            continue
        if isinstance(response, float) and np.isnan(response):
            continue
        if isinstance(survival, float) and np.isnan(survival):
            continue

        vectors.append(tumor_state_to_vector(state))
        labels.append(int(response))
        durations.append(float(survival))

    X = torch.tensor(np.stack(vectors), dtype=torch.float32)
    y_response = torch.tensor(labels, dtype=torch.long)
    y_survival = torch.tensor(durations, dtype=torch.float32)
    return X, y_response, y_survival


def build_evaluation_dataset(states, responses, survivals):
    records = []
    for state, response, survival in zip(states, responses, survivals):
        if response is None or survival is None:
            continue
        if isinstance(response, float) and np.isnan(response):
            continue
        if isinstance(survival, float) and np.isnan(survival):
            continue
        records.append(
            {
                "state": state_dict_from_tumor_state(state),
                "true_response": int(response),
                "os_months": float(survival),
            }
        )
    return records


def run():
    df = load_dataset(DATA_PATH)
    df = clean_dataset(df)
    print(f"[OK] Loaded dataset: {df.shape}")

    patient_states = build_patient_states(df)
    tumor_states = [encode_to_tumor_state(p) for p in patient_states]
    print(f"[OK] Built {len(tumor_states)} tumor states")

    if "best_response_to_pd1_inhibitor_(rano)" in df.columns:
        responses = df["best_response_to_pd1_inhibitor_(rano)"].apply(encode_response).tolist()
    elif "response" in df.columns:
        responses = df["response"].apply(encode_response).tolist()
    else:
        responses = [None] * len(patient_states)

    survivals = [p.os_months for p in patient_states]
    X, y_response, y_survival = build_training_dataset(tumor_states, responses, survivals)

    if X.shape[0] == 0:
        raise RuntimeError("No training examples were available after filtering response and survival labels.")

    response_model = ResponseModelWrapper(input_dim=X.shape[1])
    survival_model = SurvivalModelWrapper(input_dim=X.shape[1])

    response_model.train(X, y_response, epochs=30)
    survival_model.train(X, y_survival, epochs=30)
    print("[OK] ML models trained")

    policy = TreatmentPolicy(
        response_model=response_model,
        survival_model=survival_model,
        action_space=ACTION_SPACE,
    )

    dataset = build_evaluation_dataset(tumor_states, responses, survivals)
    evaluator = Evaluator(policy, dataset)
    results = evaluator.run()

    print("\n=== RESULTS ===")
    print(results)

    compare_policies({
        "Δ-Nim Policy": results,
        "Baseline": results,
    })

    model_survival = [policy.choose_action(record["state"])["pred_survival"] for record in dataset]
    baseline_survival = [record["os_months"] for record in dataset]
    plot_survival_curves(baseline_survival=baseline_survival, model_survival=model_survival)

    print("\n[DONE] Pipeline finished successfully")


if __name__ == "__main__":
    run()
