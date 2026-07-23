# evaluation/baselines.py
import random
import numpy as np

def random_policy(state):
    return random.choice(["chemo", "radiation", "immunotherapy", "none"])


def ml_only_policy(response_model, state):
    # ignores Δ-Nim, purely predictive ML
    probs = response_model.predict_proba(state)
    return np.argmax(probs)


def standard_of_care_policy():
    # fixed heuristic baseline (VERY important for biomedical judging)
    return "temozolomide"