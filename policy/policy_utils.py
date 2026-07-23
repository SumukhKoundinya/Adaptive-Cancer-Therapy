# policy/policy_utils.py
import numpy as np

def normalize(x):
    x = np.array(x, dtype=float)
    if np.std(x) == 0:
        return x
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def survival_reward(pred_survival_months):
    return np.log1p(pred_survival_months)


def toxicity_penalty(toxicity_score):
    return toxicity_score ** 2


def progression_penalty(pfs_risk):
    return pfs_risk


def total_reward(survival, toxicity, progression):
    return (
        survival_reward(survival)
        - toxicity_penalty(toxicity)
        - progression_penalty(progression)
    )
