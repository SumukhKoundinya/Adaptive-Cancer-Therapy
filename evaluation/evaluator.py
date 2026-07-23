# evaluation/evaluator.py

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from evaluation.metrics import c_index
from evaluation.baselines import random_policy, standard_of_care_policy

class Evaluator:
    def __init__(self, policy, dataset):
        self.policy = policy
        self.dataset = dataset

    def run(self):
        preds = []
        true = []
        survival_preds = []
        survival_true = []

        for patient in self.dataset:
            state = patient["state"]

            action = self.policy.choose_action(state)

            preds.append(action["response_prob"])
            true.append(patient["true_response"])

            survival_preds.append(action["pred_survival"])
            survival_true.append(patient["os_months"])

        response_preds = [1 if p >= 0.5 else 0 for p in preds]

        return {
            "response_accuracy": accuracy_score(true, response_preds),
            "response_f1": f1_score(true, response_preds),
            "response_auc": roc_auc_score(true, preds),
            "c_index": c_index(survival_preds, survival_true)
        }