# evaluation/ablation_study.py

from evaluation.evaluator import Evaluator
from evaluation.baselines import random_policy, standard_of_care_policy

def run_ablation(full_policy, ml_only_policy, dataset):

    full_eval = Evaluator(full_policy, dataset).run()
    ml_eval = Evaluator(ml_only_policy, dataset).run()

    return {
        "Δ-Nim + ML Policy": full_eval,
        "ML Only": ml_eval,
        "Improvement": {
            "response_accuracy_gain":
                full_eval["response_accuracy"] - ml_eval["response_accuracy"],
            "c_index_gain":
                full_eval["c_index"] - ml_eval["c_index"]
        }
    }