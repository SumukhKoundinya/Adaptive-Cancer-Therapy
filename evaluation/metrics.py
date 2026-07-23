# evaluation/metrics.py
import numpy as np

def c_index(predictions, outcomes):
    """
    Simplified concordance index (survival ranking quality)
    """
    n = 0
    concordant = 0

    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            if outcomes[i] != outcomes[j]:
                n += 1
                concordant += int((predictions[i] < predictions[j]) == (outcomes[i] < outcomes[j]))

    return concordant / n if n > 0 else 0


def mse(pred, true):
    return np.mean((np.array(pred) - np.array(true)) ** 2)


def response_accuracy(pred_labels, true_labels):
    return np.mean(np.array(pred_labels) == np.array(true_labels))


def survival_improvement(baseline, model):
    return np.mean(model) - np.mean(baseline)