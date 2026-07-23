# visualization/survival_curves.py

import matplotlib.pyplot as plt
import numpy as np

def plot_survival_curves(baseline_survival, model_survival, labels=("Baseline", "Δ-Nim Policy")):
    plt.figure()

    def survival_function(data):
        data = np.sort(data)
        return 1.0 - np.arange(len(data)) / len(data)

    plt.plot(survival_function(baseline_survival), label=labels[0])
    plt.plot(survival_function(model_survival), label=labels[1])

    plt.title("Survival Curves Comparison")
    plt.xlabel("Patients (sorted)")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.grid(True)
    plt.show()