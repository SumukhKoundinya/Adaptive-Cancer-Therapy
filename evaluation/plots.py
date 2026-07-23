# evaluation/plots.py

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_response_roc_curve(true_labels, prob_scores, save_path="evaluation/roc_curve.png"):
    """Plot the ROC curve for response prediction and save it to a file."""
    fpr, tpr, _ = roc_curve(true_labels, prob_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Response ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return save_path
