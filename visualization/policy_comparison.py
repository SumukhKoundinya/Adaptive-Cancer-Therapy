# visualization/policy_comparison.py

import matplotlib.pyplot as plt

def compare_policies(results):
    policies = list(results.keys())
    response_acc = [results[p]["response_accuracy"] for p in policies]
    c_index = [results[p]["c_index"] for p in policies]

    plt.figure()
    plt.bar(policies, response_acc)
    plt.title("Response Accuracy by Policy")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=30)
    plt.show()

    plt.figure()
    plt.bar(policies, c_index)
    plt.title("Survival Ranking Quality (C-index)")
    plt.ylabel("C-index")
    plt.xticks(rotation=30)
    plt.show()