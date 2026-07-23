# visualization/tumor_dynamics.py

import matplotlib.pyplot as plt

def plot_tumor_evolution(states):
    tumor_sizes = [s["tumor_burden"] for s in states]

    plt.figure()
    plt.plot(tumor_sizes, marker="o")
    plt.title("Δ-Nim Tumor Evolution Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Tumor Burden")
    plt.grid(True)
    plt.show()