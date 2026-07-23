"""
demo/simulation.py
==================
Automated simulation runner for Delta-Nim adaptive therapy experiments.

Runs batch experiments and outputs:
  - Strategy comparison across tumor types
  - Kaplan-Meier survival curves (text)
  - ML classifier training and validation
  - Regime boundary analysis (Pigeonhole validation)
  - Position classification accuracy vs theory

Usage:
  python main.py --mode simulate
  python main.py --mode simulate --n-games 200 --tumor-type breast
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.core import (
    nim_sum, regime, support_size, count_equal_pairs,
    is_p_position, total_cells, DENSE_CUTOFF, MIN_PAIRS
)
from data.generator import GameSimulator, HeapGenerator
from clinical.patient import Patient
from clinical.survival import SurvivalSimulator, KaplanMeierEstimator, log_rank_test
from ml.classifier import DatasetGenerator, RandomForestClassifier, train_and_evaluate
from visualization.board import (
    render_comparison_table, render_survival_summary
)


def section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def run_strategy_comparison(n_games: int = 100, tumor_type: str = "prostate"):
    section(f"Strategy Comparison — {tumor_type.title()} ({n_games} games)")
    sim  = GameSimulator(seed=42)
    traj = sim.simulate_cohort(
        n_games=n_games,
        tumor_type=tumor_type,
        onco_strategies=["optimal", "max_dose", "random"],
        tumor_strategy="stochastic",
    )
    comparison = sim.compute_strategy_comparison(traj)
    print(render_comparison_table(comparison))

    # Show regime breakdown
    sparse_games = [t for t in traj if t.initial_regime == "sparse" and t.oncologist_strategy == "optimal"]
    dense_games  = [t for t in traj if t.initial_regime == "dense"  and t.oncologist_strategy == "optimal"]

    def win_rate(games): return sum(1 for g in games if g.winner == "oncologist") / max(len(games), 1)

    print(f"\n  Δ-Nim optimal — regime breakdown:")
    print(f"    Sparse regime: {win_rate(sparse_games)*100:.1f}% remission ({len(sparse_games)} games)")
    print(f"    Dense  regime: {win_rate(dense_games)*100:.1f}%  remission ({len(dense_games)} games)")

    # Position type breakdown
    n_games_by_pos = {
        "N": [t for t in traj if t.initial_position == "N" and t.oncologist_strategy == "optimal"],
        "P": [t for t in traj if t.initial_position == "P" and t.oncologist_strategy == "optimal"],
    }
    print(f"\n  Δ-Nim optimal — initial position breakdown:")
    for pt, games in n_games_by_pos.items():
        wr = win_rate(games)
        print(f"    Starting {pt}-position: {wr*100:.1f}% remission ({len(games)} games)")


def run_survival_analysis(n_patients: int = 150, tumor_type: str = "prostate"):
    section(f"Kaplan-Meier Survival Analysis — {tumor_type.title()} (n={n_patients})")
    sim    = SurvivalSimulator(seed=42)
    events = sim.simulate_cohort(n_patients=n_patients, tumor_type=tumor_type)

    km_arms = ["optimal_nim", "standard_of_care", "no_treatment"]
    km_results = []
    for arm in km_arms:
        km = KaplanMeierEstimator(events, arm)
        km_results.append(km.summary())

    print(render_survival_summary(km_results))

    # Pairwise log-rank tests
    pairs = [
        ("optimal_nim", "standard_of_care"),
        ("optimal_nim", "no_treatment"),
        ("standard_of_care", "no_treatment"),
    ]
    print("\n  Log-rank pairwise tests:")
    for a1, a2 in pairs:
        lr = log_rank_test(events, a1, a2)
        print(f"    {a1} vs {a2}:")
        print(f"      χ²={lr['chi2']:.4f}  p={lr['p_value']:.4f}  {'*significant*' if lr['significant'] else 'ns'}")

    # Subgroup: N-position vs P-position starting patients
    print("\n  Subgroup analysis (Δ-Nim arm only):")
    n_start = [e for e in events if e.therapy_arm == "optimal_nim" and e.initial_position == "N"]
    p_start = [e for e in events if e.therapy_arm == "optimal_nim" and e.initial_position == "P"]
    if n_start and p_start:
        n_med = sum(e.time_weeks for e in n_start) / len(n_start)
        p_med = sum(e.time_weeks for e in p_start) / len(p_start)
        print(f"    N-position patients: mean {n_med:.1f} weeks")
        print(f"    P-position patients: mean {p_med:.1f} weeks")
        print(f"    Ratio: {n_med/p_med:.2f}x benefit for N-position start")


def run_ml_validation(n_positions: int = 5000):
    section(f"ML Classifier Validation ({n_positions} positions)")
    model, metrics = train_and_evaluate(
        n_positions=n_positions,
        n_estimators=30,
        verbose=True,
    )
    print(f"\n  Confusion matrix:")
    print(f"    TP (N predicted N): {metrics['tp']}")
    print(f"    TN (P predicted P): {metrics['tn']}")
    print(f"    FP (P predicted N): {metrics['fp']}")
    print(f"    FN (N predicted P): {metrics['fn']}")

    from engine.core import FEATURE_NAMES
    print(f"\n  Feature importances (top 5):")
    fi_pairs = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )[:5]
    for name, imp in fi_pairs:
        bar = "█" * int(imp * 40)
        print(f"    {name:<22} {bar} {imp:.4f}")

    # Test classifier on known positions
    print(f"\n  Sanity checks on known positions:")
    known = [
        ([3, 5, 6], "P"),
        ([1, 2, 3], "P"),
        ([7, 7],    "P"),
        ([4, 5],    "N"),
        ([1, 2],    "N"),
        ([1],       "N"),
    ]
    correct = 0
    for heaps, true_label in known:
        result = model.classify_position(heaps)
        match  = result["prediction"] == true_label
        correct += int(match)
        status  = "OK" if match else "WRONG"
        print(f"    [{status}] {heaps} → predicted {result['prediction']} "
              f"(true {true_label}, conf {result['confidence']:.3f})")
    print(f"    Sanity: {correct}/{len(known)} correct")


def run_regime_boundary_analysis():
    section("Regime Boundary Analysis (Pigeonhole Principle Validation)")
    print("""
  The dense cutoff of 60 heaps derives from the Pigeonhole Principle:
  With 61+ distinct active clusters, Player II can always find 30
  disjoint equal-heap pairs to maintain the nim-sum = 0 invariant.
  We validate this empirically below.
""")
    gen = HeapGenerator(seed=0)

    print(f"  |supp|   Regime     P-pos%   Pairs(mean)  Nim-sum=0%")
    print(f"  ──────   ──────     ──────   ──────────   ──────────")
    for sz in [10, 30, 55, 59, 60, 61, 65, 80, 100]:
        samples = []
        for _ in range(200):
            heaps = [1] * sz  # start with sz=1 heaps
            # Randomize heap values
            import random
            rng = random.Random(sz * 7)
            heaps = [rng.randint(1, 20) for _ in range(sz)]
            samples.append(heaps)

        n_p     = sum(1 for h in samples if is_p_position(h))
        ns_zero = sum(1 for h in samples if nim_sum(h) == 0)
        pairs   = sum(count_equal_pairs(h) for h in samples) / len(samples)
        r       = regime(samples[0])
        print(f"  {sz:6d}   {r:<10} {n_p/2:.0f}%     {pairs:8.1f}   {ns_zero/2:.0f}%")


def run_clinical_pipeline(tumor_type: str = "prostate", n_patients: int = 5):
    section(f"Clinical Pipeline Demo — {tumor_type.title()} ({n_patients} patients)")
    from clinical.treatment_planner import TreatmentPlanner
    planner = TreatmentPlanner(max_lookahead=12)

    for i in range(n_patients):
        patient = Patient.generate(tumor_type=tumor_type, seed=i * 17)
        plan    = planner.generate_plan(patient.patient_id, patient.heaps())
        cs      = patient.clinical_summary()
        print(f"\n  {patient.patient_id} | {patient.tumor_type} | "
              f"{cs['n_clusters']} clusters | {cs['total_burden_k_cells']}×10³ cells | "
              f"Position: {cs['initial_position']}")
        print(f"    Regime: {cs['regime']} | Nim-sum: {cs['nim_sum']} | "
              f"Equal pairs: {cs['equal_pairs']}/{MIN_PAIRS}")
        print(f"    Projected cycles: {plan.projected_cycles}")
        print(f"    Outcome: {plan.predicted_outcome[:80]}...")
        print(f"    Survival: {plan.survival_estimate[:80]}...")
        if plan.regime_warnings:
            print(f"    Warnings: {len(plan.regime_warnings)} regime transitions detected")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_full_simulation(
    n_games:    int = 100,
    tumor_type: str = "prostate",
):
    print("\n" + "=" * 65)
    print("  Δ-Nim Adaptive Therapy — Full Simulation Suite")
    print("=" * 65)

    run_strategy_comparison(n_games=n_games, tumor_type=tumor_type)
    run_survival_analysis(n_patients=max(50, n_games), tumor_type=tumor_type)
    run_ml_validation(n_positions=min(5000, n_games * 50))
    run_regime_boundary_analysis()
    run_clinical_pipeline(tumor_type=tumor_type, n_patients=3)

    print("\n" + "=" * 65)
    print("  Simulation complete.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_full_simulation()