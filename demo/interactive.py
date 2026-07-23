"""
demo/interactive.py
===================
Interactive CLI demonstration for ISEF judges.

Walk the judge through:
  1. Patient profile generation
  2. Delta-Nim position analysis
  3. Interactive treatment planning
  4. Comparison vs standard of care
  5. Survival analysis

Usage:
  python main.py --mode interactive
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.game import CancerNimGame
from engine.core import (
    nim_sum, regime, support_size, count_equal_pairs,
    is_p_position, total_cells, DENSE_CUTOFF, MIN_PAIRS
)
from ai.optimal import compute_optimal_move, ranked_moves
from clinical.patient import Patient
from clinical.treatment_planner import TreatmentPlanner
from visualization.board import (
    render_full_board, render_cluster_map, render_theory_panel,
    render_nim_decomposition, render_treatment_log
)


# ── Input helpers ─────────────────────────────────────────────────────────────

def prompt(msg: str) -> str:
    try:
        return input(msg).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def prompt_int(msg: str, lo: int, hi: int, default: int = None) -> int:
    while True:
        raw = prompt(msg)
        if not raw and default is not None:
            return default
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
            print(f"    Please enter a number between {lo} and {hi}.")
        except ValueError:
            print("    Please enter an integer.")


def press_enter(msg: str = "  [Press Enter to continue]"):
    prompt(msg)


# ── Demo sections ─────────────────────────────────────────────────────────────

def intro_screen():
    print("\n" + "=" * 65)
    print("  Δ-Nim Adaptive Cancer Therapy — Interactive Demo")
    print("  Sumukh Koundinya | Centerton, AR | ISEF Mathematics")
    print("=" * 65)
    print("""
  This demo applies Delta-Nim combinatorial game theory to
  adaptive cancer therapy. Tumor cell clusters are modeled as
  heaps in a Nim-style game between:

    Oncologist (Player I)  — applies targeted therapy
    Tumor (Player II)      — evolves drug resistance optimally

  The key insight: game-theoretic optimal treatment sequences
  outperform maximum-dose standard-of-care by exploiting the
  mathematical structure of the tumor's resistance evolution.

  Reference: Zhang et al. (2022) NCT02415621 — adaptive therapy
  extended median progression-free survival from 14.3 → 33 months
  in prostate cancer. Δ-Nim provides the theoretical foundation
  for WHY adaptive scheduling outperforms continuous dosing.
""")
    press_enter()


def demo_patient_generation() -> Patient:
    print("\n" + "─" * 65)
    print("  SECTION 1: Patient Profile")
    print("─" * 65)
    print("""
  Choose a tumor type to generate a synthetic patient:

    1. Prostate cancer  (4–8 clusters, sparse regime typical)
    2. Breast cancer    (6–14 clusters, mixed regime)
    3. Lung cancer      (8–16 clusters, high mutation rate)
    4. Metastatic       (20–80 clusters, dense regime, pairing dominant)
""")
    choice = prompt_int("  Select tumor type [1-4, default=1]: ", 1, 4, default=1)
    tumor_types = {1: "prostate", 2: "breast", 3: "lung", 4: "metastatic"}
    tumor_type  = tumor_types[choice]

    seed = prompt_int("  Random seed (for reproducibility) [0-999, default=42]: ", 0, 999, default=42)
    patient = Patient.generate(tumor_type=tumor_type, seed=seed)

    print(f"\n  Generated patient: {patient.patient_id}")
    print(f"  Age: {patient.age}  |  Stage: {patient.stage}")
    print(f"  ECOG performance score: {patient.ecog_score}")
    print(f"  Prior therapy lines: {patient.prior_lines}")
    print(f"\n  Tumor cluster configuration:")
    print(f"  Clusters (heap sizes): {patient.heaps()}")
    print(f"  Total burden: {patient.total_burden()}×10³ cells")
    print()
    print(patient.resistance_profile_summary())

    cs = patient.clinical_summary()
    print(f"\n  Game-theoretic prognosis:")
    print(f"    Nim-sum: {cs['nim_sum']}")
    print(f"    Regime: {cs['regime']}")
    print(f"    Initial position: {cs['initial_position']}-position")
    print(f"    → {cs['game_theoretic_prognosis']}")

    press_enter()
    return patient


def demo_position_analysis(patient: Patient):
    print("\n" + "─" * 65)
    print("  SECTION 2: Delta-Nim Position Analysis")
    print("─" * 65)
    heaps = patient.heaps()
    print(render_cluster_map(heaps))
    print(render_nim_decomposition(heaps))
    print(render_theory_panel(heaps))
    press_enter()


def demo_treatment_plan(patient: Patient):
    print("\n" + "─" * 65)
    print("  SECTION 3: Automated Treatment Plan (Delta-Nim Optimal)")
    print("─" * 65)
    print("\n  Generating optimal treatment sequence...")
    planner = TreatmentPlanner(max_lookahead=15)
    plan    = planner.generate_plan(patient.patient_id, patient.heaps())
    plan.print_plan()
    press_enter()


def demo_interactive_game(patient: Patient):
    print("\n" + "─" * 65)
    print("  SECTION 4: Interactive Treatment Game")
    print("─" * 65)
    print("""
  You are the oncologist. Apply treatment to eradicate all clusters.
  The tumor responds with optimal drug resistance (Delta-Nim strategy).

  Commands:
    treat <cluster> <dose>  — apply treatment
    hint                    — show optimal move
    top                     — show top 3 moves ranked by quality
    analyze                 — full position analysis
    quit                    — exit game
""")
    press_enter()

    game = CancerNimGame(patient.heaps(), patient_id=patient.patient_id)

    while not game.done:
        print(render_full_board(game))
        if game.turn == "tumor":
            print("  Tumor is responding (optimal resistance)...")
            rec = game.tumor_move(strategy="optimal")
            print(f"\n  {rec.clinical_description()}")
            press_enter()
            continue

        cmd = prompt("  > ").lower().split()
        if not cmd:
            continue

        if cmd[0] == "quit":
            print("  Exiting game.")
            break

        elif cmd[0] == "hint":
            print(f"\n  {game.hint()}")

        elif cmd[0] == "top":
            top = ranked_moves(game.heaps, top_k=3)
            print("\n  Top 3 treatment moves:")
            for j, m in enumerate(top):
                print(f"    {j+1}. C{m['cluster']+1}, remove {m['remove']}×10³ "
                      f"— score {m['score']}/100 — {m['rationale']}")

        elif cmd[0] == "analyze":
            print(render_theory_panel(game.heaps))
            print(render_nim_decomposition(game.heaps))

        elif cmd[0] == "treat" and len(cmd) == 3:
            try:
                cluster_idx = int(cmd[1]) - 1
                dose        = int(cmd[2])
                rec = game.oncologist_move(cluster_idx, dose)
                print(f"\n  {rec.clinical_description()}")
            except ValueError as e:
                print(f"\n  Error: {e}")

        else:
            print("  Unknown command. Try: treat <cluster> <dose> | hint | top | analyze | quit")

    if game.done:
        print("\n" + "=" * 65)
        if game.winner == "oncologist":
            print("  COMPLETE REMISSION ACHIEVED")
            print("  All tumor clusters eradicated by optimal Delta-Nim strategy.")
        else:
            print("  TREATMENT FAILURE")
            print("  Tumor maintained P-position advantage throughout.")
        print("=" * 65)
        summary = game.session_summary()
        print(f"\n  Total cycles : {summary['total_cycles']}")
        print(f"  P-position cycles: {summary['p_position_cycles']} (tumor advantage)")
        print(f"  N-position cycles: {summary['n_position_cycles']} (treatment window)")
        press_enter()


def demo_strategy_comparison(patient: Patient):
    print("\n" + "─" * 65)
    print("  SECTION 5: Strategy Comparison (Delta-Nim vs Standard of Care)")
    print("─" * 65)
    print("\n  Running 50 simulations per strategy...")
    planner  = TreatmentPlanner()
    compare  = planner.compare_strategies(patient.patient_id, patient.heaps(), n_simulations=50)
    from visualization.board import render_comparison_table
    print(render_comparison_table({
        k: v for k, v in compare.items()
        if k not in ("patient_id", "initial_heaps", "n_simulations")
    }))
    press_enter()


def demo_survival_analysis():
    print("\n" + "─" * 65)
    print("  SECTION 6: Kaplan-Meier Survival Analysis")
    print("─" * 65)
    print("\n  Simulating 100-patient cohort (all three arms)...")
    from clinical.survival import SurvivalSimulator, KaplanMeierEstimator, log_rank_test
    sim    = SurvivalSimulator(seed=42)
    events = sim.simulate_cohort(n_patients=100, tumor_type="prostate")

    km_results = []
    for arm in ["optimal_nim", "standard_of_care", "no_treatment"]:
        km = KaplanMeierEstimator(events, arm)
        km_results.append(km.summary())

    from visualization.board import render_survival_summary
    print(render_survival_summary(km_results))

    lr = log_rank_test(events, "optimal_nim", "standard_of_care")
    print(f"\n  Log-rank test (Δ-Nim vs Standard of Care):")
    print(f"    χ² = {lr['chi2']}  |  p = {lr['p_value']}")
    print(f"    {lr['interpretation']}")
    print()
    print("  Clinical interpretation:")
    opt_med = km_results[0]["median_survival"]
    soc_med = km_results[1]["median_survival"]
    if opt_med and soc_med:
        ratio = opt_med / soc_med if soc_med else 0
        print(f"    Δ-Nim optimal: {opt_med:.1f} weeks median PFS")
        print(f"    Standard SoC : {soc_med:.1f} weeks median PFS")
        print(f"    Improvement  : {(ratio-1)*100:.0f}% longer progression-free survival")
    press_enter()


# ── Entry point ───────────────────────────────────────────────────────────────

def run_interactive_demo():
    intro_screen()
    patient = demo_patient_generation()
    demo_position_analysis(patient)

    skip = prompt("  Run automated treatment plan? [Y/n]: ").lower()
    if skip != "n":
        demo_treatment_plan(patient)

    skip = prompt("  Play the interactive treatment game? [Y/n]: ").lower()
    if skip != "n":
        demo_interactive_game(patient)

    skip = prompt("  Run strategy comparison simulation? [Y/n]: ").lower()
    if skip != "n":
        demo_strategy_comparison(patient)

    skip = prompt("  Run Kaplan-Meier survival analysis? [Y/n]: ").lower()
    if skip != "n":
        demo_survival_analysis()

    print("\n  Demo complete. Thank you for your time.")
    print("  Questions? See the paper for full theoretical proofs.\n")


if __name__ == "__main__":
    run_interactive_demo()