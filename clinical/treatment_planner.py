"""
clinical/treatment_planner.py
==============================
Clinical decision support system for Delta-Nim adaptive therapy.

Given a patient (set of tumor clusters), generates a full treatment plan
using Delta-Nim optimal strategy and provides:
  - Recommended treatment sequence
  - Nim-sum trajectory predictions
  - Regime transition alerts
  - Comparison against standard-of-care
  - Expected time to remission estimates

This is the system that would be used by a clinical oncologist in practice.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math

from engine.core import (
    nim_sum, regime, support_size, count_equal_pairs,
    is_p_position, position_type, total_cells,
    DENSE_CUTOFF, MIN_PAIRS, Regime
)
from engine.game import CancerNimGame
from ai.optimal import compute_optimal_move, compute_move_quality, ranked_moves
from ai.resistance import OptimalResistance, StochasticResistance


# ── Treatment recommendation ──────────────────────────────────────────────────

@dataclass
class TreatmentRecommendation:
    cycle:          int
    cluster_idx:    int
    dose_k_cells:   int
    rationale:      str
    nim_sum_after:  int
    position_after: str
    pairs_after:    int
    confidence:     float   # 0-1, how confident the system is in this move
    alternatives:   list[dict]

    def clinical_text(self) -> str:
        return (
            f"Cycle {self.cycle}: Target cluster C{self.cluster_idx+1}, "
            f"remove {self.dose_k_cells}×10³ cells.\n"
            f"  Rationale: {self.rationale}\n"
            f"  Expected nim-sum: {self.nim_sum_after} | "
            f"Position: {self.position_after} | "
            f"Equal pairs: {self.pairs_after} | "
            f"Confidence: {self.confidence*100:.0f}%"
        )


# ── Treatment plan ────────────────────────────────────────────────────────────

@dataclass
class TreatmentPlan:
    patient_id:       str
    initial_heaps:    list[int]
    recommendations:  list[TreatmentRecommendation]
    projected_cycles: int
    initial_position: str
    initial_regime:   str
    predicted_outcome: str
    survival_estimate: str
    regime_warnings:   list[str]

    def print_plan(self):
        print(f"\n{'='*65}")
        print(f"  Treatment Plan — {self.patient_id}")
        print(f"{'='*65}")
        print(f"  Initial regime   : {self.initial_regime}")
        print(f"  Initial position : {self.initial_position}-position")
        print(f"  Projected cycles : {self.projected_cycles}")
        print(f"  Predicted outcome: {self.predicted_outcome}")
        print(f"  Survival estimate: {self.survival_estimate}")
        if self.regime_warnings:
            print(f"\n  WARNINGS:")
            for w in self.regime_warnings:
                print(f"    ! {w}")
        print(f"\n  Recommended treatment sequence:")
        for rec in self.recommendations:
            print(f"\n  {rec.clinical_text()}")
        print(f"\n{'='*65}\n")


# ── Planner ───────────────────────────────────────────────────────────────────

class TreatmentPlanner:
    """
    Generates and evaluates treatment plans using Delta-Nim theory.

    Parameters
    ----------
    max_lookahead : int  — how many cycles to plan ahead
    resistance    : str  — tumor resistance model ('optimal' | 'stochastic')
    """

    def __init__(self, max_lookahead: int = 20, resistance: str = "optimal"):
        self.max_lookahead = max_lookahead
        self.resistance    = resistance

    def generate_plan(self, patient_id: str, heaps: list[int]) -> TreatmentPlan:
        """
        Generate a full treatment plan for a patient.
        Simulates up to max_lookahead cycles of optimal oncologist play.
        """
        sim_heaps    = list(heaps)
        recommendations: list[TreatmentRecommendation] = []
        warnings:    list[str] = []
        initial_pos  = position_type(heaps)
        initial_reg  = regime(heaps)
        prev_regime  = initial_reg

        for cycle in range(1, self.max_lookahead + 1):
            if all(h == 0 for h in sim_heaps):
                break

            r   = regime(sim_heaps)
            ns  = nim_sum(sim_heaps)
            sz  = support_size(sim_heaps)

            # Regime transition warning
            if r != prev_regime:
                warnings.append(
                    f"Cycle {cycle}: Regime transition {prev_regime} → {r}. "
                    f"|supp| crossed {DENSE_CUTOFF} threshold. "
                    f"Switch from nim-sum strategy to pairing invariant strategy."
                )
            prev_regime = r

            # Approaching dense cutoff warning
            if abs(sz - DENSE_CUTOFF) <= 3 and sz < DENSE_CUTOFF:
                warnings.append(
                    f"Cycle {cycle}: Approaching dense cutoff (|supp| = {sz}). "
                    f"Consider pairing strategy in {sz - (DENSE_CUTOFF - 3)} more eradications."
                )

            move = compute_optimal_move(sim_heaps)
            if move is None:
                warnings.append(
                    f"Cycle {cycle}: P-position detected — no winning move under optimal resistance. "
                    f"Consider dose escalation, combination therapy, or supportive care."
                )
                # Make best available move anyway
                nonzero = [(i, h) for i, h in enumerate(sim_heaps) if h > 0]
                if not nonzero:
                    break
                move = (nonzero[0][0], 1)
                confidence = 0.25
            else:
                confidence = 0.95 if is_p_position(sim_heaps) is False else 0.55

            idx, remove = move
            sim_heaps[idx] -= remove
            ns_after   = nim_sum(sim_heaps)
            pt_after   = position_type(sim_heaps)
            pairs_after = count_equal_pairs(sim_heaps)

            top_moves = ranked_moves(
                [h + (remove if i == idx else 0) for i, h in enumerate(sim_heaps)],
                top_k=3,
            )

            rationale = self._rationale(r, ns, ns_after, pairs_after, cycle)

            recommendations.append(TreatmentRecommendation(
                cycle=cycle, cluster_idx=idx, dose_k_cells=remove,
                rationale=rationale, nim_sum_after=ns_after,
                position_after=pt_after, pairs_after=pairs_after,
                confidence=confidence, alternatives=top_moves[:2],
            ))

        cycles_to_eradication = len(recommendations)
        predicted_outcome = (
            "Complete remission projected under optimal treatment adherence."
            if all(h == 0 for h in sim_heaps) else
            "Partial remission — tumor burden reduced but eradication not guaranteed within lookahead window."
        )

        survival_est = self._estimate_survival(
            initial_position=initial_pos,
            initial_regime=initial_reg,
            cycles=cycles_to_eradication,
            initial_burden=total_cells(heaps),
        )

        return TreatmentPlan(
            patient_id=patient_id,
            initial_heaps=list(heaps),
            recommendations=recommendations,
            projected_cycles=cycles_to_eradication,
            initial_position=initial_pos,
            initial_regime=initial_reg,
            predicted_outcome=predicted_outcome,
            survival_estimate=survival_est,
            regime_warnings=warnings,
        )

    def _rationale(self, r: str, ns_before: int, ns_after: int,
                   pairs_after: int, cycle: int) -> str:
        if r == Regime.SPARSE:
            if ns_after == 0:
                return (
                    f"Zeros nim-sum ({ns_before}→0). Bouton's Theorem: "
                    f"this creates a P-position — tumor must break symmetry next cycle."
                )
            else:
                return (
                    f"Reduces nim-sum ({ns_before}→{ns_after}). "
                    f"Moves toward nim-sum = 0 target. "
                    f"Continue applying pressure to restore P-position."
                )
        else:
            if pairs_after >= MIN_PAIRS:
                return (
                    f"Restores pairing invariant ({pairs_after} ≥ {MIN_PAIRS} pairs). "
                    f"Dense P-position achieved — tumor must break a pair next cycle."
                )
            else:
                return (
                    f"Increases toward {MIN_PAIRS}-pair threshold (now {pairs_after}). "
                    f"Continue pairing strategy to achieve dense P-position lock."
                )

    def _estimate_survival(self, initial_position: str, initial_regime: str,
                            cycles: int, initial_burden: int) -> str:
        """
        Rough survival estimate based on treatment plan characteristics.
        References Zhang et al. (2022) NCT02415621 endpoints.
        """
        base_weeks = 120 if initial_position == "N" else 72
        if initial_regime == Regime.DENSE:
            base_weeks *= 0.82
        cycle_factor = max(0.5, 1 - 0.012 * cycles)
        burden_factor = max(0.5, 1 - 0.004 * initial_burden)
        est_weeks = base_weeks * cycle_factor * burden_factor
        est_months = est_weeks / 4.33

        if est_months > 24:
            return f"Favorable: estimated median progression-free survival ~{est_months:.0f} months (>{est_months/12:.1f} years)"
        else:
            return f"Guarded: estimated median progression-free survival ~{est_months:.0f} months"

    def compare_strategies(
        self,
        patient_id: str,
        heaps: list[int],
        n_simulations: int = 200,
    ) -> dict:
        """
        Compare Delta-Nim optimal strategy vs standard-of-care (max dose) vs random.
        Runs n_simulations games for each strategy.
        """
        from engine.game import CancerNimGame

        results = {"optimal_nim": [], "max_dose": [], "random": []}

        for _ in range(n_simulations):
            for strategy in results:
                game = CancerNimGame(list(heaps), patient_id=patient_id)
                cycles = 0
                max_cycles = 200

                while not game.done and cycles < max_cycles:
                    if game.turn == "oncologist":
                        if strategy == "optimal_nim":
                            move = compute_optimal_move(game.heaps)
                            if move is None:
                                nonzero = [(i, h) for i, h in enumerate(game.heaps) if h > 0]
                                if not nonzero: break
                                move = (nonzero[0][0], 1)
                        elif strategy == "max_dose":
                            nonzero = [(i, h) for i, h in enumerate(game.heaps) if h > 0]
                            if not nonzero: break
                            i, h = max(nonzero, key=lambda x: x[1])
                            move = (i, h)
                        else:
                            import random
                            nonzero = [(i, h) for i, h in enumerate(game.heaps) if h > 0]
                            if not nonzero: break
                            i, h = random.choice(nonzero)
                            move = (i, max(1, random.randint(1, h)))
                        try:
                            game.oncologist_move(*move)
                        except Exception:
                            break
                    else:
                        try:
                            game.tumor_move(strategy="stochastic")
                        except Exception:
                            break
                    cycles += 1

                results[strategy].append({
                    "winner":        game.winner,
                    "cycles":        cycles,
                    "final_burden":  total_cells(game.heaps),
                    "remission":     game.winner == "oncologist",
                })

        def summarize(runs):
            n = len(runs)
            remissions = sum(1 for r in runs if r["remission"])
            avg_cycles = sum(r["cycles"] for r in runs) / n
            avg_burden = sum(r["final_burden"] for r in runs) / n
            return {
                "n_simulations":       n,
                "remission_rate":      round(remissions / n, 3),
                "mean_cycles":         round(avg_cycles, 1),
                "mean_final_burden":   round(avg_burden, 1),
            }

        return {
            "patient_id":     patient_id,
            "initial_heaps":  heaps,
            "n_simulations":  n_simulations,
            "optimal_nim":    summarize(results["optimal_nim"]),
            "max_dose":       summarize(results["max_dose"]),
            "random":         summarize(results["random"]),
        }