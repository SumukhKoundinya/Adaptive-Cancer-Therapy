"""
data/generator.py
=================
Synthetic data generation for Delta-Nim adaptive therapy experiments.

Generates:
  1. Position datasets (for ML training/validation)
  2. Patient cohorts (for survival analysis)
  3. Game trajectory datasets (for treatment outcome analysis)
  4. Regime boundary samples (for Pigeonhole validation)
"""

from __future__ import annotations
from dataclasses import dataclass
import random
import math
from typing import Optional

from engine.core import (
    nim_sum, regime, support_size, count_equal_pairs,
    is_p_position, total_cells, feature_vector,
    DENSE_CUTOFF, MIN_PAIRS
)


# ── Heap generators ───────────────────────────────────────────────────────────

class HeapGenerator:
    """
    Generates heap configurations for various experimental conditions.
    Mirrors the sampling strategies described in the paper.
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def geometric(self, n: int, p: float = 0.15, max_val: int = 200) -> list[int]:
        """
        Geometric distribution heap sizes.
        Most positions have large support (dense regime dominant).
        Used in paper: |supp| ~ Geometric(p=0.15).
        """
        heaps = []
        for _ in range(n):
            v = int(math.log(self.rng.random() + 1e-12) / math.log(1 - p)) + 1
            heaps.append(min(v, max_val))
        return heaps

    def uniform_sparse(self, n_lo: int = 2, n_hi: int = 55,
                       h_lo: int = 1, h_hi: int = 100) -> list[int]:
        n = self.rng.randint(n_lo, n_hi)
        return [self.rng.randint(h_lo, h_hi) for _ in range(n)]

    def uniform_dense(self, n_lo: int = 61, n_hi: int = 120,
                      h_lo: int = 1, h_hi: int = 50) -> list[int]:
        n = self.rng.randint(n_lo, n_hi)
        return [self.rng.randint(h_lo, h_hi) for _ in range(n)]

    def near_boundary(self, delta: int = 3) -> list[int]:
        """
        Generate positions near the sparse/dense boundary (|supp| ≈ 60).
        Used to validate Pigeonhole Principle at the cutoff.
        """
        n = DENSE_CUTOFF + self.rng.randint(-delta, delta)
        n = max(1, n)
        return [self.rng.randint(1, 30) for _ in range(n)]

    def constructed_p_sparse(self, n_pairs: int = None) -> list[int]:
        """Guaranteed sparse P-position: pairs of equal heaps, nim-sum = 0."""
        if n_pairs is None:
            n_pairs = self.rng.randint(2, 10)
        heaps = []
        for _ in range(n_pairs):
            v = self.rng.randint(1, 25)
            heaps.extend([v, v])
        return heaps

    def constructed_p_dense(self, n_pairs: int = None) -> list[int]:
        """Guaranteed dense P-position: >= 30 equal pairs."""
        if n_pairs is None:
            n_pairs = self.rng.randint(MIN_PAIRS, MIN_PAIRS + 10)
        heaps = []
        for _ in range(n_pairs):
            v = self.rng.randint(1, 20)
            heaps.extend([v, v])
        n_extra = self.rng.randint(0, 8)
        for _ in range(n_extra):
            heaps.append(self.rng.randint(1, 15))
        self.rng.shuffle(heaps)
        return heaps

    def clinical_tumor(self, tumor_type: str = "prostate") -> list[int]:
        """Tumor-type-specific heap configuration."""
        from clinical.patient import TUMOR_CONFIGS
        config = TUMOR_CONFIGS.get(tumor_type, TUMOR_CONFIGS["prostate"])
        n_lo, n_hi = config["n_clusters"]
        c_lo, c_hi = config["cell_range"]
        n = self.rng.randint(n_lo, n_hi)
        return [self.rng.randint(c_lo, c_hi) for _ in range(n)]


# ── Game trajectory recorder ──────────────────────────────────────────────────

@dataclass
class GameTrajectory:
    """Complete record of a simulated game."""
    patient_id:       str
    initial_heaps:    list[int]
    initial_position: str
    initial_regime:   str
    initial_burden:   int
    winner:           str
    total_cycles:     int
    final_burden:     int
    burden_history:   list[int]
    nim_sum_history:  list[int]
    pairs_history:    list[int]
    position_history: list[str]
    oncologist_strategy: str
    tumor_strategy:   str

    def burden_reduction(self) -> float:
        if self.initial_burden == 0:
            return 0.0
        return (self.initial_burden - self.final_burden) / self.initial_burden

    def to_dict(self) -> dict:
        return {
            "patient_id":          self.patient_id,
            "initial_position":    self.initial_position,
            "initial_regime":      self.initial_regime,
            "initial_burden":      self.initial_burden,
            "winner":              self.winner,
            "total_cycles":        self.total_cycles,
            "final_burden":        self.final_burden,
            "burden_reduction":    round(self.burden_reduction(), 3),
            "oncologist_strategy": self.oncologist_strategy,
            "tumor_strategy":      self.tumor_strategy,
        }


class GameSimulator:
    """
    Simulate many games with different strategy combinations
    and record full trajectories.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.generator = HeapGenerator(seed=seed)

    def simulate_one(
        self,
        heaps:    list[int],
        onco_strategy: str = "optimal",
        tumor_strategy: str = "optimal",
        max_cycles: int = 300,
        patient_id: str = "PT-0000",
    ) -> GameTrajectory:
        from engine.game import CancerNimGame
        from ai.optimal import compute_optimal_move
        from ai.resistance import build_resistance_model

        game = CancerNimGame(list(heaps), patient_id=patient_id)
        resistance = build_resistance_model(tumor_strategy, heaps=list(heaps))

        burden_hist   = [total_cells(heaps)]
        nim_hist      = [nim_sum(heaps)]
        pairs_hist    = [count_equal_pairs(heaps)]
        pos_hist      = ["P" if is_p_position(heaps) else "N"]

        cycles = 0
        while not game.done and cycles < max_cycles:
            if game.turn == "oncologist":
                if onco_strategy == "optimal":
                    move = compute_optimal_move(game.heaps)
                    if move is None:
                        nz = [(i, h) for i, h in enumerate(game.heaps) if h > 0]
                        if not nz: break
                        move = (nz[0][0], 1)
                elif onco_strategy == "max_dose":
                    nz = [(i, h) for i, h in enumerate(game.heaps) if h > 0]
                    if not nz: break
                    move = max(nz, key=lambda x: x[1])
                    move = (move[0], move[1])
                elif onco_strategy == "random":
                    nz = [(i, h) for i, h in enumerate(game.heaps) if h > 0]
                    if not nz: break
                    i, h = self.rng.choice(nz)
                    move = (i, self.rng.randint(1, h))
                else:
                    raise ValueError(f"Unknown oncologist strategy: {onco_strategy}")
                try:
                    game.oncologist_move(*move)
                except Exception:
                    break
            else:
                move = resistance.choose_move(game.heaps, cycles)
                try:
                    game._apply(move[0], min(move[1], game.heaps[move[0]]), "tumor")
                    if not game.done:
                        game.turn = "oncologist"
                except Exception:
                    break
            cycles += 1
            burden_hist.append(total_cells(game.heaps))
            nim_hist.append(nim_sum(game.heaps))
            pairs_hist.append(count_equal_pairs(game.heaps))
            pos_hist.append("P" if is_p_position(game.heaps) else "N")

        return GameTrajectory(
            patient_id=patient_id,
            initial_heaps=list(heaps),
            initial_position="P" if is_p_position(heaps) else "N",
            initial_regime=regime(heaps),
            initial_burden=total_cells(heaps),
            winner=game.winner or "none",
            total_cycles=cycles,
            final_burden=total_cells(game.heaps),
            burden_history=burden_hist,
            nim_sum_history=nim_hist,
            pairs_history=pairs_hist,
            position_history=pos_hist,
            oncologist_strategy=onco_strategy,
            tumor_strategy=tumor_strategy,
        )

    def simulate_cohort(
        self,
        n_games:        int = 100,
        tumor_type:     str = "prostate",
        onco_strategies: list[str] = None,
        tumor_strategy:  str = "stochastic",
    ) -> list[GameTrajectory]:
        if onco_strategies is None:
            onco_strategies = ["optimal", "max_dose", "random"]
        trajectories = []
        for i in range(n_games):
            heaps = self.generator.clinical_tumor(tumor_type)
            for strat in onco_strategies:
                traj = self.simulate_one(
                    heaps=list(heaps),
                    onco_strategy=strat,
                    tumor_strategy=tumor_strategy,
                    patient_id=f"PT-{i:04d}_{strat[:3]}",
                )
                trajectories.append(traj)
        return trajectories

    def compute_strategy_comparison(self, trajectories: list[GameTrajectory]) -> dict:
        """Aggregate trajectory outcomes by oncologist strategy."""
        from collections import defaultdict
        by_strat: dict[str, list[GameTrajectory]] = defaultdict(list)
        for t in trajectories:
            by_strat[t.oncologist_strategy].append(t)

        results = {}
        for strat, trajs in by_strat.items():
            n = len(trajs)
            wins = sum(1 for t in trajs if t.winner == "oncologist")
            avg_cycles = sum(t.total_cycles for t in trajs) / n
            avg_reduction = sum(t.burden_reduction() for t in trajs) / n
            results[strat] = {
                "n_games":          n,
                "remission_rate":   round(wins / n, 3),
                "mean_cycles":      round(avg_cycles, 1),
                "mean_burden_reduction": round(avg_reduction, 3),
            }
        return results