"""
ai/resistance.py
================
Tumor drug resistance models for Delta-Nim adaptive therapy.

Models how real tumors respond to treatment using evolutionary game theory:

1. OptimalResistance    - Perfect information, always plays optimal Delta-Nim move.
                          Represents a fully evolved, maximally resistant tumor.

2. StochasticResistance - Probabilistic model where resistance strength decays
                          over treatment cycles (models acquired drug sensitivity).
                          Pr(optimal move) = base_resistance * decay^cycle

3. ClonalResistance     - Each cluster has its own resistance level. High-resistance
                          clusters preferentially survive, mimicking clonal selection.

4. AdaptiveResistance   - Tumor resistance evolves in response to treatment patterns.
                          Mimics the Zhang et al. (2022) adaptive therapy model.

Reference:
  Zhang et al. (2022). Integrating evolutionary dynamics into treatment of metastatic
  castrate-resistant prostate cancer. Nature Communications.
  NCT02415621 clinical trial.
"""

from __future__ import annotations
from typing import Optional
import random
import math

from engine.core import (
    nim_sum, regime, count_equal_pairs, is_p_position,
    support_size, total_cells, Regime, DENSE_CUTOFF, MIN_PAIRS
)
from ai.optimal import compute_optimal_move, sparse_any_move, dense_p_position_move


# ── Base resistance model ─────────────────────────────────────────────────────

class ResistanceModel:
    """Abstract base for tumor resistance strategies."""

    def choose_move(self, heaps: list[int], cycle: int) -> tuple[int, int]:
        raise NotImplementedError

    def resistance_level(self, cycle: int) -> float:
        raise NotImplementedError


# ── 1. Optimal (fully evolved) resistance ────────────────────────────────────

class OptimalResistance(ResistanceModel):
    """
    Tumor always plays the game-theoretically optimal Delta-Nim move.
    Represents complete drug resistance — the worst case for oncologists.
    """
    def __init__(self, noise: float = 0.05):
        self.noise = noise

    def choose_move(self, heaps: list[int], cycle: int) -> tuple[int, int]:
        import random
        if random.random() < self.noise:
            from ai.resistance import StochasticResistance
            return StochasticResistance()._random_move(heaps)

        move = compute_optimal_move(heaps)
        if move is not None:
            return move
        r = regime(heaps)
        if r == Regime.SPARSE:
            return sparse_any_move(heaps)
        else:
            return dense_p_position_move(heaps)

    def resistance_level(self, cycle: int) -> float:
        return 1.0

    def __repr__(self):
        return "OptimalResistance(level=1.0)"


# ── 2. Stochastic (decaying) resistance ──────────────────────────────────────

class StochasticResistance(ResistanceModel):
    """
    Pr(optimal move) = base_resistance * decay^cycle.
    With probability (1 - Pr), tumor makes a random move.
    Models acquired drug sensitivity under sustained treatment.

    Parameters
    ----------
    base_resistance : float in [0, 1]  — initial resistance probability
    decay           : float in [0, 1]  — per-cycle multiplicative decay
    seed            : int              — for reproducibility
    """

    def __init__(self, base_resistance: float = 0.85, decay: float = 0.97, seed: int = 42):
        self.base_resistance = base_resistance
        self.decay           = decay
        self.rng             = random.Random(seed)

    def resistance_level(self, cycle: int) -> float:
        return max(0.05, self.base_resistance * (self.decay ** cycle))

    def choose_move(self, heaps: list[int], cycle: int) -> tuple[int, int]:
        prob = self.resistance_level(cycle)
        if self.rng.random() < prob:
            move = compute_optimal_move(heaps)
            if move is not None:
                return move
        return self._random_move(heaps)

    def _random_move(self, heaps: list[int]) -> tuple[int, int]:
        nonzero = [(i, h) for i, h in enumerate(heaps) if h > 0]
        i, h = self.rng.choice(nonzero)
        remove = self.rng.randint(1, max(1, h // 2))
        return (i, remove)

    def __repr__(self):
        return f"StochasticResistance(base={self.base_resistance}, decay={self.decay})"


# ── 3. Clonal selection resistance ───────────────────────────────────────────

class ClonalResistance(ResistanceModel):
    """
    Each cluster has an intrinsic resistance score drawn at game start.
    Clusters with higher resistance are less affected by treatment
    and more likely to be the ones the tumor 'chooses' to sacrifice
    (models clonal selection — sensitive clones die, resistant survive).

    The tumor preferentially sheds from its lowest-resistance clusters,
    protecting high-resistance clones.
    """

    def __init__(self, heaps: list[int], seed: int = 0):
        rng = random.Random(seed)
        self.cluster_resistance = {
            i: rng.uniform(0.1, 1.0) for i in range(len(heaps))
        }

    def resistance_level(self, cycle: int) -> float:
        return sum(self.cluster_resistance.values()) / max(len(self.cluster_resistance), 1)

    def choose_move(self, heaps: list[int], cycle: int) -> tuple[int, int]:
        """
        Shed cells preferentially from low-resistance clusters
        (sacrifice sensitive clones to protect resistant core).
        """
        nonzero = [(i, h) for i, h in enumerate(heaps) if h > 0]
        if not nonzero:
            return (0, 0)
        # Sort by resistance ascending — shed from most sensitive first
        nonzero.sort(key=lambda x: self.cluster_resistance.get(x[0], 0.5))
        # With probability proportional to resistance differential, pick sensitives
        weights = [1.0 - self.cluster_resistance.get(i, 0.5) + 0.01 for i, _ in nonzero]
        total_w = sum(weights)
        r = random.random() * total_w
        chosen_i, chosen_h = nonzero[0]
        cumulative = 0
        for (idx, h), w in zip(nonzero, weights):
            cumulative += w
            if cumulative >= r:
                chosen_i, chosen_h = idx, h
                break
        # Remove a small number (tumor sheds minimally from sensitive clones)
        remove = max(1, int(chosen_h * (1 - self.cluster_resistance.get(chosen_i, 0.5)) * 0.4 + 0.5))
        remove = min(remove, chosen_h)
        return (chosen_i, remove)

    def get_resistance_profile(self) -> dict[int, float]:
        return dict(self.cluster_resistance)

    def __repr__(self):
        return f"ClonalResistance(clusters={len(self.cluster_resistance)})"


# ── 4. Adaptive resistance (Zhang et al. model) ──────────────────────────────

class AdaptiveResistance(ResistanceModel):
    """
    Tumor evolves its resistance strategy in response to observed treatment patterns.
    Models the adaptive therapy concept from Zhang et al. (2022), NCT02415621.

    Tracks oncologist's last N moves. If oncologist repeatedly targets the same
    cluster, tumor up-regulates resistance there. If oncologist spreads treatment
    evenly, tumor attempts to cluster pairs (dense regime strategy).

    resistance_memory : how many past treatment cycles to remember
    adaptation_rate   : how quickly tumor updates its strategy
    """

    def __init__(self, resistance_memory: int = 5, adaptation_rate: float = 0.3):
        self.memory          = resistance_memory
        self.adaptation_rate = adaptation_rate
        self.treatment_history: list[int] = []   # which clusters oncologist targeted
        self.cluster_evolved_resistance: dict[int, float] = {}
        self.current_strategy = "optimal"

    def record_treatment(self, cluster_idx: int):
        """Called after each oncologist move to update tumor's adaptive memory."""
        self.treatment_history.append(cluster_idx)
        if len(self.treatment_history) > self.memory:
            self.treatment_history.pop(0)
        # Up-regulate resistance in frequently targeted clusters
        for idx in self.treatment_history:
            prev = self.cluster_evolved_resistance.get(idx, 0.0)
            self.cluster_evolved_resistance[idx] = min(1.0, prev + self.adaptation_rate / self.memory)

    def resistance_level(self, cycle: int) -> float:
        if not self.cluster_evolved_resistance:
            return 0.5
        return sum(self.cluster_evolved_resistance.values()) / len(self.cluster_evolved_resistance)

    def _detect_oncologist_pattern(self) -> str:
        """Infer oncologist's strategy from recent treatment history."""
        if len(self.treatment_history) < 2:
            return "unknown"
        unique = len(set(self.treatment_history))
        if unique == 1:
            return "focused"        # targeting one cluster repeatedly
        elif unique >= len(self.treatment_history) * 0.8:
            return "spread"         # spreading treatment evenly
        else:
            return "mixed"

    def choose_move(self, heaps: list[int], cycle: int) -> tuple[int, int]:
        pattern = self._detect_oncologist_pattern()

        if pattern == "focused":
            # Oncologist focusing one cluster — tumor up-regulates there,
            # and plays to restore pairing structure elsewhere
            from ai.optimal import dense_winning_move
            move = dense_winning_move(heaps) if support_size(heaps) > DENSE_CUTOFF else compute_optimal_move(heaps)
        elif pattern == "spread":
            # Oncologist spreading — tumor plays optimal nim-sum strategy
            move = compute_optimal_move(heaps)
        else:
            move = compute_optimal_move(heaps)

        if move is None:
            nonzero = [(i, h) for i, h in enumerate(heaps) if h > 0]
            if nonzero:
                i, h = nonzero[0]
                return (i, 1)
            return (0, 1)
        return move

    def __repr__(self):
        return f"AdaptiveResistance(memory={self.memory}, rate={self.adaptation_rate})"


# ── Factory ───────────────────────────────────────────────────────────────────

def build_resistance_model(strategy: str, heaps: list[int] = None, **kwargs) -> ResistanceModel:
    """
    Factory for resistance models.
    strategy : 'optimal' | 'stochastic' | 'clonal' | 'adaptive'
    """
    if strategy == "optimal":
        return OptimalResistance()
    elif strategy == "stochastic":
        return StochasticResistance(**kwargs)
    elif strategy == "clonal":
        assert heaps is not None, "ClonalResistance requires initial heaps."
        return ClonalResistance(heaps, **kwargs)
    elif strategy == "adaptive":
        return AdaptiveResistance(**kwargs)
    else:
        raise ValueError(f"Unknown resistance strategy: {strategy}")


def stochastic_resistance_move(heaps: list[int], cycle: int = 0) -> tuple[int, int]:
    """Convenience wrapper used by CancerNimGame.tumor_move('stochastic')."""
    model = StochasticResistance()
    return model.choose_move(heaps, cycle)