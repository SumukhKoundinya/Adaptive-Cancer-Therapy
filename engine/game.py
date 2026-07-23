"""
engine/game.py
==============
CancerNimGame: full game state machine for Delta-Nim adaptive therapy.

Models one treatment course as a combinatorial game between:
  Player I  (Oncologist) — applies targeted therapy to a cluster
  Player II (Tumor)      — optimal or stochastic drug resistance response

Each "turn" maps to one treatment cycle (e.g. one week of chemo dosing).
The game ends when all clusters are eradicated (complete remission) or
the tumor achieves a P-position lock from which it cannot be dislodged.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time

from engine.core import (
    nim_sum, support_size, regime, count_equal_pairs,
    is_p_position, position_type, total_cells, feature_vector,
    DENSE_CUTOFF, MIN_PAIRS, Regime
)


# ── Move record ───────────────────────────────────────────────────────────────

@dataclass
class MoveRecord:
    move_num:      int
    actor:         str          # 'oncologist' | 'tumor'
    cluster_idx:   int
    cells_removed: int
    heaps_before:  list[int]
    heaps_after:   list[int]
    nim_sum_after: int
    regime:        str
    position_after: str         # 'P' | 'N'
    pairs_after:   int
    timestamp:     float = field(default_factory=time.time)

    def clinical_description(self) -> str:
        delta = self.cells_removed
        cluster = f"C{self.cluster_idx + 1}"
        before = self.heaps_before[self.cluster_idx]
        after  = self.heaps_after[self.cluster_idx]
        burden = total_cells(self.heaps_after)
        if self.actor == "oncologist":
            return (
                f"Cycle {self.move_num}: Oncologist targeted {cluster} "
                f"({before}→{after}×10³ cells, -{delta}×10³). "
                f"Total burden: {burden}×10³. "
                f"Nim-sum: {self.nim_sum_after}. Position: {self.position_after}."
            )
        else:
            return (
                f"Cycle {self.move_num}: Tumor resistance response at {cluster} "
                f"({before}→{after}×10³ cells, -{delta}×10³ via apoptosis/evasion). "
                f"Equal pairs maintained: {self.pairs_after}. "
                f"Nim-sum: {self.nim_sum_after}. Position: {self.position_after}."
            )


# ── Game state ────────────────────────────────────────────────────────────────

class CancerNimGame:
    """
    Full Delta-Nim game state machine.

    Parameters
    ----------
    heaps       : initial cluster sizes (×10³ cells)
    patient_id  : optional patient identifier for clinical logging
    """

    def __init__(self, heaps: list[int], patient_id: str = "PT-0000"):
        assert all(h >= 0 for h in heaps), "Heap sizes must be non-negative."
        assert any(h > 0 for h in heaps), "At least one cluster must be active."
        self.initial_heaps  = list(heaps)
        self.heaps          = list(heaps)
        self.patient_id     = patient_id
        self.turn           = "oncologist"
        self.history:  list[MoveRecord] = []
        self.winner:   Optional[str]    = None
        self.done:     bool             = False
        self.move_num: int              = 0
        self._start_time = time.time()

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        ns    = nim_sum(self.heaps)
        sz    = support_size(self.heaps)
        r     = regime(self.heaps)
        pairs = count_equal_pairs(self.heaps)
        p_pos = is_p_position(self.heaps)
        return {
            "patient_id":        self.patient_id,
            "heaps":             self.heaps[:],
            "initial_heaps":     self.initial_heaps[:],
            "support_size":      sz,
            "regime":            r,
            "nim_sum":           ns,
            "equal_pairs":       pairs,
            "pairing_invariant": pairs >= MIN_PAIRS,
            "position_type":     "P" if p_pos else "N",
            "turn":              self.turn,
            "winner":            self.winner,
            "done":              self.done,
            "total_cells":       total_cells(self.heaps),
            "initial_burden":    total_cells(self.initial_heaps),
            "burden_reduction":  round(
                (1 - total_cells(self.heaps) / max(total_cells(self.initial_heaps), 1)) * 100, 1
            ),
            "move_count":        self.move_num,
            "elapsed_cycles":    self.move_num,
            "features":          feature_vector(self.heaps),
        }

    def explain(self) -> str:
        st = self.status()
        r  = st["regime"]
        lines = [
            f"Patient       : {self.patient_id}",
            f"Clusters      : {self.heaps}",
            f"Support size  : {st['support_size']} (cutoff = {DENSE_CUTOFF})",
            f"Regime        : {r}",
            f"Nim-sum       : {st['nim_sum']}",
            f"Equal pairs   : {st['equal_pairs']} / {MIN_PAIRS} needed",
            f"Position      : {st['position_type']}-position "
            f"({'oncologist loses' if st['position_type'] == 'P' else 'oncologist can win'} with optimal play)",
            f"Turn          : {self.turn}",
            f"Burden        : {st['total_cells']}×10³ cells ({st['burden_reduction']}% reduction from initial)",
        ]
        if r == Regime.SPARSE:
            lines.append(
                "Theory: Bouton's Theorem — sparse P-position iff nim-sum = 0. "
                "Optimal treatment zeros the nim-sum each cycle."
            )
        else:
            lines.append(
                f"Theory: Pairing invariant — tumor maintains {st['equal_pairs']} equal-cluster pairs. "
                f"Needs {MIN_PAIRS} to lock a P-position. "
                f"Pigeonhole guarantees this strategy always exists for |supp| > {DENSE_CUTOFF}."
            )
        return "\n".join(lines)

    # ── Validation ────────────────────────────────────────────────────────────

    def validate_move(self, cluster_idx: int, remove: int) -> Optional[str]:
        if self.done:
            return "Game is over."
        if self.turn != "oncologist":
            return "Not the oncologist's turn."
        if not (0 <= cluster_idx < len(self.heaps)):
            return f"Cluster index {cluster_idx} out of range (0–{len(self.heaps)-1})."
        if self.heaps[cluster_idx] == 0:
            return f"Cluster C{cluster_idx+1} is already eradicated."
        if remove < 1:
            return "Must remove at least 1×10³ cells."
        if remove > self.heaps[cluster_idx]:
            return (
                f"Cannot remove {remove}×10³ from C{cluster_idx+1} "
                f"(only {self.heaps[cluster_idx]}×10³ remain)."
            )
        return None

    # ── Apply moves ───────────────────────────────────────────────────────────

    def _apply(self, cluster_idx: int, remove: int, actor: str) -> MoveRecord:
        self.move_num += 1
        before = self.heaps[:]
        self.heaps[cluster_idx] -= remove
        after  = self.heaps[:]
        ns     = nim_sum(after)
        r      = regime(after)
        pairs  = count_equal_pairs(after)
        pt     = position_type(after)
        record = MoveRecord(
            move_num=self.move_num, actor=actor,
            cluster_idx=cluster_idx, cells_removed=remove,
            heaps_before=before, heaps_after=after,
            nim_sum_after=ns, regime=r,
            position_after=pt, pairs_after=pairs,
        )
        self.history.append(record)
        if all(h == 0 for h in self.heaps):
            self.done   = True
            self.winner = actor
        return record

    def oncologist_move(self, cluster_idx: int, remove: int) -> MoveRecord:
        err = self.validate_move(cluster_idx, remove)
        if err:
            raise ValueError(err)
        record = self._apply(cluster_idx, remove, "oncologist")
        if not self.done:
            self.turn = "tumor"
        return record

    def tumor_move(self, strategy: str = "optimal") -> MoveRecord:
        if self.turn != "tumor":
            raise ValueError("Not the tumor's turn.")
        if self.done:
            raise ValueError("Game is over.")
        from ai.optimal import compute_optimal_move
        from ai.resistance import stochastic_resistance_move
        if strategy == "optimal":
            move = compute_optimal_move(self.heaps)
        elif strategy == "stochastic":
            move = stochastic_resistance_move(self.heaps)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        record = self._apply(move[0], move[1], "tumor")
        if not self.done:
            self.turn = "oncologist"
        return record

    def hint(self) -> str:
        """Compute and explain the optimal oncologist move."""
        from ai.optimal import compute_optimal_move
        move = compute_optimal_move(self.heaps)
        if move is None:
            return (
                "P-position: no winning move exists under optimal tumor resistance. "
                "Any treatment temporarily reduces burden but the tumor can always restore "
                "nim-sum = 0 or the pairing invariant. Consider combination therapy "
                "or targeting the pairing structure directly."
            )
        idx, remove = move
        ns_after = nim_sum([h - remove if i == idx else h for i, h in enumerate(self.heaps)])
        return (
            f"Optimal treatment: target C{idx+1}, remove {remove}×10³ cells "
            f"({self.heaps[idx]}→{self.heaps[idx]-remove}×10³). "
            f"Nim-sum becomes {ns_after}. "
            f"This {'zeros the nim-sum, entering a P-position for the tumor.' if ns_after == 0 else 'exploits the current N-position window.'}"
        )

    # ── Session summary ───────────────────────────────────────────────────────

    def session_summary(self) -> dict:
        onco_moves  = [r for r in self.history if r.actor == "oncologist"]
        tumor_moves = [r for r in self.history if r.actor == "tumor"]
        cells_removed_onco  = sum(r.cells_removed for r in onco_moves)
        cells_removed_tumor = sum(r.cells_removed for r in tumor_moves)
        p_positions = sum(1 for r in self.history if r.position_after == "P")
        n_positions = sum(1 for r in self.history if r.position_after == "N")
        return {
            "patient_id":            self.patient_id,
            "winner":                self.winner,
            "outcome":               "remission" if self.winner == "oncologist" else "failure",
            "total_cycles":          self.move_num,
            "oncologist_moves":      len(onco_moves),
            "tumor_moves":           len(tumor_moves),
            "cells_removed_by_tx":   cells_removed_onco,
            "cells_shed_by_tumor":   cells_removed_tumor,
            "initial_burden":        total_cells(self.initial_heaps),
            "final_burden":          total_cells(self.heaps),
            "burden_reduction_pct":  round(
                cells_removed_onco / max(total_cells(self.initial_heaps), 1) * 100, 1
            ),
            "p_position_cycles":     p_positions,
            "n_position_cycles":     n_positions,
            "regime_history":        list({r.regime for r in self.history}),
            "duration_seconds":      round(time.time() - self._start_time, 2),
        }
    