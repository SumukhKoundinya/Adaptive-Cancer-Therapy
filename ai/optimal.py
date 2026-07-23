"""
ai/optimal.py
=============
Optimal move computation for Delta-Nim in both sparse and dense regimes.

Sparse regime: classic nim-sum strategy (Bouton 1901).
  - Find heap h_i such that h_i XOR nim_sum < h_i.
  - Remove (h_i - (h_i XOR nim_sum)) cells from that heap.
  - Result: nim-sum = 0, entering P-position for opponent.

Dense regime: pairing invariant strategy.
  - Find a move that restores >= 30 equal-heap pairs.
  - Prioritize moves that create new pairs by matching unpaired heaps.
  - Fallback: reduce the largest unpaired heap to match its nearest neighbor.
"""

from __future__ import annotations
from typing import Optional
import random

from engine.core import (
    nim_sum, regime, count_equal_pairs, is_p_position,
    support_size, heap_value_counts, unpaired_heaps, regime_with_metadata,
    DENSE_CUTOFF, MIN_PAIRS, Regime
)

def state_energy(heaps):
    return sum(h * h for h in heaps)

# ── Sparse regime ─────────────────────────────────────────────────────────────

def sparse_winning_move(heaps: list[int]) -> Optional[tuple[int, int]]:
    """
    Returns (cluster_idx, cells_to_remove) that zeros the nim-sum.
    Returns None if already a P-position (no winning move).
    """
    ns = nim_sum(heaps)
    if ns == 0:
        return None  # Already P-position
    for i, h in enumerate(heaps):
        target = h ^ ns
        if target < h:
            return (i, h - target)
    return None  # Should not reach here if nim-sum != 0


def sparse_any_move(heaps: list[int]) -> tuple[int, int]:
    """Fallback: make any legal move (used when in P-position, forced suboptimal)."""
    nonzero = [(i, h) for i, h in enumerate(heaps) if h > 0]
    i, h = nonzero[0]
    return (i, 1)


# ── Dense regime ──────────────────────────────────────────────────────────────

def dense_winning_move(heaps: list[int]) -> Optional[tuple[int, int]]:
    """
    Returns (cluster_idx, cells_to_remove) that restores pairing invariant.
    Returns None if already P-position (pairing invariant already holds).

    Strategy (pairing invariant):
    Player II needs >= 30 equal-heap pairs after every move.
    We search for a move that creates a new matching pair:
      1. Find an unpaired heap h_i.
      2. Find another heap h_j != h_i that appears an odd number of times.
      3. Remove (h_i - h_j) from h_i to match h_j, creating a new pair.
    """
    """if count_equal_pairs(heaps) >= MIN_PAIRS:
        return None  # Already P-position, no winning move"""
    PAIRING_THRESHOLD = max(5, len(heaps) * 2)

    counts = heap_value_counts(heaps)

    # Strategy 1: reduce an unpaired heap to match another unpaired heap
    unp = unpaired_heaps(heaps)
    for i, h in enumerate(heaps):
        if h == 0 or counts.get(h, 0) % 2 == 0:
            continue  # skip paired heaps
        for target_val in sorted(set(unp)):
            if target_val < h and target_val != h:
                test = heaps[:]
                test[i] = target_val
                if count_equal_pairs(test) >= MIN_PAIRS:
                    return (i, h - target_val)

    # Strategy 2: eliminate an unpaired heap entirely to restore pairing
    for i, h in enumerate(heaps):
        if h == 0:
            continue
        if counts.get(h, 0) % 2 == 1:
            test = heaps[:]
            test[i] = 0
            if count_equal_pairs(test) >= MIN_PAIRS:
                return (i, h)

    # Strategy 3: brute-force search over all moves (expensive but correct)
    nonzero = [(i, h) for i, h in enumerate(heaps) if h > 0]
    for i, h in nonzero:
        for remove in range(1, h + 1):
            test = heaps[:]
            test[i] -= remove
            if count_equal_pairs(test) >= MIN_PAIRS:
                return (i, remove)

    # No pairing-restoring move found — suboptimal fallback
    i, h = min(nonzero, key=lambda x: x[1])
    return (i, 1)


def dense_p_position_move(heaps: list[int]) -> tuple[int, int]:
    """
    When already in P-position (dense), make any move (forced suboptimal).
    Pick smallest heap, remove 1 — minimizes disruption to pairing structure.
    """
    nonzero = [(i, h) for i, h in enumerate(heaps) if h > 0]
    i, h = min(nonzero, key=lambda x: x[1])
    return (i, 1)


# ── Unified optimal move ──────────────────────────────────────────────────────

def compute_optimal_move(heaps: list[int]) -> Optional[tuple[int, int]]:
    """
    Compute the optimal (cluster_idx, cells_to_remove) for the current player.
    Returns None if this is already a P-position (current player is losing).

    This is the core AI used by the tumor in CancerNimGame.tumor_move().
    It is also used by the oncologist hint system.
    """
    if all(h == 0 for h in heaps):
        return None

    #r = regime(heaps)
    meta = regime_with_metadata(heaps)
    r = meta["regime"]
    if r == Regime.SPARSE:
        return sparse_winning_move(heaps)
    else:
        return dense_winning_move(heaps)


def compute_move_quality(heaps: list[int], cluster_idx: int, remove: int) -> dict:
    """
    Score a proposed move on a scale of 0-100 and explain why.
    Used by the clinical decision support system.
    """
    after = heaps[:]
    after[cluster_idx] -= remove

    optimal = compute_optimal_move(heaps)
    is_optimal = (optimal is not None and optimal == (cluster_idx, remove))

    ns_before  = nim_sum(heaps)
    ns_after   = nim_sum(after)
    pairs_before = count_equal_pairs(heaps)
    pairs_after  = count_equal_pairs(after)
    enters_p    = is_p_position(after)

    #r = regime(heaps)
    meta = regime_with_metadata(heaps)
    r = meta["regime"]
    if r == Regime.SPARSE:
        if ns_after == 0:
            score = 100
            rationale = f"Zeros the nim-sum ({ns_before}→0). Creates P-position for tumor — guaranteed win with continued optimal play."
        elif ns_after < ns_before:
            score = 60
            rationale = f"Reduces nim-sum ({ns_before}→{ns_after}) but doesn't zero it. Suboptimal — tumor can respond to maintain advantage."
        else:
            score = 20
            rationale = f"Increases nim-sum ({ns_before}→{ns_after}). Weakens position. Avoid."
    else:
        if pairs_after >= MIN_PAIRS:
            score = 100
            rationale = f"Restores pairing invariant ({pairs_after} pairs ≥ {MIN_PAIRS}). Dense P-position achieved."
        elif pairs_after > pairs_before:
            score = 65
            rationale = f"Increases equal pairs ({pairs_before}→{pairs_after}) but not to threshold. Partial progress."
        else:
            score = 25
            rationale = f"Reduces equal pairs ({pairs_before}→{pairs_after}). Weakens pairing structure."

    return {
        "move":          (cluster_idx, remove),
        "score":         score,
        "is_optimal":    is_optimal,
        "nim_sum_delta": ns_after - ns_before,
        "pairs_delta":   pairs_after - pairs_before,
        "enters_p":      enters_p,
        "state_energy_delta": state_energy(after) - state_energy(heaps),
        "rationale":     rationale,
        "regime":        r,
    }

def ranked_moves(heaps: list[int], top_k: int = 5) -> list[dict]:
    """
    Return top_k moves ranked by quality score.
    Used by the clinical decision support panel.
    """
    moves = []
    for i, h in enumerate(heaps):
        for remove in range(1, h + 1):
            quality = compute_move_quality(heaps, i, remove)
            quality["cluster"]  = i
            quality["remove"]   = remove
            quality["heap_before"] = h
            quality["heap_after"]  = h - remove
            moves.append(quality)
    moves.sort(key=lambda m: m["score"], reverse=True)
    return moves[:top_k]
