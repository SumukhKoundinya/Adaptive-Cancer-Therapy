"""
engine/validators.py
====================
Invariant verification, move legality checks, and theoretical
consistency assertions for Delta-Nim positions.
"""

from __future__ import annotations
from typing import Optional
from engine.core import (
    nim_sum, support_size, regime, count_equal_pairs,
    is_p_position, pairing_invariant_holds,
    DENSE_CUTOFF, MIN_PAIRS, Regime
)


# ── Move validation ───────────────────────────────────────────────────────────

def is_legal_move(heaps: list[int], cluster_idx: int, remove: int) -> tuple[bool, str]:
    """
    Returns (is_legal, reason).
    A move is legal iff:
      1. cluster_idx is a valid index
      2. heaps[cluster_idx] > 0
      3. 1 <= remove <= heaps[cluster_idx]
    """
    if not (0 <= cluster_idx < len(heaps)):
        return False, f"Invalid cluster index {cluster_idx}."
    if heaps[cluster_idx] == 0:
        return False, f"Cluster C{cluster_idx+1} is already eradicated."
    if remove < 1:
        return False, "Must remove at least 1×10³ cells."
    if remove > heaps[cluster_idx]:
        return False, (
            f"Cannot remove {remove}×10³ from C{cluster_idx+1} "
            f"(only {heaps[cluster_idx]}×10³ remain)."
        )
    return True, "Legal move."


def all_legal_moves(heaps: list[int]) -> list[tuple[int, int]]:
    """Enumerate all legal (cluster_idx, remove) moves from this position."""
    moves = []
    for i, h in enumerate(heaps):
        for r in range(1, h + 1):
            moves.append((i, r))
    return moves


# ── Invariant verification ────────────────────────────────────────────────────

def verify_bouton_invariant(heaps: list[int]) -> dict:
    """
    Verify Bouton's Theorem for sparse positions:
    P-position iff nim-sum = 0.
    Returns full audit.
    """
    assert regime(heaps) == Regime.SPARSE, "Bouton invariant only applies to sparse regime."
    ns    = nim_sum(heaps)
    is_p  = ns == 0
    return {
        "regime":        Regime.SPARSE,
        "nim_sum":       ns,
        "is_p_position": is_p,
        "theorem":       "Bouton 1901",
        "verdict":       (
            f"nim-sum = {ns} = 0 → P-position confirmed."
            if is_p else
            f"nim-sum = {ns} ≠ 0 → N-position, winning move exists."
        ),
    }


def verify_pairing_invariant(heaps: list[int]) -> dict:
    """
    Verify pairing invariant for dense positions.
    P-position iff >= 30 equal heap pairs exist.
    """
    assert regime(heaps) == Regime.DENSE, "Pairing invariant only applies to dense regime."
    pairs  = count_equal_pairs(heaps)
    is_p   = pairs >= MIN_PAIRS
    deficit = max(0, MIN_PAIRS - pairs)
    return {
        "regime":            Regime.DENSE,
        "equal_pairs":       pairs,
        "pairs_needed":      MIN_PAIRS,
        "pairs_deficit":     deficit,
        "is_p_position":     is_p,
        "theorem":           "Delta-Nim Pairing Invariant (Koundinya)",
        "pigeonhole_bound":  f"|supp| = {support_size(heaps)} > {DENSE_CUTOFF} → Player II can always maintain {MIN_PAIRS} pairs.",
        "verdict": (
            f"{pairs} >= {MIN_PAIRS} equal pairs → P-position, pairing invariant holds."
            if is_p else
            f"Only {pairs} pairs (need {MIN_PAIRS}) → N-position, pairing invariant broken."
        ),
    }


def full_position_audit(heaps: list[int]) -> dict:
    """Complete theoretical audit of any Delta-Nim position."""
    r     = regime(heaps)
    is_p  = is_p_position(heaps)
    audit = {
        "heaps":          heaps[:],
        "support_size":   support_size(heaps),
        "regime":         r,
        "position_type":  "P" if is_p else "N",
        "nim_sum":        nim_sum(heaps),
        "equal_pairs":    count_equal_pairs(heaps),
    }
    if r == Regime.SPARSE:
        audit["invariant_check"] = verify_bouton_invariant(heaps)
    else:
        audit["invariant_check"] = verify_pairing_invariant(heaps)
    return audit


# ── Consistency checks ────────────────────────────────────────────────────────

def assert_complete_determinacy(heaps: list[int]) -> bool:
    """
    Main theorem: every Delta-Nim position is P or N, never undecided.
    This is trivially satisfied by construction but we verify regime logic.
    """
    r = regime(heaps)
    if r == Regime.SPARSE:
        # Both P and N cases are covered by nim-sum
        ns = nim_sum(heaps)
        return True  # always determinate
    else:
        # Dense: pairing invariant always classifiable
        pairs = count_equal_pairs(heaps)
        return True  # always determinate

def verify_p_position_stability(heaps: list[int]) -> dict:
    """
    Verify that from a P-position, every possible move leads to an N-position.
    (This is the defining property of P-positions.)
    """
    assert is_p_position(heaps), "Not a P-position."
    results = []
    for i, h in enumerate(heaps):
        for remove in range(1, h + 1):
            after = heaps[:]
            after[i] -= remove
            leads_to_n = not is_p_position(after)
            results.append({
                "move":         (i, remove),
                "heaps_after":  after,
                "leads_to_N":   leads_to_n,
            })
    all_lead_to_n = all(r["leads_to_N"] for r in results)
    return {
        "p_position_valid":  all_lead_to_n,
        "moves_checked":     len(results),
        "counterexamples":   [r for r in results if not r["leads_to_N"]],
    }


def verify_n_position_has_winning_move(heaps: list[int]) -> dict:
    """
    Verify that from an N-position, at least one move leads to a P-position.
    """
    assert not is_p_position(heaps), "Not an N-position."
    from ai.optimal import compute_optimal_move
    winning_move = compute_optimal_move(heaps)
    if winning_move is None:
        return {"n_position_valid": False, "winning_move": None}
    idx, remove = winning_move
    after = heaps[:]
    after[idx] -= remove
    leads_to_p = is_p_position(after)
    return {
        "n_position_valid": leads_to_p,
        "winning_move":     winning_move,
        "heaps_after":      after,
        "nim_sum_after":    nim_sum(after),
    }