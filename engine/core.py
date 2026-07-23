"""
engine/core.py
==============
Core mathematical primitives for Delta-Nim.

Biological mapping:
  heap          -> tumor cell cluster (spatially distinct subpopulation)
  heap size     -> number of cells (x10^3) in that cluster
  support size  -> number of active (nonzero) clusters = |supp(h)|
  move          -> treatment dose applied to one cluster in one cycle
  nim-sum = 0   -> drug-resistant equilibrium (P-position, tumor wins)
  nim-sum != 0  -> treatment window open (N-position, oncologist wins)

Regime classification (from paper):
  Sparse : |supp(h)| <= 60  -> Bouton's Theorem applies
  Dense  : |supp(h)| > 60   -> Pairing invariant applies
  Cutoff 60 derived via Pigeonhole Principle over 30 disjoint pairs.
"""

from __future__ import annotations
from functools import reduce
from typing import Optional
import math
import statistics

DENSE_CUTOFF: int = 60
MIN_PAIRS:    int = 30


# ── Heap primitives ──────────────────────────────────────────────────────────

def nim_sum(heaps: list[int]) -> int:
    """XOR of all heap sizes. Central invariant for sparse P-positions."""
    return reduce(lambda a, b: a ^ b, heaps, 0)


def nim_sum_after(heaps: list[int], idx: int, remove: int) -> int:
    """Nim-sum if we remove `remove` cells from heap at `idx`."""
    modified = heaps[:]
    modified[idx] -= remove
    return nim_sum(modified)


def support(heaps: list[int]) -> list[int]:
    """Nonzero heap values (support of the position)."""
    return [h for h in heaps if h > 0]


def support_indices(heaps: list[int]) -> list[int]:
    return [i for i, h in enumerate(heaps) if h > 0]


def support_size(heaps: list[int]) -> int:
    return sum(1 for h in heaps if h > 0)


def total_cells(heaps: list[int]) -> int:
    return sum(heaps)


def max_heap(heaps: list[int]) -> int:
    s = support(heaps)
    return max(s) if s else 0


def min_heap_nonzero(heaps: list[int]) -> int:
    s = support(heaps)
    return min(s) if s else 0


def heap_entropy(heaps: list[int]) -> float:
    """Shannon entropy of the heap size distribution."""
    s = support(heaps)
    if not s:
        return 0.0
    total = sum(s)
    probs = [h / total for h in s]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def heap_std(heaps: list[int]) -> float:
    s = support(heaps)
    return statistics.stdev(s) if len(s) > 1 else 0.0


# ── Regime ───────────────────────────────────────────────────────────────────

class Regime:
    SPARSE = "sparse"
    DENSE  = "dense"


def regime(heaps: list[int]) -> str:
    """
    Classify by |supp(h)|.
    Cutoff = 60 via Pigeonhole: with 61+ distinct nonzero heaps Player II
    can always find 30 disjoint equal-size pairs to maintain nim-sum = 0.
    """
    return Regime.SPARSE if support_size(heaps) <= DENSE_CUTOFF else Regime.DENSE

def regime_with_metadata(heaps):
    return {
        "regime": regime(heaps),
        "entropy": len(set(heaps)) / len(heaps),
        "support": support_size(heaps)
    }

# ── Pairing invariant ─────────────────────────────────────────────────────────

def heap_value_counts(heaps: list[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for h in heaps:
        if h > 0:
            counts[h] = counts.get(h, 0) + 1
    return counts


def count_equal_pairs(heaps: list[int]) -> int:
    """
    Max disjoint equal-heap pairs.
    Each pair contributes XOR = 0, so nim-sum cancels pairwise.
    Player II needs >= MIN_PAIRS (30) to hold the invariant.
    """
    counts = heap_value_counts(heaps)
    return sum(c // 2 for c in counts.values())


def pairing_invariant_holds(heaps: list[int]) -> bool:
    return count_equal_pairs(heaps) >= MIN_PAIRS


def unpaired_heaps(heaps: list[int]) -> list[int]:
    counts = heap_value_counts(heaps)
    result = []
    for val, cnt in counts.items():
        if cnt % 2 == 1:
            result.append(val)
    return sorted(result)


def pairs_deficit(heaps: list[int]) -> int:
    """How many additional pairs needed to satisfy invariant."""
    return max(0, MIN_PAIRS - count_equal_pairs(heaps))


# ── P/N classification ────────────────────────────────────────────────────────

def is_p_position(heaps: list[int]) -> bool:
    """
    P-position = current player loses under optimal play.
    Terminal (all zero) is P by convention (last mover won).
    Sparse: nim-sum == 0 (Bouton 1901).
    Dense:  pairing invariant holds (>= 30 equal pairs).
    """
    if all(h == 0 for h in heaps):
        return True
    if regime(heaps) == Regime.SPARSE:
        return nim_sum(heaps) == 0
    return pairing_invariant_holds(heaps)


def is_n_position(heaps: list[int]) -> bool:
    return not is_p_position(heaps)


def position_type(heaps: list[int]) -> str:
    return "P" if is_p_position(heaps) else "N"


# ── Bit decomposition (for nim-sum visualization) ─────────────────────────────

def bit_table(heaps: list[int]) -> list[tuple[int, int, str]]:
    """
    Returns (heap_idx, heap_val, binary_string) for nonzero heaps.
    Used to show why nim-sum is what it is.
    """
    s = support(heaps)
    max_bits = max((h.bit_length() for h in s), default=1)
    return [
        (i, h, format(h, f'0{max_bits}b'))
        for i, h in enumerate(heaps) if h > 0
    ]


# ── Feature vector (matches paper's 12-feature ML set) ───────────────────────

def feature_vector(heaps: list[int]) -> list[float]:
    """
    12 features used by the Random Forest classifier (paper Section 5).
      0  support_size
      1  nim_sum
      2  total_cells
      3  max_heap
      4  min_heap_nonzero
      5  equal_pairs
      6  pairing_invariant  (0/1)
      7  regime             (0=sparse, 1=dense)
      8  nim_sum_bits
      9  even_fraction
     10  xor_distance       (sum |h XOR nim_sum| — proxy for distance to P)
     11  heap_std
    """
    s = support(heaps)
    if not s:
        return [0.0] * 12
    ns    = nim_sum(heaps)
    sz    = len(s)
    pairs = count_equal_pairs(heaps)
    r     = 1.0 if sz > DENSE_CUTOFF else 0.0
    even_f = sum(1 for h in s if h % 2 == 0) / sz
    xor_d  = sum(abs(h ^ ns) for h in s)
    std    = heap_std(heaps)
    return [
        float(sz), float(ns), float(total_cells(heaps)),
        float(max_heap(heaps)), float(min_heap_nonzero(heaps)),
        float(pairs), float(int(pairs >= MIN_PAIRS)), r,
        float(ns.bit_length()), even_f, float(xor_d), std,
    ]


FEATURE_NAMES = [
    "support_size", "nim_sum", "total_cells", "max_heap",
    "min_heap_nonzero", "equal_pairs", "pairing_invariant",
    "regime", "nim_sum_bits", "even_fraction", "xor_distance", "heap_std",
]