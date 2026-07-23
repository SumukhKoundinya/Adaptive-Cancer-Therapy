"""
visualization/board.py
======================
ASCII visualization of Delta-Nim game state in clinical format.
Renders the tumor cluster map, nim-sum decomposition, and
treatment history for terminal/demo output.
"""

from __future__ import annotations
from engine.core import (
    nim_sum, regime, support_size, count_equal_pairs,
    is_p_position, total_cells, bit_table,
    DENSE_CUTOFF, MIN_PAIRS
)


def render_cluster_map(heaps: list[int], selected: int = None,
                       max_bar_width: int = 30) -> str:
    """
    Render tumor clusters as horizontal bar chart.
    Selected cluster is highlighted with >.
    """
    if not any(h > 0 for h in heaps):
        return "  [All clusters eradicated — complete remission]\n"

    max_h = max(heaps) or 1
    lines = ["  Tumor Cluster Map", "  " + "─" * 50]
    for i, h in enumerate(heaps):
        bar_len  = int(h / max_h * max_bar_width)
        bar      = "█" * bar_len + "░" * (max_bar_width - bar_len)
        marker   = ">" if i == selected else " "
        dead     = "[ERADICATED]" if h == 0 else f"{h:3d}×10³ cells"
        res_tag  = " [resistant]" if h > 12 else " [sensitive]" if h <= 5 else ""
        lines.append(f"  {marker} C{i+1:02d} │{bar}│ {dead}{res_tag}")

    lines.append("  " + "─" * 50)
    lines.append(f"  Total burden: {total_cells(heaps)}×10³ cells  |  Active clusters: {support_size(heaps)}")
    return "\n".join(lines)


def render_nim_decomposition(heaps: list[int]) -> str:
    """
    Render nim-sum XOR decomposition as bit table.
    Shows exactly why nim-sum is what it is.
    """
    active = [(i, h) for i, h in enumerate(heaps) if h > 0]
    if not active:
        return "  [No active clusters]\n"

    ns = nim_sum(heaps)
    max_bits = max(h.bit_length() for _, h in active)
    max_bits = max(max_bits, ns.bit_length(), 1)

    header   = "  Cluster   Value  │ " + "  ".join([f"2^{b}" for b in range(max_bits-1, -1, -1)])
    sep      = "  " + "─" * (len(header) - 2)
    lines    = ["\n  Nim-sum XOR Decomposition", sep, header, sep]

    for i, h in active[:8]:  # show up to 8 clusters
        bits = format(h, f'0{max_bits}b')
        bit_str = "   ".join(bits)
        lines.append(f"  C{i+1:02d}     {h:5d}  │ {bit_str}")

    if len(active) > 8:
        lines.append(f"  ... ({len(active) - 8} more clusters)")

    sep2 = "  " + "=" * (len(header) - 2)
    lines.append(sep2)
    ns_bits = format(ns, f'0{max_bits}b')
    ns_str  = "   ".join(ns_bits)
    lines.append(f"  XOR ⊕   {ns:5d}  │ {ns_str}  {'← nim-sum = 0 (P-position)' if ns == 0 else f'← nim-sum = {ns} (N-position)'}")
    lines.append(sep)
    return "\n".join(lines)


def render_theory_panel(heaps: list[int]) -> str:
    """
    Render full theoretical analysis panel.
    """
    r     = regime(heaps)
    sz    = support_size(heaps)
    ns    = nim_sum(heaps)
    pairs = count_equal_pairs(heaps)
    is_p  = is_p_position(heaps)
    pt    = "P" if is_p else "N"

    lines = [
        "\n  Δ-Nim Theoretical Analysis",
        "  " + "─" * 50,
        f"  Support size  |supp(h)| = {sz}",
        f"  Dense cutoff             = {DENSE_CUTOFF}",
        f"  Regime                   = {r.upper()}",
        f"  Nim-sum ⊕                = {ns}",
        f"  Equal pairs              = {pairs} / {MIN_PAIRS} needed",
        f"  Position type            = {pt}-position",
        "  " + "─" * 50,
    ]

    if r == "sparse":
        lines.append(f"  Theory: Bouton's Theorem (1901)")
        lines.append(f"    P-position iff nim-sum = 0.")
        if is_p:
            lines.append(f"    nim-sum = {ns} = 0 → P-position confirmed.")
            lines.append(f"    Any move breaks invariant. Oncologist is losing.")
        else:
            lines.append(f"    nim-sum = {ns} ≠ 0 → N-position.")
            lines.append(f"    Optimal treatment: reduce nim-sum to 0.")
    else:
        lines.append(f"  Theory: Pairing Invariant (Koundinya)")
        lines.append(f"    P-position iff {pairs} >= {MIN_PAIRS} equal pairs.")
        lines.append(f"    Pigeonhole: |supp| = {sz} > {DENSE_CUTOFF}")
        lines.append(f"    guarantees Player II can always find {MIN_PAIRS} pairs.")
        if is_p:
            lines.append(f"    {pairs} >= {MIN_PAIRS} → P-position. Tumor holds invariant.")
        else:
            lines.append(f"    {pairs} < {MIN_PAIRS} → N-position. Pairing broken.")
            lines.append(f"    Target: restore {MIN_PAIRS - pairs} more equal pairs.")

    lines.append(f"\n  Clinical interpretation:")
    if is_p:
        lines.append(f"    Tumor holds game-theoretic advantage.")
        lines.append(f"    Standard treatment will not achieve eradication.")
        lines.append(f"    Consider: adaptive scheduling, combination therapy,")
        lines.append(f"    or targeting the pairing structure directly.")
    else:
        lines.append(f"    Treatment window is open (N-position).")
        lines.append(f"    Optimal Δ-Nim dosing can guarantee eradication.")
        lines.append(f"    Apply hint to find the winning treatment move.")

    lines.append("  " + "─" * 50)
    return "\n".join(lines)


def render_treatment_log(history: list, last_n: int = 8) -> str:
    """Render the last N moves from the treatment log."""
    if not history:
        return "  [No treatment history]\n"
    lines = ["\n  Treatment Log (most recent first)", "  " + "─" * 50]
    for rec in reversed(history[-last_n:]):
        lines.append(f"  {rec.clinical_description()}")
    lines.append("  " + "─" * 50)
    return "\n".join(lines)


def render_full_board(game, selected: int = None) -> str:
    """Render complete game board."""
    st   = game.status()
    sep  = "\n" + "=" * 65 + "\n"
    parts = [
        sep,
        f"  PATIENT: {game.patient_id}  |  TURN: {st['turn'].upper()}  |  "
        f"CYCLE: {st['move_count']}  |  "
        f"{'GAME OVER — ' + (st['winner'] or '').upper() + ' WINS' if st['done'] else 'IN PROGRESS'}",
        render_cluster_map(game.heaps, selected=selected),
        render_nim_decomposition(game.heaps),
        render_theory_panel(game.heaps),
        render_treatment_log(game.history),
        sep,
    ]
    return "\n".join(parts)


def render_comparison_table(comparison: dict) -> str:
    """Render strategy comparison table."""
    lines = [
        "\n  Strategy Comparison",
        "  " + "─" * 65,
        f"  {'Strategy':<20} {'Remission%':>12} {'Mean Cycles':>12} {'Burden Δ%':>12}",
        "  " + "─" * 65,
    ]
    for strat, stats in comparison.items():
        label = {
            "optimal_nim":    "Δ-Nim Optimal",
            "max_dose":       "Max Dose (SoC)",
            "random":         "Random Dosing",
        }.get(strat, strat)
        lines.append(
            f"  {label:<20} "
            f"{stats['remission_rate']*100:>11.1f}% "
            f"{stats['mean_cycles']:>12.1f} "
            f"{stats['mean_burden_reduction']*100:>11.1f}%"
        )
    lines.append("  " + "─" * 65)
    return "\n".join(lines)


def render_survival_summary(km_results: list[dict]) -> str:
    """Render Kaplan-Meier summary table."""
    lines = [
        "\n  Kaplan-Meier Survival Summary",
        "  " + "─" * 65,
        f"  {'Arm':<22} {'Median (wk)':>12} {'S(26wk)':>10} {'S(52wk)':>10} {'N':>6}",
        "  " + "─" * 65,
    ]
    arm_labels = {
        "optimal_nim":      "Δ-Nim Adaptive",
        "standard_of_care": "Standard of Care",
        "no_treatment":     "No Treatment",
    }
    for res in km_results:
        med = res.get("median_survival")
        med_str = f"{med:.1f}" if med is not None else ">104"
        lines.append(
            f"  {arm_labels.get(res['arm'], res['arm']):<22} "
            f"{med_str:>12} "
            f"{res['surv_at_26wk']:>10.3f} "
            f"{res['surv_at_52wk']:>10.3f} "
            f"{res['n_patients']:>6}"
        )
    lines.append("  " + "─" * 65)
    return "\n".join(lines)