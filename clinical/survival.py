"""
clinical/survival.py
====================
Kaplan-Meier survival analysis for Delta-Nim adaptive therapy outcomes.

Computes and compares survival curves for:
  1. Optimal Delta-Nim therapy  (oncologist plays optimal moves)
  2. Standard-of-care therapy   (oncologist applies maximum tolerated dose)
  3. No treatment               (baseline)

Survival time is modeled as a function of:
  - Initial position type (P vs N)
  - Tumor burden reduction trajectory
  - Number of treatment cycles to eradication
  - Mean resistance level

Reference:
  Zhang et al. (2022) used Kaplan-Meier curves to show adaptive therapy
  patients had 33 months median time to progression vs 14.3 months standard.
  NCT02415621 clinical trial data.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import random
from typing import Optional


# ── Survival event record ─────────────────────────────────────────────────────

@dataclass
class SurvivalEvent:
    patient_id:     str
    time_weeks:     float       # time to event or last follow-up
    event_occurred: bool        # True = progression/death, False = censored
    therapy_arm:    str         # 'optimal_nim', 'standard_of_care', 'no_treatment'
    initial_burden: int
    initial_position: str       # 'P' or 'N'
    mean_resistance: float
    n_clusters:     int


# ── Kaplan-Meier estimator ────────────────────────────────────────────────────

class KaplanMeierEstimator:
    """
    Non-parametric Kaplan-Meier survival estimator.
    Computes S(t) = prod_{t_i <= t} (1 - d_i / n_i)
    where d_i = events at time t_i, n_i = at-risk at time t_i.
    """

    def __init__(self, events: list[SurvivalEvent], arm: str):
        self.arm    = arm
        self.events = [e for e in events if e.therapy_arm == arm]
        self._compute()

    def _compute(self):
        times = sorted(set(e.time_weeks for e in self.events if e.event_occurred))
        n_total = len(self.events)
        if n_total == 0:
            self.times  = []
            self.surv   = []
            self.at_risk = []
            self.ci_lower = []
            self.ci_upper = []
            return

        S     = 1.0
        surv_vals = [1.0]
        time_vals = [0.0]
        greenwood = 0.0  # Greenwood's formula for variance
        ci_lo = [1.0]
        ci_hi = [1.0]

        for t in times:
            n_at_risk = sum(1 for e in self.events if e.time_weeks >= t)
            n_events  = sum(1 for e in self.events if e.time_weeks == t and e.event_occurred)
            if n_at_risk == 0:
                continue
            S *= (1 - n_events / n_at_risk)
            if n_at_risk > n_events:
                greenwood += n_events / (n_at_risk * (n_at_risk - n_events))
            se    = S * math.sqrt(greenwood)
            z     = 1.96
            lo    = max(0.0, S - z * se)
            hi    = min(1.0, S + z * se)
            time_vals.append(t)
            surv_vals.append(S)
            ci_lo.append(lo)
            ci_hi.append(hi)

        self.times    = time_vals
        self.surv     = surv_vals
        self.ci_lower = ci_lo
        self.ci_upper = ci_hi

    def survival_at(self, t: float) -> float:
        """Estimate S(t) at a given time."""
        if not self.times:
            return 1.0
        s = 1.0
        for ti, si in zip(self.times, self.surv):
            if ti <= t:
                s = si
        return s

    def median_survival(self) -> Optional[float]:
        """Time at which S(t) first drops to or below 0.5."""
        for t, s in zip(self.times, self.surv):
            if s <= 0.5:
                return t
        return None  # median not reached

    def summary(self) -> dict:
        med = self.median_survival()
        return {
            "arm":             self.arm,
            "n_patients":      len(self.events),
            "median_survival": med,
            "surv_at_26wk":    round(self.survival_at(26), 3),
            "surv_at_52wk":    round(self.survival_at(52), 3),
            "surv_at_104wk":   round(self.survival_at(104), 3),
            "min_time":        min((e.time_weeks for e in self.events), default=0),
            "max_time":        max((e.time_weeks for e in self.events), default=0),
        }


# ── Synthetic survival generator ──────────────────────────────────────────────

class SurvivalSimulator:
    """
    Generate synthetic Kaplan-Meier data for three therapy arms.

    Survival time model (weeks):
      T ~ Weibull(scale, shape)
      scale = base_scale * position_bonus * resistance_penalty * burden_penalty
    """

    BASE_SCALES = {
        "optimal_nim":    120.0,   # Delta-Nim strategy — best outcomes
        "standard_of_care": 62.0,  # max tolerated dose — Zhang et al. 14.3mo baseline
        "no_treatment":    24.0,   # natural progression
    }
    SHAPES = {
        "optimal_nim":     2.2,
        "standard_of_care": 1.8,
        "no_treatment":    1.4,
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _weibull_sample(self, scale: float, shape: float) -> float:
        u = self.rng.random()
        return scale * (-math.log(1 - u + 1e-9)) ** (1 / shape)

    def simulate_patient(
        self,
        patient_id:       str,
        initial_burden:   int,
        initial_position: str,
        mean_resistance:  float,
        n_clusters:       int,
        arm:              str,
        censoring_prob:   float = 0.18,
    ) -> SurvivalEvent:
        scale = self.BASE_SCALES[arm]
        shape = self.SHAPES[arm]

        # Position bonus: N-position patients have better prognosis under nim therapy
        if arm == "optimal_nim" and initial_position == "N":
            scale *= 1.45
        elif arm == "optimal_nim" and initial_position == "P":
            scale *= 0.72

        # Resistance penalty
        scale *= (1 - 0.4 * mean_resistance)

        # Burden penalty: higher initial burden → worse prognosis
        burden_factor = max(0.4, 1 - 0.003 * initial_burden)
        scale *= burden_factor

        # Cluster count penalty for dense regime
        if n_clusters > 60:
            scale *= 0.78

        t = self._weibull_sample(max(scale, 4.0), shape)
        t = max(1.0, t)
        censored = self.rng.random() < censoring_prob

        return SurvivalEvent(
            patient_id=patient_id, time_weeks=round(t, 1),
            event_occurred=(not censored), therapy_arm=arm,
            initial_burden=initial_burden, initial_position=initial_position,
            mean_resistance=mean_resistance, n_clusters=n_clusters,
        )

    def simulate_cohort(
        self,
        n_patients:     int = 100,
        tumor_type:     str = "prostate",
        arms:           list[str] = None,
    ) -> list[SurvivalEvent]:
        """Generate a full cohort of synthetic survival events across all arms."""
        from clinical.patient import Patient
        if arms is None:
            arms = ["optimal_nim", "standard_of_care", "no_treatment"]

        all_events: list[SurvivalEvent] = []
        for j in range(n_patients):
            patient = Patient.generate(tumor_type=tumor_type, seed=j)
            heaps   = patient.heaps()
            from engine.core import is_p_position
            pos = "P" if is_p_position(heaps) else "N"
            for arm in arms:
                event = self.simulate_patient(
                    patient_id=patient.patient_id + f"_{arm[:3]}",
                    initial_burden=patient.total_burden(),
                    initial_position=pos,
                    mean_resistance=patient.mean_resistance(),
                    n_clusters=len(patient.clusters),
                    arm=arm,
                )
                all_events.append(event)
        return all_events


# ── Log-rank test ─────────────────────────────────────────────────────────────

def log_rank_test(events: list[SurvivalEvent], arm1: str, arm2: str) -> dict:
    """
    Log-rank test comparing two survival curves.
    H0: no difference in survival distributions.
    Returns chi-squared statistic and p-value approximation.
    """
    def get_data(arm):
        return [(e.time_weeks, e.event_occurred) for e in events if e.therapy_arm == arm]

    d1 = get_data(arm1)
    d2 = get_data(arm2)
    all_times = sorted(set(t for t, e in d1 + d2 if e))

    O1_total = E1_total = 0.0
    V_total  = 0.0

    for t in all_times:
        n1 = sum(1 for tt, _ in d1 if tt >= t)
        n2 = sum(1 for tt, _ in d2 if tt >= t)
        d1_t = sum(1 for tt, ev in d1 if tt == t and ev)
        d2_t = sum(1 for tt, ev in d2 if tt == t and ev)
        N  = n1 + n2
        D  = d1_t + d2_t
        if N == 0:
            continue
        E1 = D * n1 / N
        E1_total += E1
        O1_total += d1_t
        if N > 1:
            V_total += D * n1 * n2 * (N - D) / (N ** 2 * (N - 1))

    if V_total == 0:
        return {"chi2": 0.0, "p_value": 1.0, "arm1": arm1, "arm2": arm2}

    chi2 = (O1_total - E1_total) ** 2 / V_total

    # Approximate p-value from chi-squared(df=1)
    p = math.exp(-chi2 / 2) * math.sqrt(chi2 * math.pi / 2) / 2
    p = max(0.0001, min(0.9999, p))

    return {
        "arm1":            arm1,
        "arm2":            arm2,
        "observed_arm1":   O1_total,
        "expected_arm1":   round(E1_total, 2),
        "chi2":            round(chi2, 4),
        "p_value":         round(p, 4),
        "significant":     p < 0.05,
        "interpretation":  (
            f"Significant difference (p={p:.4f} < 0.05) — {arm1} and {arm2} have different survival distributions."
            if p < 0.05 else
            f"No significant difference (p={p:.4f} >= 0.05)."
        ),
    }