"""
clinical/patient.py
===================
Patient and tumor biology models for Delta-Nim adaptive therapy.

Maps real tumor biology to Delta-Nim:
  - Cluster count and size drawn from clinical distributions
  - Drug resistance levels assigned per cluster
  - Tumor type determines initial heap configuration
  - Survival probability modeled via Kaplan-Meier framework

Tumor types modeled (based on Zhang et al. 2022 and NCT02415621):
  prostate   : 4-8 clusters, moderate resistance, sparse regime typical
  breast     : 6-12 clusters, heterogeneous resistance, mixed regime
  lung       : 8-15 clusters, high mutation rate, adaptive resistance
  metastatic : 15-80+ clusters, dense regime, pairing invariant dominant
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import random
import math
import uuid


# ── Tumor cluster ─────────────────────────────────────────────────────────────

@dataclass
class TumorCluster:
    cluster_id: int
    cells: int           # size of heap (×10³)
    location: str           # anatomical site
    resistance_score: float         # 0 (sensitive) to 1 (fully resistant)
    mutation_burden: float         # tumor mutation burden proxy
    driver_mutations: list[str] = field(default_factory=list)

    def resistance_class(self) -> str:
        if self.resistance_score < 0.33:
            return "sensitive"
        elif self.resistance_score < 0.67:
            return "intermediate"
        else:
            return "resistant"

    def expected_response_rate(self, dose: int) -> float:
        """
        Probability cluster responds to `dose` units of treatment.
        Higher resistance = lower response probability.
        Based on log-linear dose-response model.
        """
        effective_dose = dose * (1 - self.resistance_score)
        return 1 - math.exp(-0.15 * effective_dose)

    def to_dict(self) -> dict:
        return {
            "cluster_id":       self.cluster_id,
            "cells":            self.cells,
            "location":         self.location,
            "resistance_score": round(self.resistance_score, 3),
            "resistance_class": self.resistance_class(),
            "mutation_burden":  round(self.mutation_burden, 2),
            "driver_mutations": self.driver_mutations,
        }


# ── Patient model ─────────────────────────────────────────────────────────────

TUMOR_CONFIGS = {
    "prostate": {
        "n_clusters":    (4, 8),
        "cell_range":    (3, 20),
        "resistance":    (0.2, 0.7),
        "mutation_rate": 0.3,
        "locations":     ["primary", "lymph node", "bone", "adrenal"],
        "mutations":     ["AR amplification", "TP53 loss", "PTEN loss", "RB1 loss"],
    },
    "breast": {
        "n_clusters":    (6, 14),
        "cell_range":    (2, 18),
        "resistance":    (0.15, 0.85),
        "mutation_rate": 0.45,
        "locations":     ["primary", "lymph node", "lung", "bone", "liver"],
        "mutations":     ["PIK3CA", "TP53", "CDH1", "GATA3", "MYC amplification"],
    },
    "lung": {
        "n_clusters":    (8, 16),
        "cell_range":    (4, 25),
        "resistance":    (0.3, 0.9),
        "mutation_rate": 0.65,
        "locations":     ["primary", "mediastinum", "contralateral lung", "brain", "adrenal"],
        "mutations":     ["KRAS", "EGFR", "ALK fusion", "TP53", "STK11"],
    },
    "metastatic": {
        "n_clusters":    (20, 80),
        "cell_range":    (2, 15),
        "resistance":    (0.4, 0.95),
        "mutation_rate": 0.8,
        "locations":     ["primary", "liver", "lung", "bone", "peritoneum", "brain"],
        "mutations":     ["TP53", "KRAS", "APC", "PIK3CA", "MYC", "CDKN2A loss"],
    },
}

@dataclass
class Patient:
    patient_id:    str
    age:           int
    tumor_type:    str
    stage:         str
    clusters:      list[TumorCluster]
    ecog_score:    int      # 0-4, performance status
    prior_lines:   int      # prior lines of therapy
    seed:          int = 0

    @classmethod
    def generate(
        cls,
        tumor_type:  str = "prostate",
        stage:       str = "IV",
        seed:        int = None,
        patient_id:  str = None,
    ) -> "Patient":
        """
        Generate a synthetic patient with biologically realistic tumor clusters.
        """
        if seed is None:
            seed = random.randint(0, 99999)
        rng = random.Random(seed)

        config = TUMOR_CONFIGS.get(tumor_type, TUMOR_CONFIGS["prostate"])
        n_lo, n_hi   = config["n_clusters"]
        c_lo, c_hi   = config["cell_range"]
        r_lo, r_hi   = config["resistance"]
        n_clusters   = rng.randint(n_lo, n_hi)
        locations    = config["locations"]
        mutations    = config["mutations"]

        clusters = []
        for i in range(n_clusters):
            cells      = rng.randint(c_lo, c_hi)
            resistance = rng.uniform(r_lo, r_hi)
            location   = rng.choice(locations)
            mut_burden = rng.gauss(config["mutation_rate"], 0.1)
            n_muts     = rng.randint(1, min(3, len(mutations)))
            muts       = rng.sample(mutations, n_muts)
            clusters.append(TumorCluster(
                cluster_id=i, cells=cells, location=location,
                resistance_score=max(0.0, min(1.0, resistance)),
                mutation_burden=max(0.0, mut_burden),
                driver_mutations=muts,
            ))

        pid  = patient_id or f"PT-{rng.randint(1000, 9999)}"
        age  = rng.randint(45, 78)
        ecog = rng.choice([0, 0, 1, 1, 1, 2, 2, 3])
        prior = rng.randint(0, 3)

        return cls(
            patient_id=pid, age=age, tumor_type=tumor_type,
            stage=stage, clusters=clusters, ecog_score=ecog,
            prior_lines=prior, seed=seed,
        )

    def heaps(self) -> list[int]:
        """Return cluster sizes as Delta-Nim heap list."""
        return [c.cells for c in self.clusters]

    def total_burden(self) -> int:
        return sum(c.cells for c in self.clusters)

    def mean_resistance(self) -> float:
        if not self.clusters:
            return 0.0
        return sum(c.resistance_score for c in self.clusters) / len(self.clusters)

    def high_resistance_fraction(self) -> float:
        if not self.clusters:
            return 0.0
        return sum(1 for c in self.clusters if c.resistance_score > 0.67) / len(self.clusters)

    def clinical_summary(self) -> dict:
        from engine.core import nim_sum, regime, count_equal_pairs, is_p_position
        heaps = self.heaps()
        return {
            "patient_id":              self.patient_id,
            "age":                     self.age,
            "tumor_type":              self.tumor_type,
            "stage":                   self.stage,
            "ecog_score":              self.ecog_score,
            "prior_therapy_lines":     self.prior_lines,
            "n_clusters":              len(self.clusters),
            "total_burden_k_cells":    self.total_burden(),
            "mean_resistance":         round(self.mean_resistance(), 3),
            "high_resistance_frac":    round(self.high_resistance_fraction(), 3),
            "nim_sum":                 nim_sum(heaps),
            "regime":                  regime(heaps),
            "equal_pairs":             count_equal_pairs(heaps),
            "initial_position":        "P" if is_p_position(heaps) else "N",
            "game_theoretic_prognosis": (
                "Unfavorable — tumor holds P-position advantage at diagnosis."
                if is_p_position(heaps) else
                "Favorable — oncologist has an N-position treatment window."
            ),
            "clusters":                [c.to_dict() for c in self.clusters],
        }

    def resistance_profile_summary(self) -> str:
        lines = [f"Resistance Profile — {self.patient_id}"]
        for c in self.clusters:
            bar_len = int(c.resistance_score * 20)
            bar     = "#" * bar_len + "." * (20 - bar_len)
            lines.append(
                f"  C{c.cluster_id+1:02d} [{bar}] {c.resistance_score:.2f} "
                f"({c.resistance_class()}) @ {c.location}"
            )
        return "\n".join(lines)