from dataclasses import dataclass
import numpy as np

@dataclass
class PatientState:
    age: float
    mutation_count: float
    tmb: float
    kps: float
    prior_recurrences: float
    pd1_injections: float

    response: int = None
    os_months: float = None
    pfs_months: float = None

    def to_vector(self):
        return np.array([
            self.age,
            self.mutation_count,
            self.tmb,
            self.kps,
            self.prior_recurrences,
            self.pd1_injections
        ], dtype=float)
    
import pandas as pd

def build_patient_states(df: pd.DataFrame):
    """
    Converts cleaned dataframe into structured PatientState objects.
    """

    states = []

    for _, row in df.iterrows():
        state = PatientState(
            age=row.get("age_at_pd1", np.nan),
            mutation_count=row.get("mutation_count", np.nan),
            tmb=row.get("tmb", np.nan),
            kps=row.get("kps", np.nan),
            prior_recurrences=row.get("number_of_prior_recurrences", np.nan),
            pd1_injections=row.get("number_of_pd1_inhibitor_injections", np.nan),

            response=row.get("response_encoded", None),
            os_months=row.get("os_months", None),
            pfs_months=row.get("pfs_months", None)
        )

        states.append(state)

    return states
