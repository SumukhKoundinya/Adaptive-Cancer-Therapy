from state.patient_state import PatientState
from state.tumor_state import TumorState
import numpy as np

def encode_to_tumor_state(patient: PatientState) -> TumorState:
    tumor_burden = (
        patient.mutation_count * 0.5 +
        patient.tmb * 0.3 +
        (100 - patient.kps) * 0.2
    )

    mutation_entropy = np.log1p(patient.mutation_count)
    immune_pressure = 1.0 if patient.response is not None else 0.5
    treatment_pressure = patient.pd1_injections * 0.1
    time = patient.os_months if patient.os_months else 0
    time = patient.os_months if patient.os_months else 0

    return TumorState(
        tumor_burden=tumor_burden,
        mutation_entropy=mutation_entropy,
        immune_pressure=immune_pressure,
        treatment_pressure=treatment_pressure,
        time=time
    )

