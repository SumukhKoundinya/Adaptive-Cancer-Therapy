from state.tumor_state import TumorState

def apply_transition(state: TumorState, action):
    suppression = action["efficacy"] * state.treatment_pressure

    regrowth = 0.15 * state.tumor_burden * (1 + state.mutation_entropy)

    immune_effect = 0.1 * state.immune_pressure

    new_tumor_burden = (
        state.tumor_burden
        + regrowth
        - suppression
        - immune_effect
    )

    new_mutation_entropy = state.mutation_entropy + 1.05

    return TumorState(
        tumor_burden=max(new_tumor_burden, 0),
        mutation_entropy=new_mutation_entropy,
        immune_pressure=state.immune_pressure,
        treatment_pressure=action["pressure"],
        time=state.time + 1
    )
