def compute_reward(prev_state, action, next_state):
    tumor_reduction = prev_state.tumor_burden - next_state.tumor_burden

    survival_bonus = 1.0 if next_state.tumor_burden < 20 else 0.0

    toxicity_penalty = action["pressure"] * 0.5

    reward = (
        2.0 * tumor_reduction
        + survival_bonus
        - toxicity_penalty
    )

    return reward
