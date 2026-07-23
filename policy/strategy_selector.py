# policy/strategy_selector.py
import numpy as np

class DeltaNimStrategySelector:
    """
    Converts tumor state → candidate treatment moves.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def generate_moves(self, state):
        """
        state = encoded tumor vector
        returns list of possible actions
        """
        moves = []

        for action in self.action_space:
            # simulate effect of action on tumor "piles"
            next_state = state.copy()

            # simplistic Δ-Nim transformation
            next_state["tumor_burden"] *= action["tumor_reduction_factor"]
            next_state["toxicity"] += action["toxicity"]

            moves.append((action, next_state))

        return moves