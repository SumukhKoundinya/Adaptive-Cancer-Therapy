from state.tumor_state import TumorState
from game.transition_rules import apply_transition
from game.payoff_function import compute_reward

class DeltaNimGame:
    def __init__(self, initial_state: TumorState):
        self.state = initial_state
        self.history = []

    def step(self, action):
        next_state = apply_transition(self.state, action)

        reward = compute_reward(self.state, action, next_state)

        self.history.append((self.state, action, reward, next_state))

        self.state = next_state

        return next_state, reward
    
    def is_terminal(self):
        return self.state.is_terminal()
