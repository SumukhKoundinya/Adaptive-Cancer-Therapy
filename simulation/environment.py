import numpy as np
from game.delta_nim import DeltaNim
from simulation.patient_trajectory import PatientTrajectory

class CancerEnvironment:
    def __init__(self):
        self.game = DeltaNim()
        self.trajectories = PatientTrajectory()
        self.state = None
        self.t = 0

    def reset(self, initial_state):
        self.state = initial_state
        self.t = 0
        self.game.reset(self.state)

        return self.state

    def step(self, action):
        next_state = self.game.transition(self.state, action)

        next_state = self.trajectory.evolve(next_state)

        reward = self.game.payoff(self.state, action, next_state)

        self.state = next_state
        self.t += 1

        done = self.state.get("os_event", False)

        return next_state, reward, done, {}
