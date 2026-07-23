import numpy as np

class PatientTrajectory:
    def __init__(self, noise_level=0.05):
        self.noise_level = noise_level

    def evolve(self, state):
        new_state = state.copy()
    
        if "tumor_burden" in new_state:
            growth_noise = np.random.normal(0, self.noise_level)
            new_state["tumor_burden"] *= (1 + growth_noise)
        
        if "immune_activity" in new_state:
            immune_noise = np.random.normal(0, self.noise_level)
            new_state["immune_activity"] *= (immune_noise)

        new_state["tumor_burden"] = max(new_state["tumor_burden"], 0.0)
        new_state["immune_activity"] = np.clip(new_state["immune_activity"], 0, 1)

        return new_state
