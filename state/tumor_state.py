from dataclasses import dataclass

@dataclass
class TumorState:
    tumor_burden: float
    mutation_entropy: float
    immune_pressure: float
    treatment_pressure: float
    time: float

    def is_terminal(self):
        return self.tumor_burden <= 0 or self.tumor_burden > 100
    
