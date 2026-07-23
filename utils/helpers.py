from __future__ import annotations
from typing import List
import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)

def normalize(values: list[float]) -> List[float]:
    total = sum(values)
    if total == 0:
        return values
    return [v / total for v in values]

def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def debug_print(msg: str, enabled: bool = False):
    if enabled:
        print(f"[DEBUG] {msg}")
