import numpy as np

from mevo import mtype
from mevo.chromosome.initializer.base import Initializer


class RandomUniform(Initializer):
    def __init__(self, low: float, high: float, seed: int = None):
        super().__init__(seed=seed)
        if low > high:
            raise ValueError(f"low is greater than high, {low} > {high}")
        self.low = low
        self.high = high

    def initialize(self, size: mtype.ChromosomeSize) -> np.ndarray:
        return self.rng.uniform(low=self.low, high=self.high, size=size).astype(np.float32)
