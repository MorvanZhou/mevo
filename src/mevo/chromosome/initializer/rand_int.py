import numpy as np

from mevo import mtype
from mevo.chromosome.initializer.base import Initializer


class RandomInt(Initializer):
    def __init__(self, low: int, high: int):
        if low > high:
            raise ValueError(f"low is greater than high, {low} > {high}")
        self.low = low
        self.high = high

    def initialize(self, size: mtype.ChromosomeSize) -> np.ndarray:
        return np.random.randint(low=self.low, high=self.high, size=size).astype(np.int32)
