import numpy as np

from mevo import mtype
from mevo.chromosome.initializer.base import Initializer


class RandomNorm(Initializer):
    def __init__(self, mean: float, std: float):
        if std < 0:
            raise ValueError(f"std must >= 0, but got {std}")
        self.mean = mean
        self.std = std

    def initialize(self, size: mtype.ChromosomeSize) -> np.ndarray:
        return np.random.normal(loc=self.mean, scale=self.std, size=size).astype(np.float32)
