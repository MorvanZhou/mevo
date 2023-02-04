from abc import ABC, abstractmethod

import numpy as np

from mevo import mtype


class Initializer(ABC):

    def __init__(self, seed: int = None):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def initialize(self, size: mtype.ChromosomeSize) -> np.ndarray:
        pass

    def set_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)
