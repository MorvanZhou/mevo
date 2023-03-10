import numpy as np

from mevo import mtype
from mevo.chromosomes.initializers.base import Initializer


class RandomOrder(Initializer):
    def __init__(self, seed: int = None):
        super().__init__(seed=seed)

    def initialize(self, size: mtype.ChromosomeSize) -> np.ndarray:
        if isinstance(size, int):
            a = self.rng.permutation(size)
        elif len(size) == 1:
            a = self.rng.permutation(size[0])
        else:
            if len(size) > 2:
                raise ValueError("len(size) should <= 2")
            if not isinstance(size[1], int) or not isinstance(size[0], int):
                raise ValueError("size should be a list of int")
            a = np.concatenate([self.rng.permutation(size[1])[None, :] for _ in range(size[0])], axis=0)
        return a.astype(np.int32)
