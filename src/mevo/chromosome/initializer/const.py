import typing as tp

import numpy as np

from mevo import mtype
from mevo.chromosome.initializer.base import Initializer


class Const(Initializer):
    def __init__(self, value: tp.Union[int, float]):
        self.value = value

    def initialize(self, size: mtype.ChromosomeSize) -> np.ndarray:
        return np.full(shape=size, fill_value=self.value).astype(np.float32)
