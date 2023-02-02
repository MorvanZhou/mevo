import numpy as np

from mevo.gene.gene_type import GeneShape
from mevo.gene.initializer.base import Initializer
from mevo.gene.tools import convert_gene_shape


class RandomInt(Initializer):
    def __init__(self, low: int, high: int):
        if low > high:
            raise ValueError(f"low is greater than high, {low} > {high}")
        self.low = low
        self.high = high

    def initialize(self, shape: GeneShape) -> np.ndarray:
        return np.random.randint(low=self.low, high=self.high, size=convert_gene_shape(shape))
