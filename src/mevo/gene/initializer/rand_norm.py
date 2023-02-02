import numpy as np

from mevo.gene.gene_type import GeneShape
from mevo.gene.initializer.base import Initializer
from mevo.gene.tools import convert_gene_shape


class RandomNorm(Initializer):
    def __init__(self, mean: float, std: float):
        if std < 0:
            raise ValueError(f"std must >= 0, but got {std}")
        self.mean = mean
        self.std = std

    def initialize(self, shape: GeneShape) -> np.ndarray:
        return np.random.normal(loc=self.mean, scale=self.std, size=convert_gene_shape(shape))
