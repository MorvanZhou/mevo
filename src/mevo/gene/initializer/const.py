import typing as tp

import numpy as np

from mevo.gene.gene_type import GeneShape
from mevo.gene.initializer.base import Initializer
from mevo.gene.tools import convert_gene_shape


class Const(Initializer):
    def __init__(self, value: tp.Union[int, float]):
        self.value = value

    def initialize(self, shape: GeneShape) -> np.ndarray:
        return np.full(shape=convert_gene_shape(shape), fill_value=self.value)
