import typing as tp

from mevo.gene.gene_type import GeneShape
from mevo.gene.initializer.base import Initializer
from mevo.gene.tools import convert_gene_shape


class Gene:
    def __init__(self, shape: GeneShape, initializer: Initializer):
        self.shape: tp.Sequence[int] = convert_gene_shape(shape)
        self.initializer = initializer
        self.data = initializer.initialize(shape)
