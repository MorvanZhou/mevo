import numpy as np

from mevo.gene.gene_type import GeneShape
from mevo.gene.initializer.base import Initializer
from mevo.gene.tools import convert_gene_shape


class RandomOrder(Initializer):

    def initialize(self, shape: GeneShape) -> np.ndarray:
        shape = convert_gene_shape(shape)
        a = np.concatenate([np.random.permutation(shape[1])[None, :] for _ in range(shape[0])], axis=0)
        return a
