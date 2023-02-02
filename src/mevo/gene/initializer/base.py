from abc import ABC, abstractmethod

import numpy as np

from mevo.gene.gene_type import GeneShape


class Initializer(ABC):

    @abstractmethod
    def initialize(self, shape: GeneShape) -> np.ndarray:
        pass
