from abc import ABC, abstractmethod

import numpy as np

from mevo import mtype


class Initializer(ABC):

    @abstractmethod
    def initialize(self, size: mtype.ChromosomeSize) -> np.ndarray:
        pass
