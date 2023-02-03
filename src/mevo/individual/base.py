import typing as tp
from abc import ABC, abstractmethod

import numpy as np

from mevo.chromosome import Chromosome


class Individual(ABC):
    chromosomes: tp.List[Chromosome]

    def __init__(self):
        self._id: str = ""

    @property
    def id(self) -> str:
        return self._id

    @abstractmethod
    def mutate(self, rate: float) -> None:
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplemented
