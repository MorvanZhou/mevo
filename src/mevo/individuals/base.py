import typing as tp
from abc import ABC, abstractmethod

import numpy as np

from mevo.chromosomes import Chromosome


class Individual(ABC):
    chromosomes: tp.List[Chromosome]

    def __init__(self, rng: tp.Optional[np.random.Generator]):
        if rng is None:
            rng = np.random.default_rng()
        self._rng = rng
        self._id: str = ""
        self.fitness: float = 0.

    @property
    def id(self) -> str:
        return self._id

    @abstractmethod
    def mutate(self, rate: float) -> None:
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplemented
