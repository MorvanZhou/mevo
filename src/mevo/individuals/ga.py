import typing as tp

import numpy as np

from mevo import chromosomes
from mevo.individuals.base import Individual


class GeneticAlgoInt(Individual):
    def __init__(self, rng: np.random.Generator = None):
        super().__init__(rng=rng)
        self.chromosomes: tp.List[chromosomes.IntChromo] = []

    def mutate(self, rate: float):
        for c in self.chromosomes:
            indices = [i for i in range(c.data.size) if self._rng.random() < rate]
            c.data.ravel()[indices] = self._rng.integers(c.low, c.high, size=len(indices), dtype=np.int32)


class GeneticAlgoOrder(Individual):
    def __init__(self, rng: np.random.Generator = None):
        super().__init__(rng=rng)
        self.chromosomes: tp.List[chromosomes.IntChromo] = []
        self.init = chromosomes.initializers.RandomOrder()

    def mutate(self, rate: float):
        lc = len(self.chromosomes)
        for i in range(lc):
            if self._rng.random() < rate:
                idx = self._rng.integers(0, lc)
                self.chromosomes[i], self.chromosomes[idx] = self.chromosomes[idx], self.chromosomes[i]


class GeneticAlgoFloat(Individual):
    def __init__(self, mutate_strength: float, rng: np.random.Generator = None):
        super().__init__(rng=rng)
        self.chromosomes: tp.List[chromosomes.FloatChromo] = []
        self.mutate_strength = mutate_strength

    def mutate(self, rate: float):
        for c in self.chromosomes:
            indices = [i for i in range(c.data.size) if self._rng.random() < rate]
            c.data.ravel()[indices] += self._rng.normal(0, self.mutate_strength, size=len(indices)).astype(np.float32)


class GeneticAlgoDense(GeneticAlgoFloat):
    def predict(self, x: np.ndarray):
        if x.ndim == 1:
            o = x[None, :]
        else:
            o = x
        for c in self.chromosomes[:-1]:
            w = c.data[:-1]
            b = c.data[-1]
            o = o.dot(w) + b
            o = np.maximum(o, 0)  # ReLU
        w = self.chromosomes[-1].data[:-1]
        b = self.chromosomes[-1].data[-1]
        logits = o.dot(w) + b
        return logits.ravel()
