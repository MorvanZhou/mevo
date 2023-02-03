import typing as tp

import numpy as np

from mevo import chromosome
from mevo.individual.base import Individual


class GeneticAlgoInt(Individual):
    def __init__(self):
        super().__init__()
        self.chromosomes: tp.List[chromosome.IntChromo] = []

    def mutate(self, rate: float):
        for c in self.chromosomes:
            indices = [i for i in range(c.data.size) if np.random.random() < rate]
            c.data.ravel()[indices] = np.random.randint(c.low, c.high, size=len(indices))


class GeneticAlgoOrder(Individual):
    def __init__(self):
        super().__init__()
        self.chromosomes: tp.List[chromosome.IntChromo] = []
        self.init = chromosome.initializer.RandomOrder()

    def mutate(self, rate: float):
        lc = len(self.chromosomes)
        for i in range(lc):
            if np.random.rand() < rate:
                idx = np.random.randint(0, lc)
                self.chromosomes[i], self.chromosomes[idx] = self.chromosomes[idx], self.chromosomes[i]


class GeneticAlgoFloat(Individual):
    def __init__(self):
        super().__init__()
        self.chromosomes: tp.List[chromosome.FloatChromo] = []

    def mutate(self, rate: float):
        for c in self.chromosomes:
            indices = [i for i in range(c.data.size) if np.random.random() < rate]
            c.data.ravel()[indices] += np.random.normal(0, 0.1, size=len(indices))


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
