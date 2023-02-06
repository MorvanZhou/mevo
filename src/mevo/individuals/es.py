import typing as tp

import numpy as np

from mevo import chromosomes
from mevo import utils
from mevo.individuals.base import Individual


class EvolutionStrategyDense(Individual):
    def __init__(self, mutate_strength: float, rng: np.random.Generator = None):
        super().__init__(rng=rng)
        self.chromosomes: tp.List[chromosomes.FloatChromo] = []
        self.mutate_strength = mutate_strength

    def mutate(self, rate: float):
        for c in self.chromosomes:
            indices = [i for i in range(c.data.size) if self._rng.random() < rate]
            c.data.ravel()[indices] += self._rng.normal(0, self.mutate_strength, size=len(indices)).astype(np.float32)

    def clone_with_mutate(self, index: int, seed: int):
        rng = np.random.default_rng(seed=seed)
        cc = []
        for c in self.chromosomes:
            p = self.mutate_strength * rng.normal(0, 1, size=c.data.shape).astype(np.float32)
            w = c.data + utils.sign(index) * p
            cc.append(chromosomes.FloatChromo(data=w))
        i = self.__class__(self.mutate_strength)
        i.chromosomes = cc
        return i

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
