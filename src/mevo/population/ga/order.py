import typing as tp

import numpy as np

from mevo import chromosomes
from mevo import individuals
from mevo import mtype
from mevo.population.ga.base import GAPopulation


class GeneticAlgoOrder(GAPopulation):
    """
    Individual is a list of chromosomes:
    [chromosome1, chromosome2, chromosome3, ...]

    In each chromosome is a np.ndarray(dtype=np.int32). Such as:
    [1,]

    Put it together, an individual is looks like:
    [[1,], [3,], [0,], [2]]

    Note, every chromosome in the GeneticAlgoOrder only contain one gene,
    and chromosomes are unique for keeping an order.
    """
    def __init__(
            self,
            max_size: int,
            chromo_size: mtype.ChromosomeSize,
            drop_rate: float = 0.4,
            mutate_rate: float = 0.01,
            n_worker: int = 1,
            seed: int = None,
    ):
        super().__init__(
            max_size=max_size,
            chromo_size=chromo_size,
            drop_rate=drop_rate,
            mutate_rate=mutate_rate,
            n_worker=n_worker,
            seed=seed,
        )
        self._build()

    def _build(self):
        for _ in range(self.max_size):
            cs = []
            order = self._rng.permutation(len(self.chromo_shape))
            for i in order:
                cs.append(chromosomes.IntChromo(low=0, high=len(self.chromo_shape), data=np.array([i])))
            ind = individuals.GeneticAlgoInt()
            ind.chromosomes = cs
            self.add(ind)

    def crossover(self, n_children: int) -> tp.List[individuals.GeneticAlgoOrder]:
        new_children = []
        for p1, p2 in self.pick_parents(n_children):
            child = individuals.GeneticAlgoOrder(rng=self._rng)
            dropped = []
            for c1 in p1.chromosomes:
                c1: chromosomes.IntChromo
                if self._rng.random() < 0.5:
                    child.chromosomes.append(c1.copy())
                else:
                    dropped.append(c1.data[0])
            for c2 in p2.chromosomes:
                c2: chromosomes.IntChromo
                if c2.data[0] in dropped:
                    child.chromosomes.append(c2.copy())
            new_children.append(child)
        return new_children
