import typing as tp

import numpy as np

from mevo import chromosome
from mevo import individual
from mevo import mtype
from mevo.population.base import Population


class GeneticAlgoInt(Population):
    default_initializer = chromosome.initializer.RandomInt(0, 1)

    def __init__(
            self,
            max_size: int,
            chromo_size: mtype.ChromosomeSize,
            chromo_initializer: tp.Optional[chromosome.initializer.RandomInt] = None,
            mutation_rate: float = 0.01,
            parallel: bool = False,
            seed: int = None,
    ):
        super().__init__(
            max_size=max_size,
            chromo_size=chromo_size,
            mutation_rate=mutation_rate,
            parallel=parallel,
            seed=seed,
        )
        if chromo_initializer is None:
            chromo_initializer = self.default_initializer
        self.chromo_initializer = chromo_initializer
        self.chromo_initializer.set_seed(seed)
        self._rng = np.random.default_rng(seed)
        self.low = self.chromo_initializer.low
        self.high = self.chromo_initializer.high
        self._build()

    def _build(self):
        for _ in range(self.max_size):
            cs = []
            for size in self.chromo_shape:
                data = self.chromo_initializer.initialize(size=size)
                cs.append(chromosome.IntChromo(low=self.low, high=self.high, data=data))
            ind = individual.GeneticAlgoInt()
            ind.chromosomes = cs
            self.add(ind)

    def crossover(self, n_children: int) -> tp.List[individual.GeneticAlgoInt]:
        new_children = []
        for p1, p2 in self.pick_parents(n_children):
            child = individual.GeneticAlgoInt()
            for c1, c2 in zip(p1.chromosomes, p2.chromosomes):
                c = c1.copy() if self._rng.random() < 0.5 else c2.copy()
                child.chromosomes.append(c)
            new_children.append(child)
        return new_children


class GeneticAlgoOrder(Population):
    def __init__(
            self,
            max_size: int,
            chromo_size: mtype.ChromosomeSize,
            mutation_rate: float = 0.01,
            parallel: bool = False,
            seed: int = None,
    ):
        super().__init__(
            max_size=max_size,
            chromo_size=chromo_size,
            mutation_rate=mutation_rate,
            parallel=parallel,
            seed=seed,
        )
        self._rng = np.random.default_rng(seed)
        self._build()

    def _build(self):
        for _ in range(self.max_size):
            cs = []
            order = self._rng.permutation(len(self.chromo_shape))
            for i in order:
                cs.append(chromosome.IntChromo(low=0, high=len(self.chromo_shape), data=np.array([i])))
            ind = individual.GeneticAlgoInt()
            ind.chromosomes = cs
            self.add(ind)

    def crossover(self, n_children: int) -> tp.List[individual.GeneticAlgoOrder]:
        new_children = []
        for p1, p2 in self.pick_parents(n_children):
            child = individual.GeneticAlgoOrder()
            dropped = []
            for c1 in p1.chromosomes:
                c1: chromosome.IntChromo
                if self._rng.random() < 0.5:
                    child.chromosomes.append(c1.copy())
                else:
                    dropped.append(c1.data[0])
            for c2 in p2.chromosomes:
                c2: chromosome.IntChromo
                if c2.data[0] in dropped:
                    child.chromosomes.append(c2.copy())
            new_children.append(child)
        return new_children


class GeneticAlgoFloat(Population):
    default_initializer = chromosome.initializer.RandomNorm(mean=0, std=0.1)

    def __init__(
            self,
            max_size: int,
            chromo_size: mtype.ChromosomeSize,
            chromo_initializer: tp.Optional[chromosome.initializer.RandomNorm] = None,
            mutation_rate: float = 0.01,
            parallel: bool = False,
            seed: int = None,
    ):
        super().__init__(
            max_size=max_size,
            chromo_size=chromo_size,
            mutation_rate=mutation_rate,
            parallel=parallel,
            seed=seed,
        )
        if chromo_initializer is None:
            chromo_initializer = self.default_initializer
        self.chromo_initializer = chromo_initializer
        self.chromo_initializer.set_seed(seed)
        self._rng = np.random.default_rng(seed)
        self._build()

    def _build(self):
        for _ in range(self.max_size):
            cs = []
            for size in self.chromo_shape:
                data = self.chromo_initializer.initialize(size=size)
                cs.append(chromosome.FloatChromo(data=data))
            ind = individual.GeneticAlgoFloat()
            ind.chromosomes = cs
            self.add(ind)

    def crossover(self, n_children: int) -> tp.List[individual.GeneticAlgoFloat]:
        new_children = []
        for p1, p2 in self.pick_parents(n_children):
            child = individual.GeneticAlgoFloat()
            for c1, c2 in zip(p1.chromosomes, p2.chromosomes):
                c = c1.copy() if self._rng.random() < 0.5 else c2.copy()
                child.chromosomes.append(c)
            new_children.append(child)
        return new_children


class GeneticAlgoNet(Population):
    default_w_initializer = chromosome.initializer.RandomNorm(mean=0, std=0.1)
    default_b_initializer = chromosome.initializer.Const(0.)

    def __init__(
            self,
            max_size: int,
            layer_size: tp.Sequence[int],
            w_initializer: tp.Optional[chromosome.initializer.Initializer] = None,
            b_initializer: tp.Optional[chromosome.initializer.Initializer] = None,
            mutation_rate: float = 0.01,
            parallel: bool = False,
            seed: int = None,
    ):
        chromo_size = []
        for i in range(len(layer_size) - 1):
            chromo_size.append((layer_size[i] + 1, layer_size[i+1]))

        super().__init__(
            max_size=max_size,
            chromo_size=chromo_size,
            mutation_rate=mutation_rate,
            parallel=parallel,
            seed=seed,
        )
        if w_initializer is None:
            w_initializer = self.default_w_initializer
        if b_initializer is None:
            b_initializer = self.default_b_initializer
        self.w_initializer = w_initializer
        self.b_initializer = b_initializer
        self.w_initializer.set_seed(seed)
        self.b_initializer.set_seed(seed)
        self._rng = np.random.default_rng(seed)
        self.n_layer = len(chromo_size) - 1
        self._build()

    def _build(self):
        for _ in range(self.max_size):
            cs = []
            for i in range(0, len(self.chromo_shape) - 1, 2):
                w_size = self.chromo_shape[i]
                b_size = self.chromo_shape[i+1]
                w_data = self.w_initializer.initialize(size=w_size)
                b_data = self.b_initializer.initialize(size=b_size)
                cs.append(chromosome.FloatChromo(data=w_data))
                cs.append(chromosome.FloatChromo(data=b_data))
            ind = individual.GeneticAlgoDense()
            ind.chromosomes = cs
            self.add(ind)

    def crossover(self, n_children: int) -> tp.List[individual.GeneticAlgoDense]:
        new_children = []
        for p1, p2 in self.pick_parents(n_children):
            child = individual.GeneticAlgoDense()
            for c1, c2 in zip(p1.chromosomes, p2.chromosomes):
                c = c1.copy() if self._rng.random() < 0.5 else c2.copy()
                child.chromosomes.append(c)
            new_children.append(child)
        return new_children
