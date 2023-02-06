import typing as tp

from mevo import chromosomes
from mevo import individuals
from mevo import mtype
from mevo.population.ga.base import GAPopulation


class GeneticAlgoFloat(GAPopulation):
    """
    Individual is a list of chromosomes:
    [chromosome1, chromosome2, chromosome3, ...]

    In each chromosome is a np.ndarray(dtype=np.float32). Such as:
    [1.1, 0.1, 2.4, 3.0, 1.4, 4.4]

    Put it together, an individual is looks like:
    [[1.1, 2.3], [1.1, 3.4, 4.5], [3.1,]]

    Each chromosome is a crossover point which can be crossed by other individual's chromosome.
    And mutation is happened on each chromosome by a mutation rate.
    """
    default_initializer = chromosomes.initializers.RandomNorm(mean=0, std=0.1)

    def __init__(
            self,
            max_size: int,
            chromo_size: mtype.ChromosomeSize,
            drop_rate: float = 0.4,
            mutate_rate: float = 0.7,
            mutate_strength: float = 0.05,
            parallel: bool = False,
            chromo_initializer: tp.Optional[chromosomes.initializers.RandomNorm] = None,
            seed: int = None,
    ):
        super().__init__(
            max_size=max_size,
            chromo_size=chromo_size,
            drop_rate=drop_rate,
            mutate_rate=mutate_rate,
            parallel=parallel,
            seed=seed,
        )
        if chromo_initializer is None:
            chromo_initializer = self.default_initializer
        self.chromo_initializer = chromo_initializer
        self.chromo_initializer.set_seed(self._rng.integers(0, 2 ** 32 - 1))
        self.mutate_strength = mutate_strength
        self._build()

    def _build(self):
        for _ in range(self.max_size):
            cs = []
            for size in self.chromo_shape:
                data = self.chromo_initializer.initialize(size=size)
                cs.append(chromosomes.FloatChromo(data=data))
            ind = individuals.GeneticAlgoFloat(mutate_strength=self.mutate_strength)
            ind.chromosomes = cs
            self.add(ind)

    def crossover(self, n_children: int) -> tp.List[individuals.GeneticAlgoFloat]:
        new_children = []
        for p1, p2 in self.pick_parents(n_children):
            child = individuals.GeneticAlgoFloat(mutate_strength=self.mutate_strength, rng=self._rng)
            for c1, c2 in zip(p1.chromosomes, p2.chromosomes):
                c = c1.copy() if self._rng.random() < 0.5 else c2.copy()
                child.chromosomes.append(c)
            new_children.append(child)
        return new_children
