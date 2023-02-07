import typing as tp

from mevo import chromosomes
from mevo import individuals
from mevo.population.ga.base import GAPopulation


class GeneticAlgoNet(GAPopulation):
    """
    Individual is a list of chromosomes:
    [chromosome1, chromosome2, chromosome3, ...]

    In each chromosome is a np.ndarray(dtype=np.float32). Such as:
    [[1.1, 0.1], [2.4, 3.0], [1.4, 4.4]]

    Each chromosome will reshaped to form a neural network's layer who's layer size is defined by layer_size.

    Put it together, an individual is looks like:
    [
        [[1.1, 0.1], [2.4, 3.0], [1.4, 4.4], [1.1, 2]],     # chromosome1: layer1's parameters
        [[1,2,3], [1,2,3], [0,0,0]],                        # chromosome2: layer2's parameters
    ]

    Each chromosome is a crossover point which can be crossed by other individual's chromosome,
    which means layer can be exchanged in each network.

    And mutation is happened on each chromosome by a mutation rate.
    """
    default_w_initializer = chromosomes.initializers.RandomNorm(mean=0, std=0.1)
    default_b_initializer = chromosomes.initializers.Const(0.)

    def __init__(
            self,
            max_size: int,
            layer_size: tp.Sequence[int],
            drop_rate: float = 0.4,
            mutate_rate: float = 0.5,
            mutate_strength: float = 0.05,
            n_worker: int = 1,
            w_initializer: tp.Optional[chromosomes.initializers.Initializer] = None,
            b_initializer: tp.Optional[chromosomes.initializers.Initializer] = None,
            seed: int = None,
    ):
        chromo_size = []
        for i in range(len(layer_size) - 1):
            chromo_size.append((layer_size[i] + 1, layer_size[i+1]))

        super().__init__(
            max_size=max_size,
            chromo_size=chromo_size,
            drop_rate=drop_rate,
            mutate_rate=mutate_rate,
            n_worker=n_worker,
            seed=seed,
        )
        if w_initializer is None:
            w_initializer = self.default_w_initializer
        if b_initializer is None:
            b_initializer = self.default_b_initializer
        self.w_initializer = w_initializer
        self.b_initializer = b_initializer
        self.w_initializer.set_seed(self._rng.integers(0, 2 ** 32 - 1))
        self.b_initializer.set_seed(self._rng.integers(0, 2 ** 32 - 1))
        self.mutate_strength = mutate_strength
        self._build()

    def _build(self):
        for _ in range(self.max_size):
            cs = []
            for i in range(0, len(self.chromo_shape) - 1, 2):
                w_size = self.chromo_shape[i]
                b_size = self.chromo_shape[i+1]
                w_data = self.w_initializer.initialize(size=w_size)
                b_data = self.b_initializer.initialize(size=b_size)
                cs.append(chromosomes.FloatChromo(data=w_data))
                cs.append(chromosomes.FloatChromo(data=b_data))
            ind = individuals.GeneticAlgoDense(mutate_strength=self.mutate_strength, rng=self._rng)
            ind.chromosomes = cs
            self.add(ind)

    def crossover(self, n_children: int) -> tp.List[individuals.GeneticAlgoDense]:
        new_children = []
        for p1, p2 in self.pick_parents(n_children):
            child = individuals.GeneticAlgoDense(mutate_strength=self.mutate_strength, rng=self._rng)
            for c1, c2 in zip(p1.chromosomes, p2.chromosomes):
                c = c1.copy() if self._rng.random() < 0.5 else c2.copy()
                child.chromosomes.append(c)
            new_children.append(child)
        return new_children
