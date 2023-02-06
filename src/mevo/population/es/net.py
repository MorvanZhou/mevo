import typing as tp

import numpy as np

from mevo import chromosomes
from mevo import individuals
from mevo import utils
from mevo.population.base import Population


class SGD:   # optimizer with momentum
    def __init__(self, cs: tp.List[chromosomes.Chromosome], learning_rate=1e-3, momentum=0.9):
        self.params = [
            np.zeros_like(c.data).astype(np.float32) for c in cs
        ]
        self.cs = cs
        self.lr, self.momentum = learning_rate, momentum

    def apply_gradients(self, cumulative_update: tp.List[np.ndarray]):
        for p, c, g in zip(self.params, self.cs, cumulative_update):
            p[:] = self.momentum * p + (1. - self.momentum) * g
            c.data += self.lr * p


class EvolutionStrategyNet(Population):
    """
    """
    default_w_initializer = chromosomes.initializers.RandomNorm(mean=0, std=0.1)
    default_b_initializer = chromosomes.initializers.Const(0.)

    def __init__(
            self,
            max_size: int,
            layer_size: tp.Sequence[int],
            mutate_strength: float = 0.01,
            learning_rate: float = 0.05,
            w_initializer: tp.Optional[chromosomes.initializers.Initializer] = None,
            b_initializer: tp.Optional[chromosomes.initializers.Initializer] = None,
            parallel: bool = False,
            seed: int = None,
    ):
        chromo_size = []
        for i in range(len(layer_size) - 1):
            chromo_size.append((layer_size[i] + 1, layer_size[i+1]))

        super().__init__(
            max_size=max_size,
            chromo_size=chromo_size,
            parallel=parallel,
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

        base = self.max_size * 2  # *2 for mirrored sampling
        rank = np.arange(1, base + 1)
        util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
        self.utility = util_ / util_.sum() - 1 / base
        self.mutate_strength = mutate_strength

        self._build()
        self.opt = SGD(cs=self.individual.chromosomes, learning_rate=learning_rate, momentum=0.9)
        self.top = self.individual

    def _build(self):
        cs: tp.List[chromosomes.chromosome] = []
        for i in range(0, len(self.chromo_shape) - 1, 2):
            w_size = self.chromo_shape[i]
            b_size = self.chromo_shape[i+1]
            w_data = self.w_initializer.initialize(size=w_size)
            b_data = self.b_initializer.initialize(size=b_size)
            cs.append(chromosomes.FloatChromo(data=w_data))
            cs.append(chromosomes.FloatChromo(data=b_data))
        self.individual = individuals.EvolutionStrategyDense(mutate_strength=self.mutate_strength, rng=self._rng)
        self.individual.chromosomes = cs


    def evolve(
            self,
            fitness_fn: tp.Callable[[individuals.EvolutionStrategyDense, dict], float],
    ):
        # mirrored sampling, pass seed instead whole noise matrix to parallel will save your time
        noise_seed = self._rng.integers(0, 2 ** 32 - 1, size=self.max_size, dtype=np.uint32).repeat(2)

        if self.parallel:
            self._check_parallel_condition()

            try:
                res = [
                    self._pool.apply_async(
                        fitness_fn,
                        (self.individual, {"index": i, "seed": int(noise_seed[i])})
                    ) for i in range(self.max_size * 2)
                ]
                fitness = np.array([r.get() for r in res])

            except AttributeError as err:
                raise AttributeError(
                    "check your fitness function,"
                    " try move fitness function on the outside of any class or function."
                    f" The original error message is:\n {err}")
        else:
            fitness = map(fitness_fn,
                          [self.individual for _ in range(self.max_size * 2)],
                          [{"index": i, "seed": int(noise_seed[i])} for i in range(self.max_size * 2)],
                          )
            fitness = np.array(fitness)
        rank = np.argsort(fitness)[::-1]
        self.individual.fitness = fitness[rank[0]]

        cumulative_update = []  # initialize update values
        for c in self.individual.chromosomes:
            cumulative_update.append(np.zeros_like(c.data))
        for ui, k_id in enumerate(rank):
            # reconstruct noise using seed
            rng = np.random.default_rng(seed=noise_seed[k_id])
            for i, c in enumerate(self.individual.chromosomes):
                cumulative_update[i] += self.utility[ui] * utils.sign(k_id) * rng.normal(0, 1, c.data.shape) / (
                        2 * self.max_size * self.mutate_strength)

        self.opt.apply_gradients(cumulative_update)
