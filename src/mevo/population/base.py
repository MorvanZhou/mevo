import multiprocessing
import textwrap
import typing as tp
from abc import ABC, abstractmethod

import numpy as np

from mevo import chromosomes
from mevo import mtype
from mevo.individuals.base import Individual


class Population(ABC):
    default_initializer: chromosomes.Initializer

    def __init__(
            self,
            max_size: int,
            chromo_size: mtype.ChromosomeSize,
            n_worker: int = 1,
            seed: int = None,
    ):
        self.max_size = max_size
        if isinstance(chromo_size, int):
            chromo_size = [1] * chromo_size
        self.chromo_shape: tp.Sequence[int] = chromo_size
        if n_worker <= -1:
            n_worker = multiprocessing.cpu_count()
        self.n_worker = n_worker
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._parallel_in_with_statement = False
        self._pool = None

        self.global_iid = 0
        self._members: tp.Dict[str, Individual] = {}
        self.top: tp.Optional[Individual] = None

    def _check_parallel_condition(self):
        if not self._parallel_in_with_statement:
            t = textwrap.dedent("""
                with mevo.GeneticAlgoNet(...) as pop:
                    for _ in range(step):
                        pop.evolve(
                            fitness_fn=fitness_fn,
                            drop_rate=0.3,
                        )
                """)
            raise RuntimeError(
                f"if run on parallel mode, please use 'with' statement. For example:{t}"
            )

    @abstractmethod
    def _build(self):
        pass

    @abstractmethod
    def evolve(
            self,
            fitness_fn: tp.Callable[[Individual, int], float],
    ):
        pass

    def run(
            self,
            step: int,
            fitness_fn: tp.Callable[[Individual, int], float],
    ):
        for _ in range(step):
            self.evolve(fitness_fn=fitness_fn)

    def __enter__(self):
        self._parallel_in_with_statement = True
        if self.n_worker > 1:
            ctx = multiprocessing.get_context(method='spawn')
            self._pool = ctx.Pool(
                processes=self.max_size if self.max_size < self.n_worker else self.n_worker,
            )
        return self

    def __exit__(self, *args):
        if self._pool is not None:
            self._pool.close()
        self._parallel_in_with_statement = False
