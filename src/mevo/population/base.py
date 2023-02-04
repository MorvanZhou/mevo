import multiprocessing
import textwrap
import typing as tp
from abc import ABC, abstractmethod
from queue import PriorityQueue

import numpy as np

from mevo import chromosome
from mevo import mtype
from mevo.individual.base import Individual


class Population(ABC):
    default_initializer: chromosome.Initializer

    def __init__(
            self,
            max_size: int,
            chromo_size: mtype.ChromosomeSize,
            mutation_rate: float = 0.01,
            parallel: bool = False,
            seed: int = None,
    ):
        self.max_size = max_size
        if isinstance(chromo_size, int):
            chromo_size = [1] * chromo_size
        self.chromo_shape: tp.Sequence[int] = chromo_size
        self.mutation_rate = mutation_rate
        self.parallel = parallel
        self.seed = seed
        self._parallel_in_with_statement = False
        self._pool = None

        self.global_iid = 0
        self.members: tp.Dict[str, Individual] = {}

    @abstractmethod
    def _build(self):
        pass

    @abstractmethod
    def crossover(self, n_children) -> tp.List[Individual]:
        ...

    def add(self, ind: Individual) -> str:
        if ind.id != "" or ind.id in self.members:
            raise KeyError(f"individual '{ind.id}' has been added")
        iid = str(self.global_iid)
        ind._id = iid
        self.members[iid] = ind
        self.global_iid += 1
        return iid

    def remove_by_id(self, ind: str):
        if ind not in self.members:
            raise KeyError(f"individual id '{ind}' is not found")
        del self.members[ind]

    def remove(self, ind: Individual):
        if ind.id == "":
            raise ValueError("this individual is not added to population")
        self.remove_by_id(ind.id)

    def reproduce(self, n_children: int):
        children = self.crossover(n_children=n_children)
        for c in children:
            c.mutate(self.mutation_rate)
            self.add(c)

    def pick_parents(self, n: int) -> tp.Tuple[Individual, Individual]:
        pop_id = list(self.members.keys())
        parents = np.random.choice(pop_id, size=n * 2, replace=True)
        fathers_id = parents[:n]
        mothers_2id = parents[n:]
        for (fid, mid) in zip(fathers_id, mothers_2id):
            yield self.members[fid], self.members[mid]

    def filter(
            self,
            fitness_fn: tp.Callable[[Individual], float],
            drop_rate: float,
    ) -> tp.Tuple[Individual, tp.Dict[str, float], tp.Dict[str, float]]:
        if drop_rate >= 1. or drop_rate <= 0:
            raise ValueError(f"drop_rate must in range of (0, 1), but got {drop_rate}")
        ids = list(self.members.keys())
        inds = [self.members[k] for k in ids]

        q = PriorityQueue(maxsize=self.max_size)
        kept = {}

        if self.parallel:
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
                    f"if run on parallel mode, please use with statement. For example:{t}"
                    )

            try:
                res = self._pool.map_async(fitness_fn, inds)
                for f in res.get():
                    iid = ids.pop(0)
                    q.put((f, iid))
                    kept[iid] = f
            except AttributeError as err:
                raise AttributeError(
                    "check your fitness function,"
                    " try move fitness function on the outside of any class or function."
                    f" The original error message is:\n {err}")
        else:
            fitness = map(fitness_fn, inds)
            for f in fitness:
                iid = ids.pop(0)
                q.put((f, iid))
                kept[iid] = f

        drop_num = max(1, int(self.max_size * drop_rate))
        dropped = {}
        for _ in range(drop_num):
            f, iid = q.get()
            dropped[iid] = kept.pop(iid)
            self.remove_by_id(iid)
        while q.qsize() > 1:
            q.get_nowait()
        iid = q.get()[1]
        top = self.members[iid]
        return top, kept, dropped

    def evolve(
            self,
            fitness_fn: tp.Callable[[Individual], float],
            drop_rate: float = 0.4,
    ) -> tp.Tuple[Individual, tp.Dict[str, float], tp.Dict[str, float]]:
        # selection
        top, kept, dropped = self.filter(
            fitness_fn=fitness_fn,
            drop_rate=drop_rate,
        )

        # reproduce
        self.reproduce(n_children=len(dropped))
        return top, kept, dropped

    def run(
            self,
            step: int,
            fitness_fn: tp.Callable[[Individual], float],
            drop_rate: float = 0.4,
    ):
        for _ in range(step):
            self.evolve(fitness_fn=fitness_fn, drop_rate=drop_rate)

    def __enter__(self):
        self._parallel_in_with_statement = True
        if self.parallel:
            ctx = multiprocessing.get_context(method='spawn')
            cc = multiprocessing.cpu_count()
            self._pool = ctx.Pool(
                processes=self.max_size if self.max_size < cc else cc,
            )
        return self

    def __exit__(self, *args):
        if self._pool is not None:
            self._pool.close()
        self._parallel_in_with_statement = False
