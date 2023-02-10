import inspect
import multiprocessing
import typing as tp
from abc import ABC, abstractmethod
from queue import PriorityQueue

from mevo import mtype, utils
from mevo.individuals.base import Individual
from mevo.population.base import Population


class GAPopulation(Population, ABC):
    def __init__(
            self,
            max_size: int,
            chromo_size: mtype.ChromosomeSize,
            mutate_rate: float,
            drop_rate: float = 0.4,
            n_worker: int = 1,
            seed: int = None,
    ):
        super().__init__(
            max_size=max_size,
            chromo_size=chromo_size,
            n_worker=n_worker,
            seed=seed,
        )
        self.mutate_rate = mutate_rate
        if drop_rate >= 1. or drop_rate <= 0:
            raise ValueError(f"drop_rate must in range of (0, 1), but got {drop_rate}")
        self.drop_rate = drop_rate

    @property
    def members(self):
        return self._members

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

    @abstractmethod
    def crossover(self, n_children: int) -> tp.List[Individual]:
        pass

    def reproduce(self, n_children: int):
        children = self.crossover(n_children=n_children)
        for c in children:
            c.mutate(self.mutate_rate)
            self.add(c)

    def pick_parents(self, n: int) -> tp.Tuple[Individual, Individual]:
        pop_id = list(self.members.keys())
        parents = self._rng.choice(pop_id, size=n * 2, replace=True)
        fathers_id = parents[:n]
        mothers_2id = parents[n:]
        for (fid, mid) in zip(fathers_id, mothers_2id):
            yield self.members[fid], self.members[mid]

    def filter(
            self,
            fitness_fn: tp.Callable[[Individual, dict], float],
            drop_num: int,
    ):
        res = inspect.getfullargspec(fitness_fn)
        if "ind" not in res.args or "conf" not in res.args:
            raise ValueError("cannot find 'ind' or 'conf' argument in the definition of fitness function")

        ids = list(self.members.keys())
        inds = [self.members[k] for k in ids]

        q = PriorityQueue(maxsize=self.max_size)
        kept = {}

        if self.n_worker > 1:
            self._check_parallel_condition()

            try:
                res = [
                    self._pool.apply_async(
                        fitness_fn, (ind, {"seed": int(self._rng.integers(utils.MAX_UINT32))})) for ind in inds
                ]
                for r in res:
                    f = r.get()
                    iid = ids.pop(0)
                    q.put((f, iid))
                    kept[iid] = f
                    self.members[iid].fitness = f
            except AttributeError as err:
                raise AttributeError(
                    "check your fitness function,"
                    " try move fitness function on the outside of any class or function."
                    f" The original error message is:\n {err}")
        else:
            fitness = map(fitness_fn,
                          inds,
                          [{"seed": int(self._rng.integers(utils.MAX_UINT32))} for _ in range(self.max_size)])
            for f in fitness:
                iid = ids.pop(0)
                q.put((f, iid))
                kept[iid] = f
                self.members[iid].fitness = f

        [self.remove_by_id(q.get()[1]) for _ in range(drop_num)]

        while q.qsize() > 1:
            q.get_nowait()
        iid = q.get()[1]
        self.top = self.members[iid]

    def evolve(
            self,
            fitness_fn: tp.Callable[[Individual, dict], float],
    ):
        drop_num = max(1, int(self.max_size * self.drop_rate))
        # selection
        self.filter(
            fitness_fn=fitness_fn,
            drop_num=drop_num,
        )

        # reproduce
        self.reproduce(n_children=drop_num)

    def run(
            self,
            step: int,
            fitness_fn: tp.Callable[[Individual, dict], float],
    ):
        for _ in range(step):
            self.evolve(fitness_fn=fitness_fn)

    def __enter__(self):
        self._parallel_in_with_statement = True
        if self.n_worker > 1:
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
