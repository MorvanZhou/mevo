import typing as tp
from abc import ABC, abstractmethod

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
            mutation_rate: float = 0.01
    ):
        self.max_size = max_size
        if isinstance(chromo_size, int):
            chromo_size = [1] * chromo_size
        self.chromo_shape: tp.Sequence[int] = chromo_size
        self.mutation_rate = mutation_rate

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
