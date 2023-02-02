import typing as tp
from abc import ABC, abstractmethod

from mevo.gene import Initializer, Gene, GeneShape
from mevo.individual.base import Individual


class Population(ABC):
    individual_type: tp.Type[Individual]
    default_initializer: Initializer

    def __init__(
            self,
            num: int,
            gene_shape: GeneShape,
            gene_initializer: Initializer,
    ):
        self.num = num
        self.gene_shape = gene_shape
        self.gene_initializer = gene_initializer

        if self.gene_initializer is None:
            self.gene_initializer = self.default_initializer
        if not isinstance(self.gene_initializer, self.default_initializer.__class__):
            raise TypeError(f"this initializer must be {self.default_initializer.__class__.__name__}")

        self.global_iid = 0
        self.members: tp.Dict[str, Individual] = {}
        self._build()

    def _build(
            self,
    ):
        for _ in range(self.num):
            gene = Gene(shape=self.gene_shape, initializer=self.gene_initializer)
            ind = self.individual_type(gene)
            self.add(ind)

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
            c.mutate()
            self.add(c)

    @abstractmethod
    def crossover(self, n_children) -> tp.List[Individual]:
        ...
