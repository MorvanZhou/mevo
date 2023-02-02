from abc import ABC, abstractmethod

from mevo.gene.gene import Gene


class Individual(ABC):

    def __init__(self, gene: Gene):
        self._id: str = ""
        self.gene = gene

    @property
    def id(self) -> str:
        return self._id

    @abstractmethod
    def mutate(self) -> None:
        pass
