import numpy as np

from mevo.gene import initializer
from mevo.gene.gene import Gene
from mevo.individual.base import Individual


class GAInt(Individual):
    def __init__(self, gene: Gene):
        super().__init__(gene=gene)
        self.gene = gene
        self.mutate_rate = 0.05

        if not isinstance(self.gene.initializer, initializer.RandomInt):
            raise TypeError("GA must use initializer.RandomInt")
        init = self.gene.initializer
        self.low = init.low
        self.high = init.high

    def mutate(self):
        flat_gene = self.gene.data.ravel()
        length = len(flat_gene)
        indices = np.arange(length)[np.random.rand(length) < self.mutate_rate]
        flat_gene[indices] = np.random.randint(self.low, self.high, size=len(indices))


class GAOrder(Individual):
    def __init__(self, gene: Gene):
        super().__init__(gene=gene)
        self.gene = gene
        self.mutate_rate = 0.05

        if not isinstance(self.gene.initializer, initializer.RandomOrder):
            raise TypeError("GA must use initializer.RandomOrder")

    def mutate(self):
        for row in self.gene.data:
            if np.random.rand() < self.mutate_rate:
                idx = np.random.randint(0, len(row), size=2)
                row[idx] = row[idx[::-1]]
