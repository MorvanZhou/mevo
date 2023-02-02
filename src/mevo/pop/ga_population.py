import typing as tp

import numpy as np

from mevo import individual
from mevo.gene import Gene, initializer, GeneShape
from mevo.pop.base import Population


class GAInt(Population):
    individual_type = individual.GAInt
    default_initializer = initializer.RandomInt(0, 1)
    gene_initializer: initializer.RandomInt

    def __init__(
            self,
            num: int,
            gene_shape: GeneShape,
            gene_initializer: tp.Optional[initializer.RandomInt] = None,
    ):
        super().__init__(num=num, gene_shape=gene_shape, gene_initializer=gene_initializer)
        self.low = self.gene_initializer.low
        self.high = self.gene_initializer.high

    def crossover(self, n_children: int) -> tp.List[individual.GAInt]:
        new_children = []
        pop_id = list(self.members.keys())
        parents = np.random.choice(pop_id, size=n_children * 2, replace=True)
        p1id = parents[:n_children]
        p2id = parents[n_children:]
        for i in range(n_children):
            p1 = self.members[p1id[i]]
            p2 = self.members[p2id[i]]
            flat_child_gene = p1.gene.data.ravel().copy()
            length = len(flat_child_gene)
            mask = np.random.rand(length) < 0.5
            flat_child_gene[mask] = p2.gene.data.ravel()[mask]
            child_gene_data = flat_child_gene.reshape(p1.gene.shape)
            new_gene = Gene(shape=p1.gene.shape, initializer=initializer.RandomInt(low=self.low, high=self.high))
            new_gene.data = child_gene_data
            child = individual.GAInt(gene=new_gene)
            new_children.append(child)
        return new_children


class GAOrder(Population):
    individual_type = individual.GAOrder
    default_initializer = initializer.RandomOrder()
    gene_initializer: initializer.RandomInt

    def __init__(
            self,
            num: int,
            gene_shape: GeneShape,
            gene_initializer: tp.Optional[initializer.RandomOrder] = None,
    ):
        super().__init__(num=num, gene_shape=gene_shape, gene_initializer=gene_initializer)
        self.low = self.gene_initializer.low
        self.high = self.gene_initializer.high

    def crossover(self, n_children) -> tp.List[individual.GAOrder]:
        pass
