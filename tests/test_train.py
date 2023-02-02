import unittest

import numpy as np

import mevo


def fitness_fn(ind: mevo.individual.Individual) -> float:
    binary = ind.gene.data.ravel()
    x = binary.dot(2 ** np.arange(len(binary))[::-1]) / float(2 ** len(binary) - 1) * 5
    o = np.sin(10 * x) * x + np.cos(2 * x) * x
    return o


class GATest(unittest.TestCase):
    def test_ga_runner(self):
        pop = mevo.pop.GAInt(
            num=20,
            gene_shape=(10,),
            gene_initializer=mevo.gene.initializer.RandomInt(0, 1)
        )

        for _ in range(100):
            kept, dropped = mevo.evolve(pop, fitness_fn=fitness_fn, drop_ratio=0.3)
            self.assertLessEqual(max(list(dropped.values())), min(list(kept.values())))
            self.assertEqual(20, len(kept) + len(dropped))
