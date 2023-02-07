import unittest

import numpy as np

import mevo


class ChromosomeTest(unittest.TestCase):
    def test_bool(self):
        for t, o_ in [
            ([-1, 0, 1, 2, 3], [True, False, True, True, True]),
            ([1, 0, 0, 1], [True, False, False, True]),
            ([True, True, False], [True, True, False]),
            ([[1, 1], [1, 0]], [[True, True], [True, False]])
        ]:
            o = mevo.BinaryChromo(t)
            self.assertTrue(np.all(o_ == o.data), msg=f"{o_}, {o.data}")

        o = mevo.BinaryChromo()
        o.random_init(size=10)
        self.assertEqual(np.bool_, o.data.dtype)

    def test_float(self):
        for t, o_, low, high in [
            ([-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3], -10, 10),
            ([1.2, 0.1, 0.3, 1.1], [1.2, 0.1, 0.3, 1.1], -10, 10),
            ([[1.1, 1.2], [1.151, 1.152]], [[1.15, 1.16], [1.151, 1.152]], 1.15, 1.16),
            ([[[1.2]], [[0.1]], [[0.3]], [[1.1]]], [[[1.2]], [[0.1]], [[0.3]], [[1.1]]], -10, 10),
        ]:
            o = mevo.FloatChromo(low=low, high=high, data=t)
            self.assertTrue(np.all(np.array(o_, dtype=np.float32) == o.data), msg=f"{o_}, {o.data}")

        o = mevo.FloatChromo(low=1, high=3)
        o.random_init(10)
        self.assertTrue(np.all((o.data <= 3) & (o.data >= 1)))

    def test_int(self):
        for t, o_, low, high in [
            ([-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3], -10, 10),
            ([1.2, 0.1, 0.3, 1.1], [1, 0, 0, 1], -10, 10),
            ([[0, 6], [2, 3]], [[1, 5], [2, 3]], 1, 5)
        ]:
            o = mevo.IntChromo(low=low, high=high, data=t)
            self.assertTrue(np.all(np.array(o_, dtype=np.int32) == o.data), msg=f"{o_}, {o.data}")

        o = mevo.IntChromo(low=1, high=3)
        o.random_init(10)
        self.assertTrue(np.all((o.data <= 3) & (o.data >= 1)))
