import unittest

import numpy as np

from mevo.chromosome import initializer


class InitTest(unittest.TestCase):

    def test_const(self):
        init = initializer.Const(2)
        for t in [
            2,
            3,
            (2, 3),
            (3, )
        ]:
            if isinstance(t, int):
                t = (t,)
            self.assertTrue(np.all(np.full(t, 2) == init.initialize(t)))

    def test_randint(self):
        low = -1
        high = 3
        init = initializer.RandomInt(low=low, high=high)
        for t in [
            2,
            3,
            (2, 3),
            (3,)
        ]:
            data = init.initialize(t)
            if isinstance(t, int):
                t = (t,)
            self.assertEqual(t, data.shape)
            self.assertTrue(np.all((data >= low) & (data <= high)), msg=data)

    def test_rand_norm(self):
        init = initializer.RandomNorm(mean=0, std=0)
        for t in [
            2,
            3,
            1,
            (2, 3),
            (3,)
        ]:
            data = init.initialize(t)
            if isinstance(t, int):
                t = (t,)
            self.assertEqual(t, data.shape)
            self.assertTrue(np.all(data == 0), msg=data)

    def test_rand_order(self):
        init = initializer.RandomOrder()
        for t in [
            2,
            3,
            1,
            (2, 3),
            (3,)
        ]:
            data = init.initialize(t)
            data = np.sort(data, axis=-1)
            if data.ndim == 1:
                if isinstance(t, int):
                    t = (t,)
                self.assertTrue(np.all(data == np.arange(len(data))), msg=data)
            else:
                self.assertTrue(
                    np.all(data == np.repeat(np.arange(data.shape[-1])[None, :], data.shape[0], axis=0)),
                    msg=data)
            self.assertEqual(t, data.shape)

    def test_rand_uniform(self):
        low = 1
        high = 3
        init = initializer.RandomUniform(low=low, high=high)
        for t in [
            2,
            3,
            1,
            (2, 3),
            (3,)
        ]:
            data = init.initialize(t)
            if isinstance(t, int):
                t = (t,)
            self.assertEqual(t, data.shape)
            self.assertTrue(np.all((data >= low) & (data <= high)), msg=data)

    def test_seed(self):
        init = initializer.RandomInt(0, 10, seed=1)
        a = init.rng.integers(low=1, high=10, size=10)
        b = init.rng.integers(low=1, high=10, size=10)
        self.assertTrue(np.all(a != b))
        init.set_seed(1)
        b = init.rng.integers(low=1, high=10, size=10)
        self.assertTrue(np.all(a == b))
