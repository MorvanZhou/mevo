import unittest

import numpy as np

from mevo.gene import initializer


class InitTest(unittest.TestCase):
    def check_shape(self, t, data):
        if isinstance(t, tuple) and len(t) > 1:
            self.assertEqual(t, data.shape)
        elif isinstance(t, int):
            self.assertEqual((1, t), data.shape)
        else:
            self.assertEqual((1, t[0]), data.shape)

    def test_const(self):
        init = initializer.Const(2)
        for t in [
            2,
            (3, 2),
            (1, 3),
            (3,),
            (1,)
        ]:
            self.assertTrue(np.all(np.full(t, 2) == init.initialize(t)))

    def test_randint(self):
        low = -1
        high = 3
        init = initializer.RandomInt(low=low, high=high)
        for t in [
            2,
            (3, 2),
            (1, 3),
            (3,),
            (1,)
        ]:
            data = init.initialize(t)
            self.check_shape(t, data)
            self.assertTrue(np.all((data >= low) & (data <= high)), msg=data)

    def test_rand_norm(self):
        init = initializer.RandomNorm(mean=0, std=0)
        for t in [
            2,
            (3, 2),
            (1, 3),
            (3,),
            (1,)
        ]:
            data = init.initialize(t)
            self.check_shape(t, data)
            self.assertTrue(np.all(data == 0), msg=data)

    def test_rand_order(self):
        init = initializer.RandomOrder()
        for t in [
            2,
            (3, 2),
            (1, 3),
            (3,),
            (1,)
        ]:
            data = init.initialize(t)
            self.check_shape(t, data)
            for row in data:
                s = np.sort(row)
                self.assertTrue(np.all(s == np.arange(len(s))), msg=s)

    def test_rand_uniform(self):
        low = 1
        high = 3
        init = initializer.RandomUniform(low=low, high=high)
        for t in [
            2,
            (3, 2),
            (1, 3),
            (3,),
            (1,)
        ]:
            data = init.initialize(t)
            self.check_shape(t, data)
            self.assertTrue(np.all((data >= low) & (data <= high)), msg=data)
