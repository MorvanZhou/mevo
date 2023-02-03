import typing as tp
from abc import ABC, abstractmethod

import numpy as np

from mevo import mtype
from mevo.chromosome import initializer


class Chromosome(ABC):
    data: np.ndarray
    initializer: initializer.Initializer

    @abstractmethod
    def random_init(self, size: mtype.ChromosomeSize):
        pass

    @abstractmethod
    def copy(self):
        pass

    def __repr__(self):
        return str(self.data)

    def __str__(self):
        return str(self.data)


class BinaryChromo(Chromosome):
    def __init__(self, data: tp.Union[tp.Sequence, np.ndarray] = None):
        if data is None:
            return
        if isinstance(data, (tuple, list)):
            data = np.array(data, dtype=np.bool_)
        if data.dtype not in (np.bool_, np.integer):
            raise TypeError("array type must be bool or integer")
        self.data = data.astype(np.bool_)

    def random_init(self, size: mtype.ChromosomeSize):
        self.data = initializer.RandomInt(low=0, high=1).initialize(size=size).astype(np.bool_)

    def copy(self):
        return BinaryChromo(data=self.data.copy())


class IntChromo(Chromosome):
    def __init__(self, low: int, high: int, data: tp.Union[tp.Sequence[int], np.ndarray] = None):
        self.low = low
        self.high = high
        if data is None:
            return
        if isinstance(data, (tuple, list)):
            data = np.array(data, dtype=np.int32)
        if not np.issubdtype(data.dtype, np.integer):
            raise TypeError("array type must be integer")
        data = np.clip(data, a_min=self.low, a_max=self.high)
        self.data = data.astype(np.int32)

    def random_init(self, size: mtype.ChromosomeSize):
        self.data = initializer.RandomInt(low=self.low, high=self.high).initialize(size=size)

    def copy(self):
        return IntChromo(low=self.low, high=self.high, data=self.data.copy())


class FloatChromo(Chromosome):
    def __init__(self, low: float = None, high: float = None, data: tp.Union[tp.Sequence[float], np.ndarray] = None):
        self.low = low
        self.high = high
        if data is None:
            return
        if isinstance(data, (tuple, list)):
            data = np.array(data, dtype=np.float32)
        data = data.astype(np.float32)
        if self.low is not None:
            data = np.maximum(data, self.low)
        if self.high is not None:
            data = np.minimum(data, self.high)
        self.data = data

    def random_init(self, size: mtype.ChromosomeSize):
        if self.low is not None and self.high is not None:
            self.data = initializer.RandomUniform(low=self.low, high=self.high).initialize(size=size)
        else:
            self.data = initializer.RandomNorm(0, 1).initialize(size=size)

    def copy(self):
        return FloatChromo(low=self.low, high=self.high, data=self.data.copy())
