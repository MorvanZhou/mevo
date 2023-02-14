from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, *inputs) -> np.ndarray:
        return self.forward(*inputs)


class Linear(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x


class ReLU(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(self.alpha * x, x)


class ELU(Activation):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, self.alpha * (np.exp(x) - 1))


class Tanh(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)


class Sigmoid(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1. / (1. + np.exp(-x))


class SoftPlus(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.log(1. + np.exp(x))


class SoftMax(Activation):
    def forward(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        shift_x = x - np.max(x, axis=axis, keepdims=True)  # stable softmax
        exp = np.exp(shift_x + 1e-6)
        return exp / np.sum(exp, axis=axis, keepdims=True)


linear = Linear()
relu = ReLU()
leakyrelu = LeakyReLU()
elu = ELU()
tanh = Tanh()
sigmoid = Sigmoid()
softplus = SoftPlus()
softmax = SoftMax()

ACTIVATION_MAP = {
    "linear": linear,
    "relu": relu,
    "leakyrelu": leakyrelu,
    "elu": elu,
    "tanh": tanh,
    "sigmoid": sigmoid,
    "softplus": softplus,
    "softmax": softmax,
}
