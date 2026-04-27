"""Activation functions as parameterless layers."""

import numpy as np

from nn.layers import Layer


class ReLU(Layer):
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._mask


class Tanh(Layer):
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        self._y = np.tanh(x)
        return self._y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (1.0 - self._y**2)


class Sigmoid(Layer):
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        self._y = 1.0 / (1.0 + np.exp(-x))
        return self._y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._y * (1.0 - self._y)
