"""Optimizers updating layer parameters in place."""

import numpy as np

from nn.layers import Layer


class Optimizer:
    def __init__(self, lr: float, weight_decay: float = 0.0) -> None:
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, layers: list[Layer]) -> None:
        for layer in layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                if self.weight_decay and name == "W":
                    grad = grad + self.weight_decay * param
                param -= self._update((id(layer), name), grad)

    def _update(self, key: tuple[int, str], grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0) -> None:
        super().__init__(lr, weight_decay)
        self.momentum = momentum
        self._velocity: dict[tuple[int, str], np.ndarray] = {}

    def _update(self, key, grad):
        v = self.momentum * self._velocity.get(key, 0.0) + grad
        self._velocity[key] = v
        return self.lr * v


class Adam(Optimizer):
    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(lr, weight_decay)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self._m: dict[tuple[int, str], np.ndarray] = {}
        self._v: dict[tuple[int, str], np.ndarray] = {}
        self._t: dict[tuple[int, str], int] = {}

    def _update(self, key, grad):
        t = self._t.get(key, 0) + 1
        m = self.beta1 * self._m.get(key, 0.0) + (1 - self.beta1) * grad
        v = self.beta2 * self._v.get(key, 0.0) + (1 - self.beta2) * grad**2
        self._t[key], self._m[key], self._v[key] = t, m, v
        m_hat = m / (1 - self.beta1**t)
        v_hat = v / (1 - self.beta2**t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
