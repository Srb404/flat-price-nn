"""Layers with forward and backward pass."""

import numpy as np


class Layer:
    """Base class. Layers with weights keep them in `params` and gradients in `grads`."""

    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Take dL/d(output), store gradients of params, return dL/d(input)."""
        raise NotImplementedError


class Dense(Layer):
    """Fully connected layer: y = x @ W + b."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator | None = None) -> None:
        super().__init__()
        rng = rng or np.random.default_rng()
        # He initialization, suited for ReLU networks
        self.params["W"] = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
        self.params["b"] = np.zeros(n_out)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        self._x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.grads["W"] = self._x.T @ grad
        self.grads["b"] = grad.sum(axis=0)
        return grad @ self.params["W"].T


class Dropout(Layer):
    """Inverted dropout: zeroes a fraction `p` of activations during training only."""

    def __init__(self, p: float, rng: np.random.Generator | None = None) -> None:
        super().__init__()
        self.p = p
        self.rng = rng or np.random.default_rng()
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        if not training or self.p == 0:
            self._mask = None
            return x
        self._mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * self._mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad if self._mask is None else grad * self._mask
