"""Loss functions for regression. Each returns the mean over all elements."""

import numpy as np


class Loss:
    def __call__(self, pred: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError

    def grad(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """dL/d(pred)."""
        raise NotImplementedError


class MSE(Loss):
    def __call__(self, pred, target):
        return float(np.mean((pred - target) ** 2))

    def grad(self, pred, target):
        return 2.0 * (pred - target) / pred.size


class MAE(Loss):
    def __call__(self, pred, target):
        return float(np.mean(np.abs(pred - target)))

    def grad(self, pred, target):
        return np.sign(pred - target) / pred.size


class Huber(Loss):
    """Quadratic for small errors, linear for large ones: robust to outliers."""

    def __init__(self, delta: float = 1.0) -> None:
        self.delta = delta

    def __call__(self, pred, target):
        err = np.abs(pred - target)
        quad = np.minimum(err, self.delta)
        return float(np.mean(0.5 * quad**2 + self.delta * (err - quad)))

    def grad(self, pred, target):
        return np.clip(pred - target, -self.delta, self.delta) / pred.size
