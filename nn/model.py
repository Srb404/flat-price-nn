"""Sequential model: layer stack, training loop, predict, save/load."""

from pathlib import Path

import numpy as np

from nn.layers import Layer
from nn.losses import Loss
from nn.optimizers import Optimizer


class Sequential:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        return np.concatenate(
            [self.forward(x[i : i + batch_size]) for i in range(0, len(x), batch_size)]
        )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        loss: Loss,
        optimizer: Optimizer,
        epochs: int = 10,
        batch_size: int = 64,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
        patience: int | None = None,
        rng: np.random.Generator | None = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Mini-batch training. With `patience`, stops early and restores the best weights."""
        rng = rng or np.random.default_rng()
        history: dict[str, list[float]] = {"loss": [], "val_loss": []}
        best_val, best_params, bad_epochs = np.inf, None, 0

        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(x))
            total = 0.0
            for i in range(0, len(x), batch_size):
                idx = order[i : i + batch_size]
                pred = self.forward(x[idx], training=True)
                total += loss(pred, y[idx]) * len(idx)
                self.backward(loss.grad(pred, y[idx]))
                optimizer.step(self.layers)
            history["loss"].append(total / len(x))

            msg = f"epoch {epoch:3d}  loss {history['loss'][-1]:.4f}"
            if validation is not None:
                val = loss(self.predict(validation[0]), validation[1])
                history["val_loss"].append(val)
                msg += f"  val_loss {val:.4f}"
                if patience is not None:
                    if val < best_val:
                        best_val, best_params, bad_epochs = val, self.get_params(), 0
                    else:
                        bad_epochs += 1
                        if bad_epochs >= patience:
                            if verbose:
                                print(msg + "  early stop")
                            break
            if verbose:
                print(msg)

        if best_params is not None:
            self.set_params(best_params)
        return history

    def get_params(self) -> dict[str, np.ndarray]:
        return {
            f"{i}.{name}": p.copy()
            for i, layer in enumerate(self.layers)
            for name, p in layer.params.items()
        }

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        for key, value in params.items():
            i, name = key.split(".", 1)
            self.layers[int(i)].params[name][...] = value

    def save(self, path: str | Path) -> None:
        np.savez(path, **self.get_params())

    def load(self, path: str | Path) -> None:
        with np.load(path) as data:
            self.set_params(dict(data))
