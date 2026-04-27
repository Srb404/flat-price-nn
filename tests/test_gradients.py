"""Gradient checking: analytic gradients from backward() vs. central finite differences."""

import numpy as np
import pytest

from nn import MAE, MSE, Dense, Huber, ReLU, Sequential, Sigmoid, Tanh

EPS = 1e-6


def numeric_grad(f, x):
    grad = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        orig = x[i]
        x[i] = orig + EPS
        plus = f()
        x[i] = orig - EPS
        minus = f()
        x[i] = orig
        grad[i] = (plus - minus) / (2 * EPS)
    return grad


def rel_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


@pytest.mark.parametrize("activation", [ReLU, Tanh, Sigmoid])
def test_network_param_and_input_gradients(activation):
    rng = np.random.default_rng(0)
    model = Sequential(
        [Dense(4, 5, rng), activation(), Dense(5, 3, rng), activation(), Dense(3, 1, rng)]
    )
    x = rng.normal(size=(6, 4))
    y = rng.normal(size=(6, 1))
    loss = MSE()

    def f():
        return loss(model.forward(x), y)

    pred = model.forward(x)
    grad_out = loss.grad(pred, y)
    grad_in = grad_out
    for layer in reversed(model.layers):
        grad_in = layer.backward(grad_in)

    for layer in model.layers:
        for name, param in layer.params.items():
            assert rel_error(layer.grads[name], numeric_grad(f, param)) < 1e-5, name
    assert rel_error(grad_in, numeric_grad(f, x)) < 1e-5


@pytest.mark.parametrize("loss", [MSE(), MAE(), Huber(delta=0.5)])
def test_loss_gradients(loss):
    rng = np.random.default_rng(1)
    pred = rng.normal(size=(8, 1))
    target = rng.normal(size=(8, 1))
    num = numeric_grad(lambda: loss(pred, target), pred)
    assert rel_error(loss.grad(pred, target), num) < 1e-5
