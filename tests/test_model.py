import numpy as np
import pytest

from nn import MSE, SGD, Adam, Dense, Dropout, ReLU, Sequential


def make_model(rng):
    return Sequential([Dense(2, 16, rng), ReLU(), Dense(16, 1, rng)])


@pytest.mark.parametrize("optimizer", [SGD(lr=0.05, momentum=0.9), Adam(lr=0.01)])
def test_fits_nonlinear_function(optimizer):
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(256, 2))
    y = x[:, :1] ** 2 + np.sin(3 * x[:, 1:])
    model = make_model(rng)
    before = MSE()(model.predict(x), y)
    model.fit(x, y, MSE(), optimizer, epochs=200, batch_size=32, rng=rng, verbose=False)
    after = MSE()(model.predict(x), y)
    assert after < 0.05 * before


def test_early_stopping_restores_best_weights():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 2))
    y = rng.normal(size=(64, 1))  # pure noise: validation loss will stop improving
    x_val, y_val = rng.normal(size=(64, 2)), rng.normal(size=(64, 1))
    model = make_model(rng)
    history = model.fit(
        x,
        y,
        MSE(),
        Adam(lr=0.05),
        epochs=500,
        validation=(x_val, y_val),
        patience=5,
        rng=rng,
        verbose=False,
    )
    assert len(history["val_loss"]) < 500
    assert np.isclose(MSE()(model.predict(x_val), y_val), min(history["val_loss"]))


def test_dropout_is_identity_at_inference():
    rng = np.random.default_rng(0)
    layer = Dropout(0.5, rng)
    x = rng.normal(size=(10, 10))
    assert np.array_equal(layer.forward(x, training=False), x)
    out = layer.forward(np.ones((1000, 100)), training=True)
    assert abs(out.mean() - 1.0) < 0.05


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    model, other = make_model(rng), make_model(rng)
    x = rng.normal(size=(5, 2))
    model.save(tmp_path / "w.npz")
    other.load(tmp_path / "w.npz")
    assert np.allclose(model.predict(x), other.predict(x))
