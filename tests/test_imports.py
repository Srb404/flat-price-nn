import importlib

import pytest

MODULES = [
    "nn.layers",
    "nn.activations",
    "nn.losses",
    "nn.optimizers",
    "nn.model",
    "flat_price.data",
    "flat_price.features",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)
