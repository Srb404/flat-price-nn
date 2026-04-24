# flat-price-nn

Apartment price estimator for Polish cities: a neural network built from scratch in NumPy (no PyTorch, no TensorFlow).

> Work in progress.

## Goal

Given an apartment's parameters (city, area, rooms, floor, year built, ...) estimate its market price.
The network, backpropagation and optimizers are implemented by hand to show how they work under the hood.

## Data

[Apartment Prices in Poland](https://www.kaggle.com/datasets/krzysztofjamroz/apartment-prices-in-poland) (Kaggle), listings from Otodom.
Download the CSV files into `data/raw/`.

## Project structure

```
nn/            # neural network library from scratch: layers, activations, losses, optimizers, model
flat_price/    # data loading and feature engineering for the apartment dataset
tests/         # unit tests (incl. gradient checking)
notebooks/     # exploration and visualizations
train.py       # training entry point
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python train.py --epochs 50
```

## Results

_TBD: MAE / RMSE on the test set, comparison with linear regression and a PyTorch baseline._

## License

MIT
