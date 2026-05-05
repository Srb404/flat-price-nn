"""Price predictor: feature encoder + network + target scaling, saved as one directory."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from flat_price.features import FeatureEncoder
from nn import Dense, Dropout, ReLU, Sequential


def build_network(n_in: int, hidden: list[int], dropout: float = 0.0, seed: int = 0) -> Sequential:
    rng = np.random.default_rng(seed)
    layers, prev = [], n_in
    for size in hidden:
        layers += [Dense(prev, size, rng), ReLU()]
        if dropout:
            layers.append(Dropout(dropout, rng))
        prev = size
    layers.append(Dense(prev, 1, rng))
    return Sequential(layers)


class PricePredictor:
    """The network predicts standardized log(price); this class hides that from callers."""

    def __init__(
        self, encoder: FeatureEncoder, hidden: list[int], dropout: float = 0.0, seed: int = 0
    ) -> None:
        self.encoder = encoder
        self.hidden = hidden
        self.dropout = dropout
        self.network = build_network(len(encoder.feature_names), hidden, dropout, seed)
        self.target_mean = 0.0
        self.target_std = 1.0
        # month of the newest listings, used when pricing an apartment "today"
        self.latest_month_index = 0

    def fit_target(self, prices: np.ndarray) -> None:
        log_price = np.log(prices)
        self.target_mean, self.target_std = float(log_price.mean()), float(log_price.std())

    def encode_target(self, prices: np.ndarray) -> np.ndarray:
        return ((np.log(prices) - self.target_mean) / self.target_std)[:, None]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = self.network.predict(self.encoder.transform(df))[:, 0]
        return np.exp(out * self.target_std + self.target_mean)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.encoder.save(directory / "encoder.json")
        self.network.save(directory / "weights.npz")
        meta = {
            "hidden": self.hidden,
            "dropout": self.dropout,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "latest_month_index": self.latest_month_index,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "PricePredictor":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        predictor = cls(
            FeatureEncoder.load(directory / "encoder.json"), meta["hidden"], meta["dropout"]
        )
        predictor.network.load(directory / "weights.npz")
        predictor.target_mean, predictor.target_std = meta["target_mean"], meta["target_std"]
        predictor.latest_month_index = meta["latest_month_index"]
        return predictor
