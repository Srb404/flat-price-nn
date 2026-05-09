"""Simple reference models the network has to beat."""

import numpy as np
import pandas as pd


class CityPricePerM2:
    """Median price per square metre in the city times the apartment area."""

    def fit(self, df: pd.DataFrame) -> "CityPricePerM2":
        self.ppm = (df["price"] / df["squareMeters"]).groupby(df["city"]).median()
        self.fallback = float((df["price"] / df["squareMeters"]).median())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ppm = df["city"].map(self.ppm).fillna(self.fallback).to_numpy()
        return ppm * df["squareMeters"].to_numpy()


class LinearRegression:
    """Least squares on the same encoded features, predicting log(price)."""

    def fit(self, x: np.ndarray, prices: np.ndarray) -> "LinearRegression":
        x1 = np.hstack([x, np.ones((len(x), 1))])
        self.coef, *_ = np.linalg.lstsq(x1, np.log(prices), rcond=None)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.exp(np.hstack([x, np.ones((len(x), 1))]) @ self.coef)
