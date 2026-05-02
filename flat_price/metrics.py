"""Regression metrics on prices in PLN."""

import numpy as np


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    ape = np.abs(err) / y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE": float(np.mean(ape) * 100),
        "MdAPE": float(np.median(ape) * 100),
        "within_10pct": float(np.mean(ape <= 0.10) * 100),
        "R2": float(1 - np.sum(err**2) / np.sum((y_true - y_true.mean()) ** 2)),
    }
