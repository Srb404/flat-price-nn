"""Feature engineering: encoding categories, scaling numeric columns.

All statistics (medians, means, stds, category lists) are fitted on the training set
only and can be saved to JSON, so the demo applies exactly the same transformation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = [
    "squareMeters",
    "rooms",
    "floor",
    "floorCount",
    "buildYear",
    "latitude",
    "longitude",
    "centreDistance",
    "monthIndex",
]
# Right-skewed columns, log1p makes their distribution closer to normal
LOG_NUMERIC = [
    "poiCount",
    "schoolDistance",
    "clinicDistance",
    "postOfficeDistance",
    "kindergartenDistance",
    "restaurantDistance",
    "collegeDistance",
    "pharmacyDistance",
]
# Missing values carry information (e.g. no build year often means an old tenement)
MISSING_FLAGS = ["floor", "buildYear"]
CATEGORICAL = [
    "city",
    "type",
    "ownership",
    "buildingMaterial",
    "condition",
    "hasParkingSpace",
    "hasBalcony",
    "hasElevator",
    "hasSecurity",
    "hasStorageRoom",
]
MISSING = "unknown"


class FeatureEncoder:
    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        num = self._numeric(df)
        self.medians = num.median().to_dict()
        num = num.fillna(self.medians)
        self.means = num.mean().to_dict()
        self.stds = num.std().replace(0, 1).to_dict()
        self.categories = {
            col: sorted(df[col].fillna(MISSING).astype(str).unique()) for col in CATEGORICAL
        }
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num = self._numeric(df).fillna(self.medians)
        num = (num - pd.Series(self.means)) / pd.Series(self.stds)
        parts = [num.to_numpy(float)]
        parts += [df[col].isna().to_numpy(float)[:, None] for col in MISSING_FLAGS]
        for col, cats in self.categories.items():
            values = df[col].fillna(MISSING).astype(str).to_numpy()
            parts.append((values[:, None] == np.array(cats)[None, :]).astype(float))
        return np.hstack(parts)

    @property
    def feature_names(self) -> list[str]:
        names = NUMERIC + [f"log_{c}" for c in LOG_NUMERIC]
        names += [f"{c}_missing" for c in MISSING_FLAGS]
        names += [f"{col}={cat}" for col, cats in self.categories.items() for cat in cats]
        return names

    @staticmethod
    def _numeric(df: pd.DataFrame) -> pd.DataFrame:
        num = df[NUMERIC].astype(float).copy()
        for col in LOG_NUMERIC:
            num[f"log_{col}"] = np.log1p(df[col].astype(float))
        return num

    def save(self, path: str | Path) -> None:
        state = {k: getattr(self, k) for k in ("medians", "means", "stds", "categories")}
        Path(path).write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FeatureEncoder":
        enc = cls()
        for key, value in json.loads(Path(path).read_text()).items():
            setattr(enc, key, value)
        return enc
