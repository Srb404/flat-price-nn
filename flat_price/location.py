"""Typical location features per city and distance from the centre.

The demo asks only for the city and the distance to the centre; coordinates and
distances to points of interest are filled in with medians of similar listings.
"""

from pathlib import Path

import numpy as np
import pandas as pd

LOCATION_COLUMNS = [
    "latitude",
    "longitude",
    "poiCount",
    "schoolDistance",
    "clinicDistance",
    "postOfficeDistance",
    "kindergartenDistance",
    "restaurantDistance",
    "collegeDistance",
    "pharmacyDistance",
]
BIN_KM = 1.0
MIN_LISTINGS = 20


def build_location_table(df: pd.DataFrame) -> pd.DataFrame:
    binned = df.assign(distanceBin=np.floor(df["centreDistance"] / BIN_KM).astype(int))
    groups = binned.groupby(["city", "distanceBin"])
    table = groups[LOCATION_COLUMNS].median().round(6)
    table["listings"] = groups.size()
    return table[table["listings"] >= MIN_LISTINGS].reset_index()


def fill_location(city: str, centre_distance: float, table: pd.DataFrame) -> dict[str, float]:
    """Location features of the nearest distance bin available for the city."""
    rows = table[table["city"] == city]
    if rows.empty:
        raise ValueError(f"Unknown city: {city}")
    target = np.floor(centre_distance / BIN_KM)
    row = rows.iloc[int(np.argmin(np.abs(rows["distanceBin"] - target)))]
    return {col: float(row[col]) for col in LOCATION_COLUMNS}


def save_location_table(table: pd.DataFrame, path: str | Path) -> None:
    table.to_csv(path, index=False)


def load_location_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
