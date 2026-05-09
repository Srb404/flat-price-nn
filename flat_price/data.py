"""Loading and cleaning the raw apartment listings dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_listings(raw_dir: str | Path = RAW_DIR) -> pd.DataFrame:
    """Load monthly sale snapshots and keep the latest snapshot of every listing.

    The same listing appears in several monthly files; without deduplication it would
    leak between the train and test sets.
    """
    files = sorted(Path(raw_dir).glob("apartments_pl_*.csv"))
    if not files:
        raise FileNotFoundError(f"No apartments_pl_*.csv files in {raw_dir}, see README")
    frames = [pd.read_csv(f).assign(month=f.stem.removeprefix("apartments_pl_")) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("month", kind="stable").drop_duplicates("id", keep="last")

    months = sorted(df["month"].unique())
    df["monthIndex"] = df["month"].map({m: i for i, m in enumerate(months)})
    return df.reset_index(drop=True)


def split(
    df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random train/validation/test split."""
    order = np.random.default_rng(seed).permutation(len(df))
    n_test, n_val = int(len(df) * test_frac), int(len(df) * val_frac)
    test = df.iloc[order[:n_test]]
    val = df.iloc[order[n_test : n_test + n_val]]
    train = df.iloc[order[n_test + n_val :]]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
