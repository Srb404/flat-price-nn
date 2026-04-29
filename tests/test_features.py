import numpy as np
import pandas as pd

from flat_price.data import load_listings, split
from flat_price.features import CATEGORICAL, LOG_NUMERIC, NUMERIC, FeatureEncoder


def make_df(n=20, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.uniform(1, 100, n) for col in NUMERIC + LOG_NUMERIC})
    for col in CATEGORICAL:
        df[col] = rng.choice(["a", "b", None], n)
    df.loc[0, "buildYear"] = np.nan
    return df


def test_encoder_shape_scaling_and_unknown_categories():
    train = make_df()
    enc = FeatureEncoder().fit(train)
    x = enc.transform(train)
    assert x.shape == (len(train), len(enc.feature_names))
    assert not np.isnan(x).any()
    assert np.allclose(x[:, : len(NUMERIC)].mean(axis=0), 0, atol=0.1)

    # a category unseen during fit gets all zeros in its one-hot block, not an error
    other = make_df(seed=1)
    other["city"] = "atlantis"
    city_cols = [i for i, n in enumerate(enc.feature_names) if n.startswith("city=")]
    assert enc.transform(other)[:, city_cols].sum() == 0


def test_encoder_save_load(tmp_path):
    df = make_df()
    enc = FeatureEncoder().fit(df)
    enc.save(tmp_path / "enc.json")
    assert np.allclose(FeatureEncoder.load(tmp_path / "enc.json").transform(df), enc.transform(df))


def test_load_listings_keeps_latest_snapshot(tmp_path):
    base = make_df(n=3)
    base["id"] = ["x", "y", "z"]
    base["price"] = [1, 2, 3]
    base.to_csv(tmp_path / "apartments_pl_2024_01.csv", index=False)
    base.assign(price=[10, 20, 30]).iloc[:2].to_csv(
        tmp_path / "apartments_pl_2024_02.csv", index=False
    )
    df = load_listings(tmp_path).set_index("id")
    assert df["price"].to_dict() == {"x": 10, "y": 20, "z": 3}
    assert df["monthIndex"].to_dict() == {"x": 1, "y": 1, "z": 0}


def test_split_is_disjoint_and_complete():
    df = pd.DataFrame({"id": range(100)})
    parts = split(df, seed=0)
    ids = np.concatenate([p["id"].to_numpy() for p in parts])
    assert sorted(ids) == list(range(100))
    assert [len(p) for p in parts] == [70, 15, 15]
