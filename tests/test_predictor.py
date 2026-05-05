import numpy as np

from flat_price.features import FeatureEncoder
from flat_price.predictor import PricePredictor
from tests.test_features import make_df


def test_save_load_roundtrip_with_dropout(tmp_path):
    df = make_df()
    df["price"] = np.linspace(3e5, 9e5, len(df))
    predictor = PricePredictor(FeatureEncoder().fit(df), hidden=[8, 4], dropout=0.2, seed=0)
    predictor.fit_target(df["price"].to_numpy())
    predictor.latest_month_index = 7
    predictor.save(tmp_path)

    loaded = PricePredictor.load(tmp_path)
    assert np.allclose(loaded.predict(df), predictor.predict(df))
    assert loaded.latest_month_index == 7


def test_target_encoding_roundtrip():
    df = make_df()
    predictor = PricePredictor(FeatureEncoder().fit(df), hidden=[4])
    prices = np.array([2e5, 5e5, 1.2e6])
    predictor.fit_target(prices)
    encoded = predictor.encode_target(prices)
    assert np.allclose(np.exp(encoded[:, 0] * predictor.target_std + predictor.target_mean), prices)
