import pandas as pd
import pytest

from flat_price.location import LOCATION_COLUMNS, build_location_table, fill_location


def make_df():
    rows = []
    for dist, lat in [(0.5, 50.0), (2.5, 51.0)]:
        for _ in range(25):
            rows.append(
                {"city": "x", "centreDistance": dist, **dict.fromkeys(LOCATION_COLUMNS, lat)}
            )
    rows.append({"city": "y", "centreDistance": 1.0, **dict.fromkeys(LOCATION_COLUMNS, 0.0)})
    return pd.DataFrame(rows)


def test_fill_location_uses_nearest_bin_and_drops_sparse_cities():
    table = build_location_table(make_df())
    assert set(table["city"]) == {"x"}
    assert fill_location("x", 0.2, table)["latitude"] == 50.0
    assert fill_location("x", 9.0, table)["latitude"] == 51.0
    with pytest.raises(ValueError):
        fill_location("y", 1.0, table)
