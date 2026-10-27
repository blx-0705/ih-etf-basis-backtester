import numpy as np
import pandas as pd
import pytest

from basis_bt.cleaning import level_columns


def make_ticks(rows, n_levels=1):
    """rows: list of dicts with datetime, last, volume, amount, bid, ask, bid_qty, ask_qty."""
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["seq"] = np.arange(len(df))
    out = {"datetime": df["datetime"], "seq": df["seq"], "last": df["last"].astype(float),
           "volume": df["volume"].astype(float), "amount": df["amount"].astype(float)}
    for c in level_columns(n_levels):
        out[c] = np.nan
    out = pd.DataFrame(out)
    out["bid_px_1"], out["ask_px_1"] = df["bid"], df["ask"]
    out["bid_qty_1"], out["ask_qty_1"] = df.get("bid_qty", 10), df.get("ask_qty", 10)
    return out


@pytest.fixture(scope="session")
def synthetic_raw(tmp_path_factory):
    from basis_bt.synthetic import generate
    return generate(tmp_path_factory.mktemp("raw"), n_days=3, seed=11)
