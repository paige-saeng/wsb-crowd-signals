import numpy as np
import pandas as pd

from src.data.prices import add_returns, coverage_report


def sample():
    return pd.DataFrame({
        "date": pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-04", "2021-01-05"]),
        "ticker": ["GME", "GME", "SPY", "SPY"],
        "adj_close": [20.0, 25.0, 100.0, 99.0],
        "volume": [1, 2, 3, 4],
    })


def test_returns_per_ticker_no_leakage():
    df = add_returns(sample()).set_index(["ticker", "date"])
    # First day of each ticker has no previous price; SPY must not use GME's last price
    assert np.isnan(df.loc[("GME", "2021-01-04"), "ret"])
    assert np.isnan(df.loc[("SPY", "2021-01-04"), "ret"])
    assert np.isclose(df.loc[("GME", "2021-01-05"), "ret"], 0.25)
    assert np.isclose(df.loc[("SPY", "2021-01-05"), "log_ret"], np.log(0.99))


def test_coverage_flags_missing():
    rep = coverage_report(sample(), ["GME", "SPY", "BBBY"])
    assert rep.loc["BBBY", "rows"] == 0
    assert rep.loc["GME", "rows"] == 2
