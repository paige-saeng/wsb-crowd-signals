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


# ---- recovery of tickers Yahoo no longer lists

def fake_downloader(available, flaky=()):
    """Mimics yf: batch calls drop `flaky` symbols; single calls return them."""
    def dl(tickers, start, end):
        rows = []
        for t in tickers:
            if t not in available or (t in flaky and len(tickers) > 1):
                continue
            first = available[t]
            for d in pd.bdate_range(first, end)[:5]:
                rows.append({"date": d, "ticker": t, "close": 1.0, "adj_close": 1.0, "volume": 10})
        return pd.DataFrame(rows, columns=["date", "ticker", "close", "adj_close", "volume"])
    return dl


def test_download_with_recovery():
    from src.data.prices import download_with_recovery
    # LAAC exists but only from mid-2021: inside the window, after the cutoff
    available = {"GME": "2020-06-01", "LC": "2020-06-01", "META": "2020-06-01", "LAAC": "2021-09-01"}
    renames = {"FB": "META", "LAC": "LAAC", "BBBY": "BBBYQ"}
    df, rep = download_with_recovery(
        ["GME", "LC", "FB", "LAC", "BBBY", "TWTR"], "2020-06-01", "2022-02-01",
        renames=renames, require_by="2021-02-01", tolerance_days=0,
        downloader=fake_downloader(available, flaky={"LC"}))
    st = dict(zip(rep["ticker"], rep["status"]))
    assert st["GME"] == "ok"
    assert st["LC"] == "ok on retry"
    assert st["FB"] == "recovered via META"
    assert st["LAC"] == "missing (LAAC history starts 2021-09-01, after 2021-02-01)"  # partial: rejected
    assert st["BBBY"] == "missing (renamed BBBYQ has no data)"
    assert st["TWTR"] == "missing (no rename known)"
    fb = df[df["ticker"] == "FB"]
    assert len(fb) and set(fb["source_symbol"]) == {"META"}           # labeled with the 2021 symbol
    assert "META" not in set(df["ticker"])                             # not double-counted
    assert set(df["ticker"]) == {"GME", "LC", "FB"}


def test_rename_file_loads():
    from src.data.prices import load_renames
    r = load_renames("data/reference/ticker_renames.csv")
    assert r["FB"] == "META" and r["CCIV"] == "LCID" and r["WISH"] == "LOGC"


def test_rename_judged_from_listing_date():
    """DWAC listed in Sept 2021; DJT history starting then is complete for it."""
    from src.data.prices import download_with_recovery
    available = {"DJT": "2021-09-30", "PARA": "2021-02-12", "LATE": "2021-12-01"}
    renames = {"DWAC": "DJT", "VIAC": "PARA", "OLD": "LATE"}
    first_seen = {"DWAC": "2021-10-01", "VIAC": "2021-01-30", "OLD": "2021-01-30"}
    df, rep = download_with_recovery(["DWAC", "VIAC", "OLD"], "2020-06-01", "2022-02-01", renames=renames,
                                     require_by="2021-02-01", first_seen=first_seen, tolerance_days=30,
                                     downloader=fake_downloader(available))
    st = dict(zip(rep["ticker"], rep["status"]))
    assert st["DWAC"] == "recovered via DJT"        # judged from its own Oct 2021 listing
    assert st["VIAC"] == "recovered via PARA"       # 2 weeks late, inside the 30-day tolerance
    assert st["OLD"].startswith("missing (LATE history starts 2021-12-01")
