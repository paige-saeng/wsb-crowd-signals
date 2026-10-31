"""Earnings timing, flags, download recovery, and a known-answer check that the
earnings control removes an attention effect that is really earnings news."""

import numpy as np
import pandas as pd

from src.analysis.attention import H1_CONTROLS, Spec, earnings_specs, fit
from src.data.calendar import trading_sessions
from src.data.earnings import add_earnings_flags, download_earnings, reaction_days
from src.features.panel import build_panel


def test_reaction_day_follows_the_close_rule():
    e = pd.DataFrame({"ticker": ["A", "B", "C"], "announced": [
        "2021-05-04 07:00-04:00",     # before the open -> same day
        "2021-05-04 16:05-04:00",     # after the close -> next day
        "2021-05-07 16:30-04:00",     # Friday after close -> Monday
    ]})
    r = reaction_days(e).set_index("ticker")["date"]
    assert r["A"] == pd.Timestamp("2021-05-04")
    assert r["B"] == pd.Timestamp("2021-05-05")
    assert r["C"] == pd.Timestamp("2021-05-10")


def test_untimed_announcement_flags_two_days():
    e = pd.DataFrame({"ticker": ["A"], "announced": ["2021-05-04 00:00-04:00"]})
    r = reaction_days(e)
    assert list(r["date"]) == [pd.Timestamp("2021-05-04"), pd.Timestamp("2021-05-05")]


def test_flags_and_uncovered_stocks():
    dates = pd.bdate_range("2021-03-01", periods=10)
    panel = pd.DataFrame({"ticker": ["A"] * 10 + ["Z"] * 10, "date": list(dates) * 2})
    days = pd.DataFrame({"ticker": ["A"], "date": [dates[5]]})
    f = add_earnings_flags(panel, days, window=2).set_index(["ticker", "date"])
    assert f.loc[("A", dates[5]), "earn_today"] == 1 and f.loc[("A", dates[4]), "earn_next"] == 1
    assert f.loc[("A", dates[3]), "earn_window"] == 1 and f.loc[("A", dates[0]), "earn_window"] == 0
    assert f.loc["Z", "earn_today"].isna().all()          # no earnings data -> unknown, not zero


def test_download_recovers_renamed_symbol():
    def fetch(sym):
        if sym == "NEW":
            return pd.DataFrame({"announced": pd.to_datetime(["2021-08-01 16:05", "2019-01-01 16:05"], utc=True)})
        raise ValueError("no data")
    earn, status = download_earnings(["OLD", "GONE"], "2021-01-01", "2022-01-01", {"OLD": "NEW"},
                                     fetch=fetch, pause=0, log=None)
    assert list(earn["ticker"]) == ["OLD"] and len(earn) == 1          # 2019 date filtered out
    st = status.set_index("ticker")["status"]
    assert st["OLD"] == "recovered via NEW" and st["GONE"] == "missing"


def _earnings_market(seed=0, n_tickers=40, n_days=260):
    """Attention rises the day BEFORE earnings; earnings days move a lot.
    Attention itself has no effect, so a correct control should kill it."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-12-01", periods=n_days)
    spy = rng.normal(0, 0.01, n_days)
    rows, ment, earn = [], [], []
    for k in range(n_tickers):
        t = f"T{k:02d}"
        e_idx = np.arange(rng.integers(20, 60), n_days, 63)            # quarterly reaction days
        is_e = np.zeros(n_days, bool)
        is_e[e_idx] = True
        sigma = 0.02 * (1 + 3 * is_e)                                    # earnings days move 4x
        ret = 0.8 * spy + rng.normal(0, 1, n_days) * sigma
        pre = np.r_[is_e[1:], False]                                     # day before earnings
        log_m = np.log1p(rng.poisson(8, n_days)) + 2.0 * pre + rng.normal(0, 0.3, n_days)
        mentions = np.round(np.expm1(np.clip(log_m, 0, None)))
        rows.append(pd.DataFrame({"date": dates, "ticker": t, "adj_close": 20 * np.cumprod(1 + ret),
                                  "volume": np.exp(13 + rng.normal(0, 0.3, n_days))}))
        ment.append(pd.DataFrame({"trading_day": dates, "ticker": t, "mentions": mentions,
                                  "unique_authors": mentions}))
        earn.append(pd.DataFrame({"ticker": t, "date": dates[e_idx]}))
    spy_px = pd.DataFrame({"date": dates, "ticker": "SPY", "adj_close": 300 * np.cumprod(1 + spy), "volume": 1e8})
    m = pd.concat(ment, ignore_index=True)
    m = m[(m["trading_day"] >= "2021-01-04") & (m["mentions"] > 0)]
    return pd.concat(rows + [spy_px], ignore_index=True), m, pd.concat(earn, ignore_index=True)


def test_earnings_control_removes_spurious_attention_effect():
    prices, mentions, days = _earnings_market()
    panel = add_earnings_flags(build_panel(mentions, prices), days)
    specs = {s.name: s for s in earnings_specs()}
    raw = fit(panel, specs["H1 volatility [earnings-covered stocks]"])
    ctl = fit(panel, specs["H1 volatility + earnings-day controls"])
    excl = fit(panel, specs["H1 volatility [no earnings within 2 days]"])
    assert raw["t"] > 4                       # looks like attention predicts moves...
    assert abs(ctl["t"]) < 2.5                # ...but it was earnings
    assert abs(excl["t"]) < 2.5
