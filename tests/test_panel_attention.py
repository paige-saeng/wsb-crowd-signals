"""Panel timing tests and a known-answer simulation for the H1/H2 regressions."""

import numpy as np
import pandas as pd
import pytest

from src.features.panel import build_panel
from src.analysis.attention import Spec, H1_CONTROLS, H2_CONTROLS, fit


def make_market(n_tickers=40, n_days=260, effect=0.0, vol_effect=0.0, seed=0):
    """Simulated prices + mentions. Attention shocks raise NEXT-day |return|
    by `effect` and next-day log volume by `vol_effect` (both 0 = no signal)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-12-01", periods=n_days)
    rows, ment = [], []
    spy = rng.normal(0, 0.01, n_days)
    for k in range(n_tickers):
        t = f"T{k:02d}"
        base_sigma = rng.uniform(0.015, 0.04)
        base_m = rng.uniform(0.5, 3.0)
        shock = (rng.random(n_days) < 0.08) * rng.uniform(1, 3, n_days)   # attention spikes
        log_m = np.log1p(rng.poisson(np.exp(base_m))) + shock
        mentions = np.round(np.expm1(log_m))
        sigma = base_sigma * (1 + effect * np.r_[0, shock[:-1]])          # yesterday's shock
        ret = 0.8 * spy + rng.normal(0, 1, n_days) * sigma
        logv = 13 + rng.normal(0, 0.3, n_days) + vol_effect * np.r_[0, shock[:-1]]
        price = 20 * np.cumprod(1 + ret)
        rows.append(pd.DataFrame({"date": dates, "ticker": t, "adj_close": price, "volume": np.exp(logv)}))
        ment.append(pd.DataFrame({"trading_day": dates, "ticker": t, "mentions": mentions, "unique_authors": mentions}))
    spy_px = pd.DataFrame({"date": dates, "ticker": "SPY", "adj_close": 300 * np.cumprod(1 + spy), "volume": 1e8})
    prices = pd.concat(rows + [spy_px], ignore_index=True)
    mentions = pd.concat(ment, ignore_index=True)
    # Reddit coverage starts 2021-01-04: drop earlier mention rows like the real data
    mentions = mentions[mentions["trading_day"] >= "2021-01-04"]
    mentions = mentions[mentions["mentions"] > 0]
    return prices, mentions


# ---------------------------------------------------------------- timing

def test_timing_and_coverage():
    prices, mentions = make_market(n_tickers=2, n_days=120)
    p = build_panel(mentions, prices)
    a = p[p["ticker"] == "T00"].set_index("date")
    # before Reddit coverage: attention unknown, not zero
    assert a.loc[:"2021-01-01", "mentions"].isna().all()
    assert a.loc["2021-01-04":, "mentions"].notna().all()
    # baseline excludes day t: abn_attn_t = log_m_t - mean(log_m over t-30..t-1)
    t = a.index[70]
    prior = a["log_m"].iloc[40:70]
    assert a.loc[t, "abn_attn"] == pytest.approx(a.loc[t, "log_m"] - prior.mean())
    # targets come from the next row
    assert a["abs_ret_next"].iloc[50] == pytest.approx(abs(a["ret"].iloc[51]))
    assert pd.isna(a["abs_ret_next"].iloc[-1])
    # volume target baseline is known at t (window t-59..t), not t+1
    vb = a["log_vol"].iloc[11:71].mean()
    assert a["abn_vol_next"].iloc[70] == pytest.approx(a["log_vol"].iloc[71] - vb)
    # SPY is the market, not a study stock
    assert "SPY" not in set(p["ticker"])


def test_unknown_attention_never_becomes_a_spike():
    prices, mentions = make_market(n_tickers=3, n_days=120)
    p = build_panel(mentions, prices)
    first_days = p[(p["date"] >= "2021-01-04") & (p["date"] < "2021-02-01")]
    assert first_days["abn_attn"].isna().all()   # not enough covered history yet


# ---------------------------------------------------------------- known answers

@pytest.fixture(scope="module")
def panels():
    yes = build_panel(*reversed(make_market(effect=0.6, vol_effect=0.4, seed=1)))
    no = build_panel(*reversed(make_market(effect=0.0, vol_effect=0.0, seed=2)))
    return yes, no


def test_h1_detects_real_effect(panels):
    yes, _ = panels
    r = fit(yes, Spec("t", "H1", "abs_ret_next", "abn_attn", H1_CONTROLS))
    assert r["coef"] > 0 and r["t"] > 4


def test_h1_no_false_alarm(panels):
    _, no = panels
    r = fit(no, Spec("t", "H1", "abs_ret_next", "abn_attn", H1_CONTROLS))
    assert abs(r["t"]) < 3


def test_h2_detects_real_effect_and_no_false_alarm(panels):
    yes, no = panels
    assert fit(yes, Spec("t", "H2", "abn_vol_next", "abn_attn", H2_CONTROLS))["t"] > 4
    assert abs(fit(no, Spec("t", "H2", "abn_vol_next", "abn_attn", H2_CONTROLS))["t"]) < 3
