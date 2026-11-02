"""Known-answer checks for the H5 lead-lag (Granger-style) tests."""

import numpy as np
import pandas as pd

from src.analysis.leadlag import add_lags, granger


def _panel(price_leads=0.0, attn_leads=0.0, n_tickers=40, n_days=250, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    rows = []
    for k in range(n_tickers):
        ret = np.zeros(n_days)
        attn = np.zeros(n_days)
        for t in range(n_days):
            prev_attn = attn[t - 1] if t else 0.0
            prev_abs = abs(ret[t - 1]) if t else 0.0
            sigma = 0.02 * (1 + attn_leads * max(prev_attn, 0))
            ret[t] = rng.normal(0, sigma)
            attn[t] = 0.3 * prev_attn + price_leads * (prev_abs - 0.016) / 0.012 + rng.normal(0, 1)
        rows.append(pd.DataFrame({"ticker": f"T{k:02d}", "date": dates, "ret": ret, "abs_ret": np.abs(ret),
                                  "abn_attn": attn}))
    return pd.concat(rows, ignore_index=True)


def test_detects_price_leading_attention_only():
    df = add_lags(_panel(price_leads=0.5), ["abs_ret", "ret", "abn_attn"])
    p_price = granger(df, "abn_attn", "abs_ret")["p_joint"]
    p_attn = granger(df, "abs_ret", "abn_attn")["p_joint"]
    assert p_price < 1e-4
    assert p_attn > 0.01


def test_detects_attention_leading_volatility():
    df = add_lags(_panel(attn_leads=0.5, seed=1), ["abs_ret", "ret", "abn_attn"])
    r = granger(df, "abs_ret", "abn_attn")
    assert r["p_joint"] < 1e-4 and r["lag1_std"] > 0


def test_no_link_either_way():
    df = add_lags(_panel(seed=2), ["abs_ret", "ret", "abn_attn"])
    assert granger(df, "abn_attn", "abs_ret")["p_joint"] > 0.01
    assert granger(df, "abs_ret", "abn_attn")["p_joint"] > 0.01
