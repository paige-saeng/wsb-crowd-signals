"""Event study tests on simulated stocks with a planted (or absent) reversal."""

import numpy as np
import pandas as pd
import pytest

from src.finance.returns import market_model, abnormal_returns
from src.analysis.event_study import find_events, event_returns, cars, run


def sim_panel(n_tickers=60, n_days=420, jump=0.10, reversal=0.01, rev_days=10, seed=0):
    """Each stock: beta * market + noise. Three spikes per stock; on each spike
    day price jumps `jump`, then falls `reversal` per day for `rev_days` days."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-06-01", periods=n_days)
    mkt = rng.normal(0.0004, 0.01, n_days)
    frames = []
    for k in range(n_tickers):
        beta = rng.uniform(0.5, 2.0)
        ret = beta * mkt + rng.normal(0, 0.02, n_days)
        z = rng.normal(0, 0.8, n_days)
        mentions = rng.poisson(20, n_days).astype(float)
        spikes = rng.choice(np.arange(280, n_days - 25, 30), size=3, replace=False)
        for d in spikes:
            z[d], mentions[d] = 6.0, 200
            ret[d] += jump
            ret[d + 1:d + 1 + rev_days] -= reversal
        frames.append(pd.DataFrame({"ticker": f"T{k:02d}", "date": dates, "ret": ret, "mkt_ret": mkt,
                                    "abn_attn_z": np.minimum(z, 2.5) * (z < 6) + z * (z >= 6),
                                    "mentions": mentions}))
    return pd.concat(frames, ignore_index=True)


def test_market_model_recovers_beta():
    rng = np.random.default_rng(1)
    m = rng.normal(0, 0.01, 300)
    r = 0.001 + 1.7 * m + rng.normal(0, 0.002, 300)
    a, b = market_model(r, m)
    assert b == pytest.approx(1.7, abs=0.05) and a == pytest.approx(0.001, abs=0.0005)
    assert np.abs(abnormal_returns(r, m, a, b)).mean() < 0.003
    assert market_model(r[:50], m[:50]) is None          # too little data


def test_events_declustered_and_thresholded():
    p = sim_panel(n_tickers=2)
    ev = find_events(p)
    assert len(ev) == 6                                  # 3 planted spikes x 2 stocks, nothing else
    # clean series: a 3-day episode (days 100-102), a spike 10 days later (merged),
    # a spike 40 days after that (new event), and a spike with too few mentions
    g = p[p.ticker == "T00"].copy().reset_index(drop=True)
    g["abn_attn_z"], g["mentions"] = 0.0, 20.0
    for d in (100, 101, 102, 112, 152):
        g.loc[d, ["abn_attn_z", "mentions"]] = [6.0, 200]
    g.loc[250, ["abn_attn_z", "mentions"]] = [6.0, 5]      # under the 10-mention minimum
    ev2 = find_events(g)
    assert ev2["date"].tolist() == [g.at[100, "date"], g.at[152, "date"]]


def test_estimation_window_is_before_event():
    p = sim_panel(n_tickers=1)
    ev = find_events(p)
    er = event_returns(p, ev.head(1))
    # beta estimate must not be contaminated by the +10% event: close to a clean estimate
    assert er["rel_day"].min() == -10 and er["rel_day"].max() == 20


def test_planted_reversal_is_recovered():
    p = sim_panel(jump=0.10, reversal=0.01, rev_days=10, seed=3)
    events, er, table, path = run(p)
    t = table[table.group == "all events"].set_index("window")
    assert t.loc["day 0", "mean"] == pytest.approx(0.10, abs=0.01)
    assert t.loc["month (+1,+20)", "mean"] == pytest.approx(-0.10, abs=0.02)
    assert t.loc["month (+1,+20)", "t"] < -5
    assert abs(t.loc["pre (-10,-1)", "t"]) < 3            # nothing planted before the spike
    caar = path[path.group == "all events"].set_index("rel_day")["caar"]
    assert caar.loc[0] == pytest.approx(0.10, abs=0.015) and caar.loc[20] == pytest.approx(0.0, abs=0.025)


def test_no_false_reversal():
    p = sim_panel(jump=0.10, reversal=0.0, seed=4)
    _, _, table, _ = run(p)
    t = table[table.group == "all events"].set_index("window")
    assert abs(t.loc["month (+1,+20)", "t"]) < 3
    assert abs(t.loc["week (+1,+5)", "t"]) < 3


def test_groups_split_by_day0_direction():
    p = sim_panel(seed=5)
    _, _, table, _ = run(p)
    groups = set(table["group"])
    assert {"price up on day 0", "price down on day 0", "without GME and AMC"} <= groups


def test_alpha_bias_is_avoided():
    """A stock that rallied in the estimation window has a large positive alpha.
    The beta-only main model must not turn that into a fake post-event decline;
    the full market model does, which is why it is only a labeled variant."""
    p = sim_panel(jump=0.10, reversal=0.0, seed=6)
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)
    day = p.groupby("ticker").cumcount()
    p.loc[day < 270, "ret"] += 0.004          # +0.4%/day rally before the events
    _, _, table, _ = run(p)
    main = table[table.group == "all events"].set_index("window")
    biased = table[table.group == "all events, full market model (alpha-biased)"].set_index("window")
    assert abs(main.loc["month (+1,+20)", "t"]) < 3
    assert biased.loc["month (+1,+20)", "mean"] < -0.04
