"""Daily sentiment aggregation, text scoring and a known-answer check for H3."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.attention import fit
from src.analysis.direction import add_sentiment, h3_specs
from src.features.panel import build_panel
from src.features.sentiment_daily import daily_sentiment, score_texts
from tests.test_panel_attention import make_market


def test_daily_sentiment_averages_and_single_stock():
    mentions = pd.DataFrame({"id": ["a", "b", "b", "c"], "ticker": ["GME", "GME", "AMC", "GME"],
                             "trading_day": ["2021-03-01"] * 4})
    scores = pd.DataFrame({"id": ["a", "b", "c"], "p_bullish": [0.8, 0.1, 0.3],
                           "p_bearish": [0.1, 0.7, 0.2], "p_neutral": [0.1, 0.2, 0.5]})
    d = daily_sentiment(mentions, scores).set_index("ticker")
    assert d.loc["GME", "n_scored"] == 3 and d.loc["AMC", "n_scored"] == 1
    assert d.loc["GME", "sent_net"] == pytest.approx(np.mean([0.7, -0.6, 0.1]))
    assert d.loc["GME", "sent_net_single"] == pytest.approx(np.mean([0.7, 0.1]))   # b names two stocks
    assert d.loc["GME", "n_single"] == 2
    assert d.loc["GME", "bullishness"] == pytest.approx(np.log(2 / 2))           # 1 bull, 1 bear, 1 neutral
    assert np.isnan(d.loc["AMC", "sent_net_single"])


def test_score_texts_streams_and_resumes(tmp_path):
    interim = tmp_path / "interim"
    interim.mkdir()
    cols = dict(kind="comment", title=None)
    pd.DataFrame({"id": ["p1"], "kind": "post", "title": ["GME moon"], "text": ["[deleted]"]}).to_parquet(
        interim / "posts.parquet")
    pd.DataFrame({"id": ["c1", "c2", "c3"], "text": ["buy", "sell", "meh"], **cols}).to_parquet(
        interim / "comments.parquet")

    class Fake:
        classes_ = np.array(["bearish", "bullish", "neutral"])

        def predict_proba(self, texts):
            return np.array([[0.1, 0.8, 0.1] if "moon" in t or "buy" in t else [0.6, 0.2, 0.2] for t in texts])

    parts = tmp_path / "parts"
    s = score_texts(Fake(), ["p1", "c1", "c2"], interim, parts, batch_size=2, log=lambda *_: None)
    assert sorted(s["id"]) == ["c1", "c2", "p1"]
    assert s.set_index("id").loc["p1", "p_bullish"] == pytest.approx(0.8)       # title is scored for posts
    assert s.set_index("id").loc["c2", "p_bearish"] == pytest.approx(0.6)
    n_parts = len(list(parts.glob("*.parquet")))
    s2 = score_texts(Fake(), ["p1", "c1", "c2"], interim, parts, batch_size=2, log=lambda *_: None)
    assert len(list(parts.glob("*.parquet"))) == n_parts and len(s2) == 3      # resumed, nothing redone


def _sim(signal: float, seed: int = 0):
    prices, mentions = make_market(n_tickers=40, n_days=260, seed=seed)
    panel = build_panel(mentions, prices)
    rng = np.random.default_rng(seed + 1)
    nxt = panel.groupby("ticker")["ret"].shift(-1)
    sd = nxt.std()
    sent = signal * nxt / sd + rng.normal(0, 1, len(panel))       # sentiment that knows tomorrow (or not)
    daily = pd.DataFrame({"ticker": panel["ticker"], "trading_day": panel["date"], "n_scored": 10,
                          "n_single": 10, "sent_net": sent, "bullishness": sent, "sent_net_single": sent,
                          "bull_share": 0.3, "bear_share": 0.1}).dropna()
    return add_sentiment(panel, daily, min_texts=5)


def test_h3_finds_planted_direction_signal():
    df = _sim(signal=0.3)
    r = fit(df, h3_specs()[0])
    assert r["coef"] > 0 and r["t"] > 4


def test_h3_finds_nothing_without_signal():
    df = _sim(signal=0.0, seed=3)
    r = fit(df, h3_specs()[0])
    assert abs(r["t"]) < 3


def test_thin_days_are_dropped():
    df = _sim(signal=0.0)
    df2 = add_sentiment(df.drop(columns=["n_scored", "n_single", "sent_net", "bullishness", "sent_net_single",
                                         "bull_share", "bear_share", "ret_next", "xret_next", "ret_next5",
                                         "abn_sent"]),
                        pd.DataFrame({"ticker": ["T00"], "trading_day": [df["date"].iloc[50]], "n_scored": [2],
                                      "n_single": [2], "sent_net": [0.5], "bullishness": [0.5],
                                      "sent_net_single": [0.5], "bull_share": [0.5], "bear_share": [0.0]}),
                        min_texts=5)
    assert df2["sent_net"].notna().sum() == 0
