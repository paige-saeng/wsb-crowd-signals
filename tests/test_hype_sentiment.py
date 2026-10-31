import numpy as np
import pandas as pd

from src.analysis.event_study import WINDOWS
from src.analysis.hype_sentiment import GROUPS, attach_sentiment, by_tercile, slope


def _cars(effect, n=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-02-01", periods=120)
    sent = rng.normal(0.2, 0.25, n)
    car = pd.DataFrame({"ticker": [f"T{i % 50}" for i in range(n)],
                        "event_date": rng.choice(dates, n)})
    for w in WINDOWS:
        car[w] = rng.normal(0, 0.08, n)
    car["month (+1,+20)"] += effect * (sent - sent.mean())
    daily = pd.DataFrame({"ticker": car["ticker"], "trading_day": car["event_date"],
                          "sent_net": sent, "n_scored": 20})
    return car, daily.drop_duplicates(["ticker", "trading_day"])


def test_attach_drops_thin_days():
    car, daily = _cars(0.0)
    daily.loc[daily.index[:10], "n_scored"] = 2
    out = attach_sentiment(car, daily, min_texts=5)
    assert out["n_scored"].min() >= 5 and len(out) < len(car)


def test_finds_planted_slope_and_ordered_terciles():
    car, daily = _cars(effect=-0.3)               # more bullish spike -> bigger give-back
    c = attach_sentiment(car, daily)
    s = slope(c)
    assert s["coef"] < 0 and s["t"] < -3
    t = by_tercile(c).query("window == 'month (+1,+20)'").set_index("group")["mean"]
    assert t[GROUPS[2]] < t[GROUPS[0]]


def test_no_slope_without_effect():
    car, daily = _cars(effect=0.0, seed=4)
    assert abs(slope(attach_sentiment(car, daily))["t"]) < 3
