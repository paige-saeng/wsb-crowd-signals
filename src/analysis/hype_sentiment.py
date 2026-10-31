"""H3 x H4: does the tone of a WSB hype spike change what happens after it?

H4 found stocks lag by about 3.7% in the month after a hype spike. H3 found
that sentiment does not predict next-day direction on ordinary days. This
asks whether, among hype spikes, the bullish ones are followed by a bigger
give-back than the neutral or bearish ones.

Method
    - Events and cumulative abnormal returns (CARs) come from the H4 event
      study (data/processed/event_returns.parquet; beta-only abnormal returns).
    - Each event gets the day-0 WSB sentiment (sent_net, the mean of
      P(bullish) - P(bearish) over that day's texts) from
      data/processed/daily_sentiment.parquet. Events with fewer than
      `min_texts` scored texts are left out.
    - Events are split into thirds by day-0 sentiment (least to most
      bullish), and each third's mean CAR is tested with standard errors
      clustered by event date, as in H4.
    - A regression of the month CAR on day-0 sentiment (clustered by date)
      tests the slope directly, without choosing cut points.

The thirds are cut on the full sample of events, so they describe the data
rather than a tradable rule (a trader would not know the cut points in advance).

Usage
    python -m src.analysis.event_study          # first, if not yet run
    python -m src.analysis.hype_sentiment
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.event_study import WINDOWS, _winsor, cars, mean_test

MIN_TEXTS = 5
GROUPS = ("least bullish third", "middle third", "most bullish third")


def attach_sentiment(car: pd.DataFrame, daily: pd.DataFrame, min_texts: int = MIN_TEXTS) -> pd.DataFrame:
    """Add day-0 sentiment to each event's CARs; drop events with too few texts."""
    d = daily.rename(columns={"trading_day": "event_date"})[["ticker", "event_date", "sent_net", "n_scored"]].copy()
    d["event_date"] = pd.to_datetime(d["event_date"]).dt.normalize()
    out = car.copy()
    out["event_date"] = pd.to_datetime(out["event_date"]).dt.normalize()
    out = out.merge(d, on=["ticker", "event_date"], how="left")
    return out[out["n_scored"].fillna(0) >= min_texts].reset_index(drop=True)


def by_tercile(car: pd.DataFrame, windows: dict = WINDOWS) -> pd.DataFrame:
    car = car.assign(group=pd.qcut(car["sent_net"].rank(method="first"), 3, labels=list(GROUPS)))
    rows = []
    for gname in GROUPS:
        sub = car[car["group"] == gname]
        for w in windows:
            t = mean_test(sub[w], sub["event_date"])
            rows.append({"group": gname, "window": w, "n_events": int(sub[w].notna().sum()),
                         "sent_min": float(sub["sent_net"].min()), "sent_max": float(sub["sent_net"].max()),
                         **t, "median": float(sub[w].median())})
    return pd.DataFrame(rows)


def slope(car: pd.DataFrame, window: str = "month (+1,+20)") -> dict:
    """OLS of the window CAR on day-0 sentiment, errors clustered by event date."""
    import statsmodels.api as sm

    d = car[[window, "sent_net", "event_date"]].dropna()
    y = _winsor(d[window])
    X = sm.add_constant(d["sent_net"].to_numpy())
    res = sm.OLS(y.to_numpy(), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["event_date"])[0]})
    sd = d["sent_net"].std()
    return {"window": window, "coef": float(res.params[1]), "se": float(res.bse[1]), "t": float(res.tvalues[1]),
            "p": float(res.pvalues[1]), "sd_sent": float(sd), "effect_1sd": float(res.params[1] * sd),
            "n_events": int(len(d))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", default="data/processed/event_returns.parquet")
    p.add_argument("--daily", default="data/processed/daily_sentiment.parquet")
    p.add_argument("--min-texts", type=int, default=MIN_TEXTS)
    p.add_argument("--out", default="reports/tables/hype_sentiment.csv")
    args = p.parse_args(argv)

    car = attach_sentiment(cars(pd.read_parquet(args.events)), pd.read_parquet(args.daily), args.min_texts)
    table = by_tercile(car)
    slopes = pd.DataFrame([slope(car, w) for w in ("next day (+1)", "week (+1,+5)", "month (+1,+20)")])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    slopes.to_csv(Path(args.out).with_name("hype_sentiment_slope.csv"), index=False)

    print(f"{len(car)} hype spikes with >= {args.min_texts} scored texts on the spike day")
    for g, t in table.groupby("group", sort=False):
        print(f"\n{g}  (day-0 sentiment {t['sent_min'].iloc[0]:+.2f} to {t['sent_max'].iloc[0]:+.2f}, "
              f"{int(t['n_events'].max())} events)")
        for r in t.itertuples():
            star = "***" if r.p < 0.01 else "**" if r.p < 0.05 else "*" if r.p < 0.1 else ""
            print(f"  {r.window:<16} mean {r.mean:+7.2%} (t={r.t:5.1f}){star:<3} median {r.median:+7.2%}")
    print("\nSlope: CAR on day-0 sentiment (clustered by date)")
    for r in slopes.itertuples():
        print(f"  {r.window:<16} +1 SD sentiment -> {r.effect_1sd:+.2%} CAR (t={r.t:4.1f}, {r.n_events} events)")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
