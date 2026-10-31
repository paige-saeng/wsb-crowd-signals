"""H3: does WSB sentiment on day t predict the DIRECTION of the stock's
return on day t+1?

H1 found attention predicts the SIZE of tomorrow's move. H3 asks whether
bullish talk is followed by up moves and bearish talk by down moves.

Model (same setup as H1/H2: two-way fixed effects, errors clustered by stock
and by date, all variables winsorized at 1%/99%):

    ret_{i,t+1} = b * sentiment_{i,t} + controls_{i,t} + stock_i + date_t + e

    - date effects absorb the market's move, so this is a return relative to
      other stocks that day
    - controls: today's and the past week's return (momentum or reversal),
      volatility, abnormal attention and abnormal volume, so b is the effect of
      sentiment beyond attention and recent price action

Only stock-days with at least `min_texts` scored texts enter (a sentiment
average over 1 or 2 comments is mostly noise).

Usage
    python -m src.analysis.direction
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.attention import Spec, fit

H3_CONTROLS = ["ret", "ret5", "abs_ret", "rv5", "abn_attn", "abn_vol"]
MIN_TEXTS = 5
SAMPLES = ["all", "no_gme_amc", "no_squeeze", "h1_2021", "h2_2021"]


def add_sentiment(panel: pd.DataFrame, daily: pd.DataFrame, min_texts: int = MIN_TEXTS) -> pd.DataFrame:
    """Merge daily sentiment into the panel and add H3 targets.

    Sentiment is set to NaN on stock-days with fewer than `min_texts` texts.
    Targets: ret_next (t+1), xret_next (t+1 minus SPY), ret_next5 (t+1..t+5)."""
    d = daily.rename(columns={"trading_day": "date"}).copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    cols = ["sent_net", "bullishness", "sent_net_single", "bull_share", "bear_share"]
    df = panel.merge(d[["ticker", "date", "n_scored", "n_single", *cols]], on=["ticker", "date"], how="left")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    thin = df["n_scored"].fillna(0) < min_texts
    df.loc[thin, cols] = np.nan
    df.loc[df["n_single"].fillna(0) < min_texts, "sent_net_single"] = np.nan

    g = df.groupby("ticker", group_keys=False)
    df["ret_next"] = g["ret"].shift(-1)
    df["xret_next"] = df["ret_next"] - g["mkt_ret"].shift(-1)
    fwd = [g["ret"].shift(-k) for k in range(1, 6)]
    df["ret_next5"] = np.prod([1 + f for f in fwd], axis=0) - 1
    # sentiment relative to the stock's own recent tone (prior 30 sentiment days)
    prior = g["sent_net"].transform(lambda s: s.shift(1).rolling(30, min_periods=10).mean())
    df["abn_sent"] = df["sent_net"] - prior
    return df


def h3_specs(min_texts: int = MIN_TEXTS) -> list[Spec]:
    s = [Spec(f"H3 direction [{smp}]", "H3", "ret_next", "sent_net", H3_CONTROLS, smp) for smp in SAMPLES]
    s += [
        Spec("H3 direction, market-adjusted return", "H3", "xret_next", "sent_net", H3_CONTROLS),
        Spec("H3 direction, bullishness index", "H3", "ret_next", "bullishness", H3_CONTROLS),
        Spec("H3 direction, single-stock texts only", "H3", "ret_next", "sent_net_single", H3_CONTROLS),
        Spec("H3 direction, sentiment vs own 30-day norm", "H3", "ret_next", "abn_sent", H3_CONTROLS),
        Spec("H3 next week (t+1..t+5)", "H3", "ret_next5", "sent_net", H3_CONTROLS),
    ]
    return s


def summarize(results: pd.DataFrame) -> str:
    lines = []
    for r in results.itertuples():
        star = "***" if r.p < 0.01 else "**" if r.p < 0.05 else "*" if r.p < 0.1 else ""
        lines.append(f"{r.spec:<46} b={r.coef:+.4f} (t={r.t:5.1f}){star:<3} "
                     f"+1 SD {r.x} -> {r.effect_1sd * 1e4:+5.1f} bp  "
                     f"[N={r.n_obs:,}, {r.n_tickers} stocks, {r.n_days} days]")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", default="data/processed/panel.parquet")
    p.add_argument("--daily", default="data/processed/daily_sentiment.parquet")
    p.add_argument("--min-texts", type=int, default=MIN_TEXTS)
    p.add_argument("--out", default="reports/tables/direction_results.csv")
    args = p.parse_args(argv)

    df = add_sentiment(pd.read_parquet(args.panel), pd.read_parquet(args.daily), args.min_texts)
    res = pd.DataFrame([fit(df, s) for s in h3_specs(args.min_texts)])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"stock-days with >= {args.min_texts} scored texts: {df['sent_net'].notna().sum():,}\n")
    print(summarize(res))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
