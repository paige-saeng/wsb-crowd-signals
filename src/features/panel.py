"""Build the stock-day panel: WSB attention features on day t, outcomes on day t+1.

Timing (no lookahead):
    - Mentions for trading day t = posts/comments made after the close of t-1
      and before the close of t (see src/data/calendar.py).
    - Features for t use only data up to and including the close of t.
    - Targets are measured on day t+1 (close t -> close t+1).

Main attention measure:
    abn_attn_t = log(1 + mentions_t) - mean(log(1 + mentions)) over the prior
    30 trading days (t-30..t-1). A value of 1.0 means about e^1 = 2.7x the
    stock's normal attention. The z-score version from the research plan,
    which divides by the prior 30-day standard deviation, is kept as
    `abn_attn_z` with that standard deviation floored at 0.25: quiet stocks
    have near-zero spread, which would otherwise blow the z-score up.

Targets:
    abs_ret_next   = |return on t+1|                    (H1, volatility proxy)
    abs_xret_next  = |return on t+1 - SPY return on t+1| (H1 robustness)
    abn_vol_next   = log(volume on t+1) - mean log volume over t-59..t   (H2)

Usage:
    python -m src.features.panel --mentions data/processed/daily_mentions.parquet \
        --prices data/raw/prices.parquet --tickers data/processed/study_tickers.txt \
        --out data/processed/panel.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ATTN_WINDOW = 30
VOL_WINDOW = 60
Z_STD_FLOOR = 0.25
MARKET = "SPY"


def _prior_mean(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Rolling mean over the PRIOR `window` rows (excludes the current row)."""
    return s.shift(1).rolling(window, min_periods=min_periods).mean()


def _prior_std(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=min_periods).std()


def build_panel(
    daily_mentions: pd.DataFrame,
    prices: pd.DataFrame,
    tickers: list[str] | None = None,
    market: str = MARKET,
) -> pd.DataFrame:
    """Return one row per (ticker, date) with features and next-day targets.

    daily_mentions: ticker, trading_day, mentions, unique_authors
    prices: date, ticker, adj_close, volume (+ ret; computed if missing)
    """
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px = px.sort_values(["ticker", "date"])
    if "ret" not in px.columns:
        px["ret"] = px.groupby("ticker")["adj_close"].pct_change()

    mkt = px.loc[px["ticker"] == market, ["date", "ret"]].rename(columns={"ret": "mkt_ret"})
    if mkt.empty:
        raise ValueError(f"market ticker {market} missing from prices")
    if tickers is not None:
        keep = {t.upper() for t in tickers} - {market}
        px = px[px["ticker"].isin(keep)]
    else:
        px = px[px["ticker"] != market]

    dm = daily_mentions.rename(columns={"trading_day": "date"}).copy()
    dm["date"] = pd.to_datetime(dm["date"]).dt.normalize()
    df = px.merge(dm[["ticker", "date", "mentions", "unique_authors"]], on=["ticker", "date"], how="left")
    # Zero mentions only means "no mentions" inside the window the Reddit data
    # covers. Outside it, attention is unknown (NaN), not zero; filling zeros
    # there would make the first weeks of coverage look like a huge spike.
    covered = df["date"].between(dm["date"].min(), dm["date"].max())
    df[["mentions", "unique_authors"]] = df[["mentions", "unique_authors"]].fillna(0)
    df.loc[~covered, ["mentions", "unique_authors"]] = np.nan
    df = df.merge(mkt, on="date", how="left").sort_values(["ticker", "date"]).reset_index(drop=True)

    g = df.groupby("ticker", group_keys=False)

    # ---- attention (day t)
    df["log_m"] = np.log1p(df["mentions"])
    df["log_a"] = np.log1p(df["unique_authors"])
    base_m = g["log_m"].transform(lambda s: _prior_mean(s, ATTN_WINDOW, 20))
    std_m = g["log_m"].transform(lambda s: _prior_std(s, ATTN_WINDOW, 20))
    df["abn_attn"] = df["log_m"] - base_m
    df["abn_attn_z"] = df["abn_attn"] / std_m.clip(lower=Z_STD_FLOOR)
    df["abn_authors"] = df["log_a"] - g["log_a"].transform(lambda s: _prior_mean(s, ATTN_WINDOW, 20))

    # ---- price and volume state (day t)
    df["abs_ret"] = df["ret"].abs()
    df["rv5"] = g["abs_ret"].transform(lambda s: s.rolling(5, min_periods=5).mean())     # t-4..t
    df["rv20"] = g["abs_ret"].transform(lambda s: s.rolling(20, min_periods=15).mean())  # t-19..t
    df["ret5"] = g["ret"].transform(lambda s: (1 + s).rolling(5, min_periods=5).apply(np.prod, raw=True) - 1)
    df["log_vol"] = np.log(df["volume"].where(df["volume"] > 0))
    df["abn_vol"] = df["log_vol"] - g["log_vol"].transform(lambda s: _prior_mean(s, VOL_WINDOW, 40))
    df["abn_vol5"] = g["abn_vol"].transform(lambda s: s.rolling(5, min_periods=5).mean())

    # ---- next-day targets (t+1), each baseline known at the close of t
    vol_base_t = g["log_vol"].transform(lambda s: s.rolling(VOL_WINDOW, min_periods=40).mean())  # t-59..t
    df["abs_ret_next"] = g["abs_ret"].shift(-1)
    df["abs_xret_next"] = (g["ret"].shift(-1) - df.groupby("ticker")["mkt_ret"].shift(-1)).abs()
    df["abn_vol_next"] = g["log_vol"].shift(-1) - vol_base_t
    df["abn_attn_next"] = g["abn_attn"].shift(-1)  # for the reverse-direction check (H5 preview)
    df["next_date"] = g["date"].shift(-1)

    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mentions", default="data/processed/daily_mentions.parquet")
    p.add_argument("--prices", default="data/raw/prices.parquet")
    p.add_argument("--tickers", default="data/processed/study_tickers.txt")
    p.add_argument("--out", default="data/processed/panel.parquet")
    args = p.parse_args(argv)

    tickers = Path(args.tickers).read_text().split()
    panel = build_panel(pd.read_parquet(args.mentions), pd.read_parquet(args.prices), tickers)
    panel.to_parquet(args.out, index=False)
    usable = panel.dropna(subset=["abn_attn", "abs_ret_next", "abn_vol_next", "rv20", "abn_vol5"])
    print(f"panel: {len(panel):,} rows, {panel['ticker'].nunique()} tickers, "
          f"{panel['date'].min().date()} to {panel['date'].max().date()}; "
          f"{len(usable):,} rows usable for regressions -> {args.out}")
    missing = sorted(set(t.upper() for t in tickers) - {MARKET} - set(panel["ticker"]))
    if missing:
        print(f"no price data for {len(missing)} study tickers: {' '.join(missing)}")


if __name__ == "__main__":
    main()
