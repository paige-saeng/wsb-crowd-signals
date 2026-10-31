"""Earnings announcement dates, to control for news in H1-H4.

Why: attention may only predict big moves because WSB talks about stocks that
are about to report earnings, and earnings move prices anyway. Flagging
earnings days lets the regressions separate the two.

Source: Yahoo Finance via yfinance (`Ticker.get_earnings_dates`), the same
unofficial source as the prices: free, rate-limited, and missing most
delisted companies. Renamed symbols are retried under their new symbol, as
for prices (data/reference/ticker_renames.csv).

Timing: an announcement's timestamp is mapped to the first trading session
whose close is after it, the same rule as for Reddit posts
(src/data/calendar.py). Before the open or during the session -> that day;
after the close -> the next trading day. That session is the "reaction day",
when the news first moves the price. Yahoo sometimes gives only a date
(midnight); then both that day and the next session are flagged.

Usage
    python -m src.data.earnings --ticker-file data/processed/study_tickers.txt \
        --start 2020-12-01 --end 2022-02-01 --out data/raw/earnings_dates.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.calendar import assign_trading_day, trading_sessions


def yahoo_earnings(ticker: str, limit: int = 40) -> pd.DataFrame:
    """Raw earnings timestamps from Yahoo for one ticker (newest first)."""
    import yfinance as yf

    df = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    if df is None or df.empty:
        return pd.DataFrame(columns=["announced"])
    return pd.DataFrame({"announced": pd.DatetimeIndex(df.index)})


def download_earnings(tickers, start: str, end: str, renames: dict[str, str] | None = None,
                      fetch=yahoo_earnings, pause: float = 1.0, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Announcement timestamps in [start, end) per ticker, plus a status table."""
    renames = renames or {}
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    rows, status = [], []
    for i, t in enumerate(tickers):
        got, source = None, None
        for sym in [t] + ([renames[t]] if t in renames else []):
            try:
                df = fetch(sym)
            except Exception as e:          # Yahoo errors, unknown symbols
                df, err = None, str(e)[:80]
            if df is not None and len(df):
                ts = pd.to_datetime(df["announced"], utc=True)
                ts = ts[(ts >= lo) & (ts < hi)]
                if len(ts):
                    got, source = ts, sym
                    break
            if pause:
                time.sleep(pause)
        if got is None:
            status.append({"ticker": t, "status": "missing", "n": 0})
        else:
            rows.append(pd.DataFrame({"ticker": t, "announced": got.values, "source_symbol": source}))
            status.append({"ticker": t, "status": "ok" if source == t else f"recovered via {source}", "n": len(got)})
        if log and (i + 1) % 25 == 0:
            log(f"  {i + 1}/{len(tickers)} tickers")
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["ticker", "announced", "source_symbol"])
    return out, pd.DataFrame(status)


def reaction_days(earnings: pd.DataFrame, sessions: pd.DataFrame | None = None) -> pd.DataFrame:
    """ticker, date: the trading day(s) each announcement first moves the price.

    A timestamp at exactly midnight New York time means Yahoo gave no time of
    day, so both that day's session and the next one are returned."""
    if earnings.empty:
        return pd.DataFrame(columns=["ticker", "date"])
    ts = pd.to_datetime(earnings["announced"], utc=True)
    if sessions is None:
        sessions = trading_sessions((ts.min() - pd.Timedelta(days=7)).date(), (ts.max() + pd.Timedelta(days=14)).date())
    day = assign_trading_day(ts.values, sessions)["trading_day"].to_numpy()
    out = [pd.DataFrame({"ticker": earnings["ticker"].to_numpy(), "date": day})]
    local = ts.dt.tz_convert("America/New_York")
    untimed = ((local.dt.hour == 0) & (local.dt.minute == 0)).to_numpy()
    if untimed.any():
        idx = pd.DatetimeIndex(sessions.index)
        pos = np.searchsorted(idx.values, day[untimed].astype("datetime64[ns]"), side="right")
        nxt = np.where(pos < len(idx), idx.values[np.minimum(pos, len(idx) - 1)], np.datetime64("NaT", "ns"))
        out.append(pd.DataFrame({"ticker": earnings["ticker"].to_numpy()[untimed], "date": nxt}))
    r = pd.concat(out, ignore_index=True).dropna()
    r["date"] = pd.to_datetime(r["date"]).dt.normalize()
    return r.drop_duplicates().sort_values(["ticker", "date"]).reset_index(drop=True)


def add_earnings_flags(panel: pd.DataFrame, days: pd.DataFrame, window: int = 2) -> pd.DataFrame:
    """Flag earnings reaction days in the panel.

    earn_today   day t is a reaction day
    earn_next    day t+1 is a reaction day (the news lands on the outcome day)
    earn_window  a reaction day falls within `window` trading days of t
    Stocks with no earnings dates at all get NaN flags, so they drop out of
    specifications that use the flags instead of counting as "no earnings"."""
    df = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    key = set(zip(days["ticker"], pd.to_datetime(days["date"]).dt.normalize()))
    df["earn_today"] = [float((t, d) in key) for t, d in zip(df["ticker"], df["date"])]
    g = df.groupby("ticker")["earn_today"]
    df["earn_next"] = g.shift(-1)
    near = sum(g.shift(k).fillna(0) for k in range(-window, window + 1))
    df["earn_window"] = (near > 0).astype(float)
    covered = df["ticker"].isin(set(days["ticker"]))
    df.loc[~covered, ["earn_today", "earn_next", "earn_window"]] = np.nan
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticker-file", default="data/processed/study_tickers.txt")
    p.add_argument("--start", default="2020-12-01")
    p.add_argument("--end", default="2022-02-01")
    p.add_argument("--renames", default="data/reference/ticker_renames.csv")
    p.add_argument("--pause", type=float, default=1.0, help="Seconds between Yahoo requests")
    p.add_argument("--out", default="data/raw/earnings_dates.csv")
    args = p.parse_args(argv)

    from src.data.prices import load_renames

    tickers = [t for t in Path(args.ticker_file).read_text().split() if t != "SPY"]
    renames = load_renames(args.renames) if Path(args.renames).exists() else {}
    print(f"earnings dates for {len(tickers)} tickers, {args.start} to {args.end} (~{len(tickers) * args.pause / 60:.0f}+ min)")
    earn, status = download_earnings(tickers, args.start, args.end, renames, pause=args.pause)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    earn.to_csv(args.out, index=False)
    status.to_csv(Path(args.out).with_name("earnings_coverage.csv"), index=False)
    ok = status["status"] != "missing"
    print(f"-> {args.out}: {len(earn):,} announcements for {ok.sum()} of {len(status)} tickers; "
          f"missing: {', '.join(status.loc[~ok, 'ticker'].head(15))}{' ...' if (~ok).sum() > 15 else ''}")


if __name__ == "__main__":
    main()
