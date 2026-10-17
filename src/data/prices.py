"""Daily prices and volume from Yahoo Finance (via yfinance), in long format.

Usage (from the repo root):
    python -m src.data.prices --tickers GME AMC BB NOK SPY \
        --start 2019-06-01 --end 2023-03-01 --out data/raw/prices.parquet

Or pass a text file with one ticker per line:
    python -m src.data.prices --ticker-file data/raw/tickers.txt ...

Notes
-----
- Always include SPY: it is the market benchmark for beta and abnormal returns.
- Start ~1 year before your Reddit window so a pre-event beta (days -250 to -30)
  can be estimated for the earliest events.
- yfinance often has NO data for delisted tickers (e.g. BBBY after 2023). The
  script reports missing tickers so the gap is documented, not silent; that list
  is your survivorship-bias limitation until you find another source.
- Uses adjusted close for returns (handles splits and dividends) and raw volume.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Return long DataFrame: date, ticker, close, adj_close, volume."""
    import yfinance as yf

    tickers = sorted({t.strip().upper() for t in tickers if t.strip()})
    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=False,
        group_by="column", progress=False, threads=True,
    )
    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "close", "adj_close", "volume"])
    if not isinstance(raw.columns, pd.MultiIndex):  # single ticker
        raw.columns = pd.MultiIndex.from_product([raw.columns, tickers])

    fields = {"Close": "close", "Adj Close": "adj_close", "Volume": "volume"}
    parts = []
    for src, dst in fields.items():
        wide = raw[src]
        parts.append(wide.stack(future_stack=True).rename(dst))
    df = pd.concat(parts, axis=1).reset_index()
    df.columns = ["date", "ticker", *fields.values()]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["adj_close"]).sort_values(["ticker", "date"])
    return df.reset_index(drop=True)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple and log daily returns per ticker (first day per ticker is NaN)."""
    df = df.sort_values(["ticker", "date"]).copy()
    prev = df.groupby("ticker")["adj_close"].shift(1)
    df["ret"] = df["adj_close"] / prev - 1
    df["log_ret"] = np.log(df["adj_close"] / prev)
    return df


def coverage_report(df: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    """Per-ticker first/last date and row count, plus tickers with no data."""
    got = df.groupby("ticker")["date"].agg(first="min", last="max", rows="count")
    missing = sorted(set(t.upper() for t in requested) - set(got.index))
    if missing:
        got = pd.concat([got, pd.DataFrame(index=pd.Index(missing, name="ticker"),
                                           data={"rows": 0})])
    return got


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="*", default=[])
    p.add_argument("--ticker-file")
    p.add_argument("--start", default="2019-06-01")
    p.add_argument("--end", default="2023-03-01")
    p.add_argument("--out", default="data/raw/prices.parquet")
    args = p.parse_args(argv)

    tickers = list(args.tickers)
    if args.ticker_file:
        tickers += Path(args.ticker_file).read_text().split()
    if "SPY" not in {t.upper() for t in tickers}:
        tickers.append("SPY")

    df = add_returns(download_prices(tickers, args.start, args.end))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    report = coverage_report(df, tickers)
    print(report.to_string())
    missing = report.index[report["rows"] == 0].tolist()
    if missing:
        print(f"\nNo data for {len(missing)} ticker(s): {', '.join(missing)}")
    print(f"\n{len(df):,} rows -> {out}")


if __name__ == "__main__":
    main()
