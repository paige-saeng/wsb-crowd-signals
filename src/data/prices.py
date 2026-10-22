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


def load_renames(path: str | Path) -> dict[str, str]:
    """old -> new symbol map from a CSV with columns old,new[,note]."""
    df = pd.read_csv(path, dtype=str, comment="#")
    return dict(zip(df["old"].str.strip().str.upper(), df["new"].str.strip().str.upper()))


def download_with_recovery(
    tickers: list[str],
    start: str,
    end: str,
    renames: dict[str, str] | None = None,
    require_by: str | None = None,
    downloader=None,
    first_seen: dict[str, str] | None = None,
    tolerance_days: int = 30,
    pause: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download prices, then try to recover tickers that came back empty.

    1. Batch download.
    2. Retry each missing ticker on its own (batch downloads sometimes drop
       symbols for transient reasons).
    3. For tickers still missing that have a rename (FB -> META), download the
       new symbol and relabel it with the old one, BUT only if its history
       starts early enough; otherwise the new symbol's data does not cover
       the old company's period and is rejected. "Early enough" = within
       `tolerance_days` of the later of `require_by` and the date the old
       symbol first appeared (`first_seen`, from data/reference/tickers_2021.csv),
       so a stock that listed mid-2021 (DWAC, Sept 2021) is judged from its
       own listing date.

    `pause` seconds are waited before each single-ticker request: Yahoo
    rate-limits bursts, and without a pause every retry can fail at once.

    Returns (prices with a `source_symbol` column, per-ticker status report).
    """
    import time
    downloader = downloader or download_prices
    renames = renames or {}
    want = sorted({t.strip().upper() for t in tickers if t.strip()})
    require_by = pd.Timestamp(require_by or start)
    first_seen = {k.upper(): pd.Timestamp(v) for k, v in (first_seen or {}).items()}

    def required_start(t: str) -> pd.Timestamp:
        return max(require_by, first_seen.get(t, require_by)) + pd.Timedelta(days=tolerance_days)

    def single(symbols):
        if pause:
            time.sleep(pause)
        return downloader(symbols, start, end)

    parts, status = [], {}
    df = downloader(want, start, end)
    if len(df):
        parts.append(df.assign(source_symbol=df["ticker"]))
        status.update({t: "ok" for t in df["ticker"].unique()})

    for t in [t for t in want if t not in status]:
        r = single([t])
        if len(r):
            parts.append(r.assign(source_symbol=t))
            status[t] = "ok on retry"

    for t in [t for t in want if t not in status]:
        new = renames.get(t)
        if not new:
            status[t] = "missing (no rename known)"
            continue
        r = single([new])
        need = required_start(t)
        if r.empty:
            status[t] = f"missing (renamed {new} has no data)"
        elif r["date"].min() > need:
            status[t] = f"missing ({new} history starts {r['date'].min().date()}, after {need.date()})"
        else:
            parts.append(r.assign(ticker=t, source_symbol=new))
            status[t] = f"recovered via {new}"

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["date", "ticker", "close", "adj_close", "volume", "source_symbol"])
    rep = pd.DataFrame({"ticker": list(status), "status": list(status.values())})
    got = out.groupby("ticker")["date"].agg(first="min", last="max", rows="count")
    rep = rep.merge(got, left_on="ticker", right_index=True, how="left").sort_values("ticker")
    return out.sort_values(["ticker", "date"]).reset_index(drop=True), rep.reset_index(drop=True)


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
    p.add_argument("--renames", default="data/reference/ticker_renames.csv",
                   help="CSV of old,new symbols tried for tickers Yahoo no longer lists")
    p.add_argument("--require-by", default="2021-02-01",
                   help="A renamed symbol's history must start by this date to be used")
    p.add_argument("--report", default="data/processed/price_coverage.csv")
    p.add_argument("--universe-csv", default="data/reference/tickers_2021.csv",
                   help="first_seen dates for judging renamed symbols (from src.data.universe)")
    p.add_argument("--pause", type=float, default=1.5,
                   help="Seconds to wait before each single-ticker retry (avoids rate limits)")
    p.add_argument("--update-missing", action="store_true",
                   help="Keep the existing --out file and only retry tickers it is missing")
    args = p.parse_args(argv)

    tickers = list(args.tickers)
    if args.ticker_file:
        tickers += Path(args.ticker_file).read_text().split()
    if "SPY" not in {t.upper() for t in tickers}:
        tickers.append("SPY")

    renames = load_renames(args.renames) if Path(args.renames).exists() else {}
    first_seen = {}
    if Path(args.universe_csv).exists():
        u = pd.read_csv(args.universe_csv, dtype=str)
        first_seen = dict(zip(u["ticker"].str.upper(), u["first_seen"]))
    out = Path(args.out)
    existing = pd.DataFrame()
    if args.update_missing and out.exists():
        existing = pd.read_parquet(out)
        have = set(existing["ticker"])
        tickers = [t for t in tickers if t.strip().upper() not in have]
        print(f"--update-missing: keeping {len(have)} tickers, retrying {len(tickers)}")
    df, report = download_with_recovery(tickers, args.start, args.end, renames, args.require_by,
                                        first_seen=first_seen, pause=args.pause)
    if len(existing):
        keep_cols = [c for c in existing.columns if c not in ("ret", "log_ret")]
        df = pd.concat([existing[keep_cols], df], ignore_index=True)
        old_rep = pd.read_csv(args.report) if Path(args.report).exists() else pd.DataFrame()
        if len(old_rep):
            report = pd.concat([old_rep[~old_rep["ticker"].isin(report["ticker"])], report], ignore_index=True)
    df = add_returns(df)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    report = report.sort_values("ticker").reset_index(drop=True)
    report.to_csv(args.report, index=False)

    counts = report["status"].str.replace(r" via .*| \(.*", "", regex=True).value_counts()
    print("\nPrice coverage: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    rec = report[report["status"].str.startswith("recovered")]
    if len(rec):
        print("Recovered: " + ", ".join(f"{r.ticker}->{r.status.split()[-1]}" for r in rec.itertuples()))
    miss = report[report["status"].str.startswith("missing")]
    if len(miss):
        print(f"Still missing ({len(miss)}): {' '.join(miss['ticker'])}")
    print(f"\n{len(df):,} rows -> {out}; per-ticker status -> {args.report}")


if __name__ == "__main__":
    main()
