"""Build a point-in-time ticker universe from archived daily ticker lists.

Why: today's NASDAQ symbol files miss tickers that traded in 2021 but were
later renamed or delisted (WISH -> LOGC, DWAC -> DJT, SKLZ, BBBY...). In the
hand-labeled validation, 8 of 12 extraction misses came from this gap, which
is survivorship bias in the ticker universe.

Source: github.com/rreichel3/US-Stock-Symbols, which has committed a daily
snapshot of NASDAQ, NYSE and AMEX tickers since 2021-01-30. Every commit is a
snapshot, so the union of all commits in a window = every symbol listed at
any point in that window.

Usage (from the repo root; needs git and internet):
    python -m src.data.universe --start 2021-01-01 --end 2022-01-31 \
        --out data/reference/tickers_2021.csv

Output CSV columns: ticker, first_seen, last_seen (snapshot dates).
Known gaps: nothing before 2021-01-30; OTC tickers (e.g. TGODF) are not
included because the source covers NASDAQ, NYSE and AMEX only.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

REPO_URL = "https://github.com/rreichel3/US-Stock-Symbols.git"
FILES = ("nasdaq/nasdaq_tickers.txt", "nyse/nyse_tickers.txt", "amex/amex_tickers.txt")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          text=True, check=True).stdout


def clone(dest: Path) -> Path:
    """Full clone (blobs included) so reading 300+ snapshots is fast and offline."""
    subprocess.run(["git", "clone", "--quiet", REPO_URL, str(dest)], check=True)
    return dest


def snapshots(repo: Path, start: str, end: str) -> list[tuple[str, str]]:
    """(commit hash, date) for each commit in [start, end], oldest first."""
    out = _git(repo, "log", "--format=%H %ad", "--date=short",
               f"--since={start}", f"--until={end}")
    rows = [line.split() for line in out.splitlines() if line.strip()]
    return [(h, d) for h, d in reversed(rows) if start <= d <= end]


def build(repo: Path, start: str, end: str) -> pd.DataFrame:
    """Union of all snapshots: one row per ticker with first and last date seen."""
    first: dict[str, str] = {}
    last: dict[str, str] = {}
    snaps = snapshots(repo, start, end)
    for h, d in snaps:
        for f in FILES:
            try:
                txt = _git(repo, "show", f"{h}:{f}")
            except subprocess.CalledProcessError:
                continue  # file absent in this snapshot
            for s in txt.split():
                s = s.strip().upper()
                if s:
                    first.setdefault(s, d)
                    last[s] = d
    df = pd.DataFrame({"ticker": list(first), "first_seen": list(first.values()),
                       "last_seen": [last[t] for t in first]})
    df.attrs["n_snapshots"] = len(snaps)
    return df.sort_values("ticker").reset_index(drop=True)


def load_universe_csv(path: str | Path) -> set[str]:
    """Tickers from a CSV written by this module (or any CSV with a `ticker` column)."""
    return set(pd.read_csv(path, dtype=str)["ticker"].dropna().str.upper())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2022-01-31")
    p.add_argument("--out", default="data/reference/tickers_2021.csv")
    p.add_argument("--repo", help="Existing clone of US-Stock-Symbols (skips cloning)")
    args = p.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(args.repo) if args.repo else clone(Path(tmp) / "US-Stock-Symbols")
        df = build(repo, args.start, args.end)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df):,} tickers from {df.attrs['n_snapshots']} snapshots "
          f"({df['first_seen'].min()} to {df['last_seen'].max()}) -> {out}")


if __name__ == "__main__":
    main()
