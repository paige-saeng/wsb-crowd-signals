"""Find stock ticker mentions in WSB text.

Two kinds of matches:

1. Cashtags, e.g. "$GME" or "$gme". High confidence: the author explicitly
   marked a ticker. Accepted if the symbol is in the universe.
2. Bare symbols, e.g. "GME to the moon". Only ALL-CAPS tokens of 2-5 letters
   that are in the universe and NOT in the stoplist. This is where false
   positives come from: "DD", "CEO", "ALL", "IT", "YOLO" are all real tickers
   (or look like them) but on WSB they are almost always plain words.

Optionally, a small set of well-known symbols that are not English words
(GME, AMC, PLTR, ...) can also be matched in lowercase, because WSB users
often type "gme" and "amc".

Main entry points:
    find_tickers(text, universe)           -> list of mentions for one text
    extract_mentions(df, universe)         -> long table of mentions for a DataFrame
    daily_mention_counts(mentions)         -> ticker x trading_day counts
    sample_for_labeling / score_labels     -> precision/recall on a hand-labeled sample
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

# ------------------------------------------------------------------ word lists

# Tokens that are (or could be) tickers but on WSB are almost always words,
# slang, or finance acronyms. Bare uppercase matches on these are dropped;
# a cashtag ($ALL) still counts. Extend this as your labeled sample reveals
# new false positives, and note each addition in the commit message.
STOPLIST = frozenset("""
A I AM AN AND ANY ARE AS AT BE BIG BUT BY CAN CAR DO EAT EVER FOR GO GOOD HAS
HE HOLD IF IN IS IT ITS ME MY NEW NO NOT NOW OF OK ON ONE OR OUT PLAY REAL RUN
SEE SO TELL THE TO TWO UP US USA WE WHO WOW YOU
ALL BEST CASH DAY DAYS FAST FUN GAIN GAINS HUGE JOB KEY LIFE LOVE LOW MAN MOON
MORE NEXT OPEN PEAK PUMP RIDE ROCK SAFE STAY TEAM TRUE WELL WORK
DD YOLO FD FDS FOMO HODL BTFD LFG GG GL TA TLDR TL DR IMO IMHO LOL LMAO WTF OMG
EOD EOW EOM EOY ATH ATL YTD QOQ YOY
CEO CFO CTO COO SEC FED FOMC IRS IPO SPAC ETF ETFS EPS PE PEG IV OTM ITM ATM
DTE OI RSI MACD EMA SMA VWAP NYSE OTC GDP CPI USD EUR UK EU AI EV EVS API
CALL CALLS PUT PUTS BUY SELL SHORT LONG BEAR BULL BAG APE APES WSB RH IRA
NFT PDT HFT MM MMS SI CTB FTD FTDS DRS
""".split())

# Symbols safe to match even in lowercase: distinctive, not English words.
# Keep this short and deliberate.
LOWERCASE_OK = frozenset("""
GME AMC BB BBBY NOK PLTR TSLA NVDA AAPL MSFT AMD SPY QQQ NIO KOSS EXPR NAKD
CLOV WISH SNDL TLRY RKT MVIS SPCE CLNE WKHS UWMC
""".split())

CASHTAG_RE = re.compile(r"(?<![\w$])\$([A-Za-z]{1,5})(?![\w])")
BARE_UPPER_RE = re.compile(r"(?<![\w$])([A-Z]{2,5})(?![\w])")
BARE_ANY_RE = re.compile(r"(?<![\w$])([A-Za-z]{2,5})(?![\w])")
# Links, emails, and Reddit user/sub references carry no ticker signal.
NOISE_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+|\b[ur]/\w+", re.IGNORECASE)


@dataclass(frozen=True)
class Mention:
    ticker: str
    source: str  # "cashtag" | "bare" | "lowercase"


# ------------------------------------------------------------------ universe

def load_nasdaqtrader(*paths: str | Path) -> set[str]:
    """Load symbols from NASDAQ Trader symbol-directory files.

    Download (free, updated daily) from:
        https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt
        https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt
    Both are pipe-delimited with a trailer line. Test issues are excluded.

    Limitation: these list only CURRENTLY listed stocks, so names delisted
    since 2021 (e.g. BBBY) are missing. Add them with `extra=` in
    `build_universe`, and document the gap (survivorship bias).
    """
    symbols: set[str] = set()
    for path in paths:
        df = pd.read_csv(path, sep="|", dtype=str)
        df = df[~df.iloc[:, 0].str.startswith("File Creation Time", na=False)]
        col = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"] != "Y"]
        symbols |= set(df[col].dropna().str.upper())
    return symbols


def build_universe(symbols: Iterable[str], extra: Iterable[str] = ()) -> frozenset[str]:
    """Clean a symbol list into the matching universe.

    Keeps plain 1-5 letter symbols (drops warrants/units/classes like
    "BRK.B", "GME.WS", "ABCDU" stays if 5 letters).
    """
    out = {s.strip().upper() for s in [*symbols, *extra] if s and isinstance(s, str)}
    return frozenset(s for s in out if re.fullmatch(r"[A-Z]{1,5}", s))


# ------------------------------------------------------------------ extraction

def find_tickers(
    text: str | None,
    universe: frozenset[str],
    stoplist: frozenset[str] = STOPLIST,
    lowercase_ok: frozenset[str] = LOWERCASE_OK,
) -> list[Mention]:
    """Return unique ticker mentions in one text, in order of first appearance.

    Each ticker appears at most once per text (a comment saying "GME GME GME"
    is one mention), with its highest-confidence source.
    """
    if not text or not isinstance(text, str):
        return []
    text = NOISE_RE.sub(" ", text)
    found: dict[str, str] = {}
    rank = {"cashtag": 0, "bare": 1, "lowercase": 2}

    def add(sym: str, source: str):
        if sym not in found or rank[source] < rank[found[sym]]:
            found[sym] = source

    for m in CASHTAG_RE.finditer(text):
        sym = m.group(1).upper()
        if sym in universe:
            add(sym, "cashtag")
    for m in BARE_UPPER_RE.finditer(text):
        sym = m.group(1)
        if sym in universe and sym not in stoplist:
            add(sym, "bare")
    if lowercase_ok:
        for m in BARE_ANY_RE.finditer(text):
            tok = m.group(1)
            sym = tok.upper()
            if tok != sym and sym in lowercase_ok and sym in universe:
                add(sym, "lowercase")
    return [Mention(t, s) for t, s in found.items()]


def extract_mentions(
    df: pd.DataFrame,
    universe: frozenset[str],
    text_cols: tuple[str, ...] = ("title", "text"),
    keep_cols: tuple[str, ...] = ("id", "kind", "trading_day", "author", "score"),
    **kwargs,
) -> pd.DataFrame:
    """Long table: one row per (post or comment, ticker).

    Title and body are combined so a post that names GME in both counts once.
    """
    cols = [c for c in text_cols if c in df.columns]
    combined = df[cols].fillna("").astype(str).agg("\n".join, axis=1)
    rows = []
    keep = [c for c in keep_cols if c in df.columns]
    records = df[keep].to_dict("records")
    for rec, text in zip(records, combined):
        for m in find_tickers(text, universe, **kwargs):
            rows.append({**rec, "ticker": m.ticker, "source": m.source})
    out = pd.DataFrame(rows, columns=[*keep, "ticker", "source"])
    return out


def daily_mention_counts(mentions: pd.DataFrame) -> pd.DataFrame:
    """Ticker x trading_day counts: mentions (texts naming the ticker) and
    unique authors (a crude guard against one account spamming)."""
    g = mentions.groupby(["ticker", "trading_day"])
    out = g.size().rename("mentions").to_frame()
    if "author" in mentions.columns:
        out["unique_authors"] = g["author"].nunique()
    return out.reset_index().sort_values(["ticker", "trading_day"])


# ------------------------------------------------------------------ validation

def sample_for_labeling(
    df: pd.DataFrame, universe: frozenset[str], n: int = 300, seed: int = 0,
    text_cols: tuple[str, ...] = ("title", "text"),
) -> pd.DataFrame:
    """Random sample with the extractor's predictions and an empty column for
    your hand labels. Save it, fill `true_tickers` (space-separated, blank if
    none) in a spreadsheet, then pass it to `score_labels`.

    Half the sample is drawn from texts with at least one prediction so the
    rare false positives show up often enough to measure.
    """
    cols = [c for c in text_cols if c in df.columns]
    base = df.assign(_text=df[cols].fillna("").astype(str).agg("\n".join, axis=1))
    base = base[base["_text"].str.strip().str.len() > 0]
    base = base.assign(predicted=base["_text"].map(
        lambda t: " ".join(m.ticker for m in find_tickers(t, universe))))
    hit = base[base["predicted"] != ""]
    miss = base[base["predicted"] == ""]
    k = min(n // 2, len(hit))
    sample = pd.concat([
        hit.sample(k, random_state=seed),
        miss.sample(min(n - k, len(miss)), random_state=seed),
    ]).sample(frac=1, random_state=seed)
    return sample[["id", "_text", "predicted"]].rename(columns={"_text": "text"}).assign(true_tickers="")


def run_on_parquet(
    inputs: Iterable[str | Path],
    universe: frozenset[str],
    out_dir: str | Path,
    batch_size: int = 250_000,
) -> pd.DataFrame:
    """Extract mentions from interim parquet files in batches (memory stays flat).

    Writes out_dir/mentions.parquet and out_dir/daily_mentions.parquet and
    returns the daily counts.
    """
    import pyarrow.parquet as pq

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for path in inputs:
        pf = pq.ParquetFile(path)
        cols = [c for c in ("id", "kind", "trading_day", "author", "score", "title", "text")
                if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            parts.append(extract_mentions(batch.to_pandas(), universe))
            print(f"  {path}: {sum(len(p) for p in parts):,} mentions so far", flush=True)
    mentions = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["id", "kind", "trading_day", "author", "score", "ticker", "source"])
    mentions.to_parquet(out_dir / "mentions.parquet", index=False)
    daily = daily_mention_counts(mentions)
    daily.to_parquet(out_dir / "daily_mentions.parquet", index=False)
    return daily


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Extract ticker mentions from interim posts/comments parquet files.",
        epilog="Example: python -m src.nlp.tickers --inputs data/interim/posts.parquet "
               "data/interim/comments.parquet --symbols data/raw/nasdaqlisted.txt "
               "data/raw/otherlisted.txt --extra BBBY --out data/processed",
    )
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--symbols", nargs="+", required=True,
                   help="NASDAQ Trader nasdaqlisted.txt / otherlisted.txt files")
    p.add_argument("--extra", nargs="*", default=["BBBY"],
                   help="Delisted or missing tickers to add to the universe")
    p.add_argument("--out", default="data/processed")
    args = p.parse_args(argv)

    universe = build_universe(load_nasdaqtrader(*args.symbols), extra=args.extra)
    print(f"universe: {len(universe):,} symbols")
    daily = run_on_parquet(args.inputs, universe, args.out)
    top = daily.groupby("ticker")["mentions"].sum().sort_values(ascending=False).head(20)
    print("\nTop 20 tickers by total mentions:\n" + top.to_string())


def score_labels(labeled: pd.DataFrame) -> dict:
    """Micro precision / recall / F1 of `predicted` vs `true_tickers`."""
    tp = fp = fn = 0
    for pred, true in zip(labeled["predicted"].fillna(""), labeled["true_tickers"].fillna("")):
        p, t = set(str(pred).split()), set(str(true).upper().split())
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    main()
