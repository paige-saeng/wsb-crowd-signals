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
CLOV SNDL TLRY RKT MVIS SPCE CLNE WKHS UWMC
""".split())

# Cashtags used as jokes, not stocks. Dropped entirely, cashtags included.
# "$ROPE" is WSB dark humor about losses (88% of its 2021 mentions are cashtags).
EXCLUDE = frozenset({"ROPE"})

CASHTAG_RE = re.compile(r"(?<![\w$])\$([A-Za-z]{1,5})(?![\w])")
BARE_UPPER_RE = re.compile(r"(?<![\w$])(?<![A-Za-z]['’])([A-Z]{2,5})(?![\w])(?!['’](?![sS]\b)[A-Za-z])")
BARE_ANY_RE = re.compile(r"(?<![\w$])(?<![A-Za-z]['’])([A-Za-z]{2,5})(?![\w])(?!['’](?![sS]\b)[A-Za-z])")
# Links, emails, and Reddit user/sub references carry no ticker signal.
NOISE_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+|\b[ur]/\w+", re.IGNORECASE)


COMMON_WORDS_PATH = Path(__file__).parent / "data" / "common_words.txt"


def load_common_words(path: str | Path = COMMON_WORDS_PATH) -> frozenset[str]:
    """2-5 letter common English words, uppercase (see data/common_words.txt)."""
    lines = Path(path).read_text().split()
    return frozenset(w.strip().upper() for w in lines if w.strip() and not w.startswith("#"))


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

def predict(
    text: str | None,
    universe: frozenset[str],
    cashtag_only: frozenset[str] = frozenset(),
    exclude: frozenset[str] = EXCLUDE,
) -> list[str]:
    """What the FULL pipeline outputs for one text: extraction, then the
    ambiguity filter (`cashtag_only` = tickers whose decision was
    "cashtags only", from ambiguous_ticker_decisions.csv) and the exclude list.
    """
    out = []
    for m in find_tickers(text, universe):
        if m.ticker in exclude:
            continue
        if m.ticker in cashtag_only and m.source != "cashtag":
            continue
        out.append(m.ticker)
    return out


def load_cashtag_only(decisions_csv: str | Path) -> frozenset[str]:
    """Tickers limited to cashtags, from the decisions file `finalize` writes."""
    d = pd.read_csv(decisions_csv, index_col=0)
    return frozenset(d.index[d["decision"] == "cashtags only"])


def sample_for_labeling(
    df: pd.DataFrame, universe: frozenset[str], n: int = 300, seed: int = 0,
    text_cols: tuple[str, ...] = ("title", "text"),
    cashtag_only: frozenset[str] = frozenset(),
    exclude: frozenset[str] = EXCLUDE,
) -> pd.DataFrame:
    """Random sample with the pipeline's predictions and an empty column for
    your hand labels. Fill `true_tickers` (space-separated, blank if none),
    then score with `score_labels(load_labels(path))`.

    Half the sample is drawn from texts with at least one prediction so the
    rare false positives show up often enough to measure.
    """
    cols = [c for c in text_cols if c in df.columns]
    base = df.assign(_text=df[cols].fillna("").astype(str).agg("\n".join, axis=1))
    base = base[base["_text"].str.strip().str.len() > 0]
    base = base[~base["_text"].str.strip().isin(["[deleted]", "[removed]"])]
    base = base.assign(predicted=base["_text"].map(
        lambda t: " ".join(predict(t, universe, cashtag_only, exclude))))
    hit = base[base["predicted"] != ""]
    miss = base[base["predicted"] == ""]
    k = min(n // 2, len(hit))
    sample = pd.concat([
        hit.sample(k, random_state=seed),
        miss.sample(min(n - k, len(miss)), random_state=seed),
    ]).sample(frac=1, random_state=seed)
    return sample[["id", "_text", "predicted"]].rename(columns={"_text": "text"}).assign(true_tickers="")


def ticker_source_stats(mentions: pd.DataFrame) -> pd.DataFrame:
    """Per ticker: mentions by source and the share that were cashtags."""
    st = mentions.groupby(["ticker", "source"]).size().unstack(fill_value=0)
    for col in ("cashtag", "bare", "lowercase"):
        if col not in st.columns:
            st[col] = 0
    st["total"] = st[["cashtag", "bare", "lowercase"]].sum(axis=1)
    st["cashtag_share"] = st["cashtag"] / st["total"]
    return st[["total", "cashtag", "bare", "lowercase", "cashtag_share"]]


def drop_ambiguous(
    mentions: pd.DataFrame,
    common_words: frozenset[str] | None = None,
    allow: frozenset[str] = LOWERCASE_OK,
    min_cashtag_share: float = 0.05,
    min_cashtags: int = 20,
    exclude: frozenset[str] = EXCLUDE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove non-cashtag mentions of tickers that are probably just words.

    A ticker is AMBIGUOUS if it is a common English word (JUST, TIME, LINE) or
    only two letters (TD, HF, PM), and is not on the allow list.
    An ambiguous ticker keeps its bare/lowercase mentions only if the crowd
    itself treats it as a ticker: at least `min_cashtag_share` of its mentions
    are cashtags AND at least `min_cashtags` cashtags overall. Its cashtag
    mentions are always kept.

    Why cashtag share: in 2021 WSB data real tickers that are also words are
    cashtagged 8-15% of the time (HOOD, COIN, PLUG, NET), while words are
    almost never cashtagged (JUST, TIME, LINE 0%; GAME, MEME 2%).

    Tickers in `exclude` (joke cashtags) are removed entirely.

    Returns (filtered mentions, decisions table for the write-up).
    """
    if common_words is None:
        common_words = load_common_words()
    excluded = mentions["ticker"].isin(exclude)
    n_excluded = mentions[excluded].groupby("ticker").size()
    mentions = mentions[~excluded]
    st = ticker_source_stats(mentions)
    idx = st.index.to_series()
    ambiguous = (idx.isin(common_words) | (idx.str.len() == 2)) & ~idx.isin(allow)
    passes = (st["cashtag_share"] >= min_cashtag_share) & (st["cashtag"] >= min_cashtags)
    st["ambiguous"] = ambiguous
    st["decision"] = "keep"
    st.loc[ambiguous & passes, "decision"] = "keep (cashtag evidence)"
    st.loc[ambiguous & ~passes, "decision"] = "cashtags only"
    cashtag_only = set(st.index[st["decision"] == "cashtags only"])
    drop = mentions["ticker"].isin(cashtag_only) & (mentions["source"] != "cashtag")
    st["dropped"] = mentions[drop].groupby("ticker").size().reindex(st.index, fill_value=0)
    decisions = st[st["ambiguous"]].sort_values("total", ascending=False)
    if len(n_excluded):
        ex = pd.DataFrame({"total": n_excluded, "dropped": n_excluded,
                           "ambiguous": True, "decision": "excluded (joke cashtag)"})
        decisions = pd.concat([ex, decisions])
    return mentions[~drop].reset_index(drop=True), decisions


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
    mentions.to_parquet(out_dir / "mentions_raw.parquet", index=False)
    return finalize(mentions, out_dir)


def finalize(mentions: pd.DataFrame, out_dir: str | Path, **filter_kwargs) -> pd.DataFrame:
    """Apply the ambiguity filter and write mentions, daily counts, decisions."""
    out_dir = Path(out_dir)
    n_in = len(mentions)
    kept, decisions = drop_ambiguous(mentions, **filter_kwargs)
    decisions.to_csv(out_dir / "ambiguous_ticker_decisions.csv")
    kept.to_parquet(out_dir / "mentions.parquet", index=False)
    daily = daily_mention_counts(kept)
    daily.to_parquet(out_dir / "daily_mentions.parquet", index=False)
    dropped = n_in - len(kept)
    print(f"ambiguity filter: dropped {dropped:,} of {n_in:,} mentions ({dropped / max(n_in, 1):.1%}); "
          f"{(decisions['decision'] == 'cashtags only').sum():,} tickers limited to cashtags; "
          f"excluded: {', '.join(decisions.index[decisions['decision'].str.startswith('excluded')]) or 'none'}")
    return daily


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Extract ticker mentions from interim posts/comments parquet files.",
        epilog="Example: python -m src.nlp.tickers --inputs data/interim/posts.parquet "
               "data/interim/comments.parquet --symbols data/raw/nasdaqlisted.txt "
               "data/raw/otherlisted.txt --extra BBBY --out data/processed",
    )
    p.add_argument("--inputs", nargs="+")
    p.add_argument("--symbols", nargs="+", default=[],
                   help="NASDAQ Trader nasdaqlisted.txt / otherlisted.txt files")
    p.add_argument("--universe-csv", nargs="+", default=[],
                   help="CSV(s) with a `ticker` column, e.g. data/reference/tickers_2021.csv "
                        "from src.data.universe; combined with --symbols")
    p.add_argument("--extra", nargs="*", default=["BBBY"],
                   help="Delisted or missing tickers to add to the universe")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--skip-extract", action="store_true",
                   help="Reuse OUT/mentions_raw.parquet and only redo the ambiguity filter")
    args = p.parse_args(argv)

    if args.skip_extract:
        raw = pd.read_parquet(Path(args.out) / "mentions_raw.parquet")
        daily = finalize(raw, args.out)
    else:
        if not args.inputs or not (args.symbols or args.universe_csv):
            p.error("--inputs and --symbols or --universe-csv are required unless --skip-extract")
        symbols = set(load_nasdaqtrader(*args.symbols)) if args.symbols else set()
        for path in args.universe_csv:
            symbols |= set(pd.read_csv(path, dtype=str)["ticker"].dropna().str.upper())
        universe = build_universe(symbols, extra=args.extra)
        print(f"universe: {len(universe):,} symbols")
        daily = run_on_parquet(args.inputs, universe, args.out)
    top = daily.groupby("ticker")["mentions"].sum().sort_values(ascending=False).head(20)
    print("\nTop 20 tickers by total mentions:\n" + top.to_string())


def load_labels(path: str | Path, sheet: str = "Label") -> pd.DataFrame:
    """Read a filled-in labeling file (.xlsx or .csv).

    Keeps only rows marked done (an "x" in the `checked` column) and skips the
    example row. Blank `true_tickers` on a checked row means "no tickers".
    """
    path = Path(path)
    df = pd.read_excel(path, sheet_name=sheet, dtype=str) if path.suffix == ".xlsx" else pd.read_csv(path, dtype=str)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "id" in df.columns:
        df = df[df["id"].fillna("").str.upper() != "EXAMPLE"]
    if "checked" in df.columns:
        df = df[df["checked"].fillna("").str.strip().str.lower() == "x"]
    return df.fillna({"predicted": "", "true_tickers": ""})


def repredict(
    labeled: pd.DataFrame,
    universe: frozenset[str],
    cashtag_only: frozenset[str] = frozenset(),
    exclude: frozenset[str] = EXCLUDE,
) -> pd.DataFrame:
    """Recompute `predicted` from each row's text with the CURRENT pipeline, so
    the same hand labels can score a new version (new universe, new filter)."""
    out = labeled.copy()
    out["predicted"] = out["text"].map(lambda t: " ".join(predict(t, universe, cashtag_only, exclude)))
    return out


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
