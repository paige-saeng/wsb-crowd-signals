"""Load r/wallstreetbets posts and comments into one clean table.

Primary source: Arctic Shift per-subreddit dumps (Academic Torrents).
These are zstandard-compressed newline-delimited JSON, one Reddit object per
line, covering the full history of the subreddit with comments and authors.
    wallstreetbets_submissions.zst
    wallstreetbets_comments.zst

Secondary source (quick prototyping only): the Kaggle "Reddit WallStreetBets
Posts" CSV. It has posts only (no comments) and no author column, so it can't
support the bot analysis or Daily Discussion work.

Usage (from the repo root):
    python -m src.data.reddit \
        --submissions data/raw/wallstreetbets_submissions.zst \
        --comments    data/raw/wallstreetbets_comments.zst \
        --start 2020-01-01 --end 2023-01-01 \
        --out data/interim

Writes:
    data/interim/posts.parquet
    data/interim/comments.parquet          (all comments in the window)
    data/interim/daily_threads.parquet     (Daily Discussion / Moves Tomorrow posts)

The comments file is tens of GB uncompressed, so it is streamed and written
in chunks; memory stays flat.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

from src.data.calendar import assign_trading_day, trading_sessions

# Threads where most real-time chatter happens. Titles have varied over the
# years ("Daily Discussion Thread for January 27, 2021", "What Are Your Moves
# Tomorrow, January 27, 2021", "Weekend Discussion Thread ..."), so match loosely.
DAILY_THREAD_RE = re.compile(
    r"daily discussion|what are your moves tomorrow|weekend discussion",
    re.IGNORECASE,
)
DAILY_FLAIRS = frozenset({"Daily Discussion", "Weekend Discussion"})
# Accounts that post the official threads. Checked against 2021 data; if you
# extend the window, rerun the audit in the README to see if this changed.
MOD_AUTHORS = frozenset({"AutoModerator", "VisualMod", "OPINION_IS_UNPOPULAR"})

DELETED_MARKERS = {"[deleted]", "[removed]"}

# Unified schema for posts and comments.
COLUMNS = [
    "id",               # Reddit id without prefix, e.g. "l6wu59"
    "kind",             # "post" or "comment"
    "created_utc",      # tz-aware UTC
    "trading_day",      # first session whose close is after created_utc
    "during_market_hours",
    "author",
    "title",            # posts only
    "text",             # selftext for posts, body for comments
    "score",
    "num_comments",     # posts only
    "link_id",          # comments: the post they belong to, without "t3_"
    "parent_id",        # comments: parent object, with prefix ("t1_" or "t3_")
    "flair",            # post flair, e.g. "DD", "YOLO", "Daily Discussion"
    "author_created_utc",  # present in some older records only
    "is_deleted",       # author or text removed
]


# ---------------------------------------------------------------- reading

def iter_zst_ndjson(path: str | Path) -> Iterator[dict]:
    """Stream JSON objects from a .zst NDJSON dump, one line at a time.

    Reddit dumps use a long compression window, so max_window_size must be raised.
    Malformed lines are skipped rather than crashing a multi-hour run.
    """
    dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        for line in text:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _strip_prefix(value):
    if isinstance(value, str) and "_" in value[:3]:
        return value.split("_", 1)[1]
    return value


def normalize_record(obj: dict, kind: str) -> dict:
    """Map one raw dump object to the unified schema (minus trading_day)."""
    is_post = kind == "post"
    text = obj.get("selftext" if is_post else "body") or ""
    author = obj.get("author")
    return {
        "id": obj.get("id"),
        "kind": kind,
        "created_utc": obj.get("created_utc"),
        "author": author,
        "title": obj.get("title") if is_post else None,
        "text": text,
        "score": obj.get("score"),
        "num_comments": obj.get("num_comments") if is_post else None,
        "link_id": None if is_post else _strip_prefix(obj.get("link_id")),
        "parent_id": None if is_post else obj.get("parent_id"),
        "flair": obj.get("link_flair_text") if is_post else None,
        "author_created_utc": obj.get("author_created_utc"),
        "is_deleted": (author in DELETED_MARKERS) or (text.strip() in DELETED_MARKERS),
    }


def _to_frame(rows: list[dict], sessions: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    df = df.dropna(subset=["created_utc", "id"])
    aligned = assign_trading_day(df["created_utc"].astype("int64").to_numpy(), sessions)
    df["created_utc"] = aligned["created_utc"].to_numpy()
    df["trading_day"] = aligned["trading_day"].to_numpy()
    df["during_market_hours"] = aligned["during_market_hours"].to_numpy()
    df["author_created_utc"] = pd.to_datetime(
        pd.to_numeric(df["author_created_utc"], errors="coerce"), unit="s", utc=True
    )
    for col in ["score", "num_comments"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["id", "kind", "author", "title", "text", "link_id", "parent_id", "flair"]:
        df[col] = df[col].astype("string")
    return df[COLUMNS]


def dump_to_parquet(
    path: str | Path,
    kind: str,
    out_path: str | Path,
    start: str,
    end: str,
    keep: Callable[[dict], bool] | None = None,
    chunk_size: int = 200_000,
) -> int:
    """Stream a dump into a parquet file, keeping records in [start, end).

    Returns the number of rows written.
    """
    t0 = pd.Timestamp(start, tz="UTC").timestamp()
    t1 = pd.Timestamp(end, tz="UTC").timestamp()
    sessions = trading_sessions(
        pd.Timestamp(start) - pd.Timedelta(days=7), pd.Timestamp(end) + pd.Timedelta(days=14)
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    rows: list[dict] = []
    written = 0

    def flush():
        nonlocal writer, rows, written
        if not rows:
            return
        table = pa.Table.from_pandas(_to_frame(rows, sessions), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        else:
            table = table.cast(writer.schema)
        writer.write_table(table)
        written += table.num_rows
        rows = []

    for obj in iter_zst_ndjson(path):
        try:
            ts = float(obj.get("created_utc"))
        except (TypeError, ValueError):
            continue
        if not (t0 <= ts < t1):
            continue
        if keep is not None and not keep(obj):
            continue
        rows.append(normalize_record(obj, kind))
        if len(rows) >= chunk_size:
            flush()
    flush()
    if writer is not None:
        writer.close()
    return written


# ---------------------------------------------------------------- helpers

def is_daily_thread(title) -> bool:
    return isinstance(title, str) and bool(DAILY_THREAD_RE.search(title))


def daily_threads(posts: pd.DataFrame, loose: bool = False) -> pd.DataFrame:
    """Official Daily Discussion / Moves Tomorrow / Weekend threads.

    A post counts if EITHER
      - its flair is "Daily Discussion" or "Weekend Discussion" (this also
        catches the squeeze-era overflow threads like "The GME Thread Part 2",
        which volunteers posted with the flair), OR
      - a moderator account posted it with a daily-thread title (catches
        AutoModerator's "Unpinned Daily Discussion" threads, which lack flair).

    Title matching alone is NOT enough: in 2021 about 170 user posts had titles
    like "Where is the Daily Discussion thread?" or "AMC DAILY THREAD".
    Pass loose=True to get the old title-or-flair behaviour.
    """
    flair = posts["flair"].fillna("")
    title_match = posts["title"].map(is_daily_thread).astype(bool)
    if loose:
        return posts[title_match | flair.str.contains("daily discussion", case=False)]
    by_flair = flair.isin(DAILY_FLAIRS)
    by_mod = posts["author"].isin(MOD_AUTHORS).fillna(False).astype(bool) & title_match
    return posts[(by_flair | by_mod).astype(bool)]


def comments_in_threads(comments: pd.DataFrame, threads: pd.DataFrame) -> pd.DataFrame:
    """Comments that belong to the given threads (matched on link_id)."""
    return comments[comments["link_id"].isin(set(threads["id"]))]


def load_kaggle_csv(path: str | Path) -> pd.DataFrame:
    """Load the Kaggle 'Reddit WallStreetBets Posts' CSV into the unified schema.

    Columns there: title, score, id, url, comms_num, created, body, timestamp.
    No author and no comments, so author-based columns are empty.
    """
    raw = pd.read_csv(path)
    rows = [
        {
            "id": r.get("id"),
            "kind": "post",
            "created_utc": r.get("created"),
            "author": None,
            "title": r.get("title"),
            "text": r.get("body") if isinstance(r.get("body"), str) else "",
            "score": r.get("score"),
            "num_comments": r.get("comms_num"),
            "link_id": None,
            "parent_id": None,
            "flair": None,
            "author_created_utc": None,
            "is_deleted": False,
        }
        for r in raw.to_dict("records")
    ]
    ts = pd.to_numeric(raw["created"], errors="coerce")
    sessions = trading_sessions(
        pd.to_datetime(ts.min(), unit="s") - pd.Timedelta(days=7),
        pd.to_datetime(ts.max(), unit="s") + pd.Timedelta(days=14),
    )
    return _to_frame(rows, sessions)


# ---------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--submissions", required=True)
    p.add_argument("--comments")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2023-01-01")
    p.add_argument("--out", default="data/interim")
    p.add_argument("--daily-only", action="store_true",
                   help="Keep only comments inside daily threads (much smaller output).")
    args = p.parse_args(argv)
    out = Path(args.out)

    n = dump_to_parquet(args.submissions, "post", out / "posts.parquet", args.start, args.end)
    print(f"posts: {n:,} rows -> {out / 'posts.parquet'}")

    posts = pd.read_parquet(out / "posts.parquet")
    threads = daily_threads(posts)
    threads.to_parquet(out / "daily_threads.parquet", index=False)
    print(f"daily threads: {len(threads):,} -> {out / 'daily_threads.parquet'}")

    if args.comments:
        keep = None
        if args.daily_only:
            ids = set(threads["id"])
            keep = lambda obj: _strip_prefix(obj.get("link_id")) in ids  # noqa: E731
        n = dump_to_parquet(args.comments, "comment", out / "comments.parquet",
                            args.start, args.end, keep=keep)
        print(f"comments: {n:,} rows -> {out / 'comments.parquet'}")


if __name__ == "__main__":
    main()
