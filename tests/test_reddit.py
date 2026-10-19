"""Loader tests on tiny synthetic dumps, so they run offline in seconds."""

import json

import pandas as pd
import pytest
import zstandard

from src.data.reddit import (
    comments_in_threads,
    daily_threads,
    dump_to_parquet,
    is_daily_thread,
    iter_zst_ndjson,
    load_kaggle_csv,
)


def epoch(et_string):
    return int(pd.Timestamp(et_string, tz="America/New_York").timestamp())


def write_zst(path, objs, junk_line=False):
    lines = [json.dumps(o) for o in objs]
    if junk_line:
        lines.insert(1, "{not json")
    data = ("\n".join(lines) + "\n").encode()
    path.write_bytes(zstandard.ZstdCompressor().compress(data))


@pytest.fixture
def dumps(tmp_path):
    posts = [
        {"id": "aaa", "created_utc": epoch("2021-01-27 07:00"), "author": "u1",
         "title": "Daily Discussion Thread for January 27, 2021", "selftext": "",
         "score": 10, "num_comments": 50000, "link_flair_text": "Daily Discussion"},
        {"id": "bbb", "created_utc": epoch("2021-01-27 17:00"), "author": "AutoModerator",
         "title": "What Are Your Moves Tomorrow, January 28, 2021", "selftext": "",
         "score": 5, "num_comments": 30000, "link_flair_text": None},
        {"id": "ccc", "created_utc": epoch("2021-01-27 12:00"), "author": "[deleted]",
         "title": "GME to the moon", "selftext": "[removed]",
         "score": 900, "num_comments": 400, "link_flair_text": "YOLO"},
        # outside the window
        {"id": "zzz", "created_utc": epoch("2019-01-01 12:00"), "author": "u9",
         "title": "old", "selftext": "", "score": 1, "num_comments": 0},
    ]
    comments = [
        {"id": "c1", "created_utc": epoch("2021-01-27 10:00"), "author": "u3",
         "body": "GME 🚀🚀🚀", "score": 3, "link_id": "t3_aaa", "parent_id": "t3_aaa"},
        {"id": "c2", "created_utc": epoch("2021-01-27 18:00"), "author": "u4",
         "body": "buying AMC calls", "score": 1, "link_id": "t3_bbb", "parent_id": "t3_bbb",
         "author_created_utc": epoch("2021-01-20 00:00")},
        {"id": "c3", "created_utc": epoch("2021-01-27 12:30"), "author": "u5",
         "body": "nice", "score": 2, "link_id": "t3_ccc", "parent_id": "t1_c9"},
    ]
    sp, cp = tmp_path / "subs.zst", tmp_path / "coms.zst"
    write_zst(sp, posts, junk_line=True)
    write_zst(cp, comments)
    return sp, cp, tmp_path


def test_stream_skips_malformed_lines(dumps):
    sp, _, _ = dumps
    assert [o["id"] for o in iter_zst_ndjson(sp)] == ["aaa", "bbb", "ccc", "zzz"]


def test_posts_to_parquet(dumps):
    sp, _, tmp = dumps
    n = dump_to_parquet(sp, "post", tmp / "posts.parquet", "2021-01-01", "2021-02-01")
    df = pd.read_parquet(tmp / "posts.parquet").set_index("id")
    assert n == 3 and "zzz" not in df.index
    # Evening post maps to the next trading day
    assert df.loc["bbb", "trading_day"] == pd.Timestamp("2021-01-28")
    assert df.loc["aaa", "trading_day"] == pd.Timestamp("2021-01-27")
    assert bool(df.loc["ccc", "is_deleted"]) is True
    assert bool(df.loc["ccc", "during_market_hours"]) is True
    assert str(df["created_utc"].dt.tz) == "UTC"


def test_comments_and_daily_filter(dumps):
    sp, cp, tmp = dumps
    dump_to_parquet(sp, "post", tmp / "posts.parquet", "2021-01-01", "2021-02-01")
    dump_to_parquet(cp, "comment", tmp / "comments.parquet", "2021-01-01", "2021-02-01")
    posts = pd.read_parquet(tmp / "posts.parquet")
    comments = pd.read_parquet(tmp / "comments.parquet").set_index("id", drop=False)

    assert comments.loc["c1", "link_id"] == "t3_aaa"[3:]
    assert comments.loc["c2", "author_created_utc"] == pd.Timestamp("2021-01-20 05:00", tz="UTC")

    threads = daily_threads(posts)
    assert set(threads["id"]) == {"aaa", "bbb"}
    in_threads = comments_in_threads(comments.reset_index(drop=True), threads)
    assert set(in_threads["id"]) == {"c1", "c2"}


def test_chunked_writes_match(dumps):
    _, cp, tmp = dumps
    n = dump_to_parquet(cp, "comment", tmp / "c.parquet", "2021-01-01", "2021-02-01", chunk_size=1)
    assert n == 3 and len(pd.read_parquet(tmp / "c.parquet")) == 3


@pytest.mark.parametrize("title, expected", [
    ("Daily Discussion Thread for March 3, 2021", True),
    ("What Are Your Moves Tomorrow, March 04, 2021", True),
    ("Weekend Discussion Thread for the Weekend of March 05, 2021", True),
    ("GME DD: why it's undervalued", False),
    (None, False),
])
def test_is_daily_thread(title, expected):
    assert is_daily_thread(title) is expected


def test_kaggle_csv(tmp_path):
    path = tmp_path / "reddit_wsb.csv"
    pd.DataFrame({
        "title": ["GME gang"], "score": [10], "id": ["k1"], "url": ["x"],
        "comms_num": [4], "created": [float(epoch("2021-01-29 18:00"))],
        "body": [None], "timestamp": ["2021-01-29 18:00:00"],
    }).to_csv(path, index=False)
    df = load_kaggle_csv(path)
    assert df.loc[0, "trading_day"] == pd.Timestamp("2021-02-01")
    assert df.loc[0, "text"] == ""


def test_daily_threads_rejects_user_lookalikes():
    posts = pd.DataFrame({
        "id": ["off", "unpinned", "overflow", "fake1", "fake2"],
        "author": ["VisualMod", "AutoModerator", "premier_", "kickrocks13", "grifev2019"],
        "title": ["Daily Discussion Thread for March 3, 2021",
                  "Unpinned Daily Discussion Thread for February 01, 2021",
                  "The GME Thread Part 2 for January 26, 2021",
                  "Where is the Daily Discussion Thread?",
                  "AMC DAILY THREAD... daily discussion for AMC"],
        "flair": ["Daily Discussion", "Discussion", "Daily Discussion", "Discussion", "YOLO"],
    })
    assert set(daily_threads(posts)["id"]) == {"off", "unpinned", "overflow"}
    assert len(daily_threads(posts, loose=True)) == 5
