import io

import pandas as pd
import pytest

from src.nlp.tickers import (
    build_universe,
    daily_mention_counts,
    extract_mentions,
    find_tickers,
    load_nasdaqtrader,
    sample_for_labeling,
    score_labels,
)

# Real tickers, including several that collide with common words.
UNIVERSE = build_universe(
    "GME AMC BB NOK PLTR TSLA SPY A IT ALL DD CEO FOR ARE ON YOLO NOW TD F T NVDA".split(),
    extra=["BBBY"],
)


def tick(text):
    return [m.ticker for m in find_tickers(text, UNIVERSE)]


def sources(text):
    return {m.ticker: m.source for m in find_tickers(text, UNIVERSE)}


@pytest.mark.parametrize("text, expected", [
    ("GME to the moon 🚀🚀🚀", ["GME"]),
    ("Bought $AMC and $gme calls", ["AMC", "GME"]),
    ("gme and amc gang", ["GME", "AMC"]),            # lowercase allow-list
    ("Holding PLTR, TSLA. Also NOK!", ["PLTR", "TSLA", "NOK"]),
    ("BBBY squeeze?", ["BBBY"]),                      # delisted, added via extra
    ("GME GME GME", ["GME"]),                         # once per text
    ("(GME)", ["GME"]),
])
def test_finds_tickers(text, expected):
    assert tick(text) == expected


@pytest.mark.parametrize("text", [
    "DD: why I'm ALL IN on this",       # DD, ALL are stoplisted words
    "The CEO said IT is going great",
    "YOLO, what ARE you doing FOR fun",
    "I want A new car",                  # single letters never bare-match
    "it is all for now",                 # lowercase common words
    "MOONING HARD",                      # not in universe
    "THE BEST DAY",
])
def test_rejects_false_positives(text):
    assert tick(text) == []


def test_cashtag_overrides_stoplist():
    assert tick("loading up on $ALL and $F") == ["ALL", "F"]


def test_ignores_tokens_inside_words_and_urls():
    assert tick("GMEgang https://x.com/AMC email@SPY.com r/NVDA u/TSLA") == []
    assert tick("see https://x.com/a then buy GME") == ["GME"]


def test_dollar_amounts_are_not_tickers():
    assert tick("made $500 on $5 calls") == []


def test_source_priority():
    s = sources("$GME is great, GME forever, gme")
    assert s == {"GME": "cashtag"}
    assert sources("bought gme")["GME"] == "lowercase"


def test_empty_and_none():
    assert find_tickers(None, UNIVERSE) == []
    assert find_tickers("", UNIVERSE) == []


def test_build_universe_drops_odd_symbols():
    u = build_universe(["GME", "brk.b", "GME.WS", " amc ", None, "TOOLONG"])
    assert u == frozenset({"GME", "AMC"})


def test_load_nasdaqtrader(tmp_path):
    nasdaq = tmp_path / "nasdaqlisted.txt"
    nasdaq.write_text(
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
        "AMC|AMC Entertainment|Q|N|N|100|N|N\n"
        "ZXZZT|Test stock|Q|Y|N|100|N|N\n"
        "File Creation Time: 1017202614:00|||||||\n"
    )
    other = tmp_path / "otherlisted.txt"
    other.write_text(
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
        "GME|GameStop|N|GME|N|100|N|GME\n"
        "File Creation Time: 1017202614:00|||||||\n"
    )
    assert load_nasdaqtrader(nasdaq, other) == {"AMC", "GME"}


@pytest.fixture
def posts():
    return pd.DataFrame({
        "id": ["p1", "p2", "c1", "c2"],
        "kind": ["post", "post", "comment", "comment"],
        "trading_day": pd.to_datetime(["2021-01-27", "2021-01-27", "2021-01-27", "2021-01-28"]),
        "author": ["u1", "u1", "u2", "u3"],
        "score": [10, 5, 1, 2],
        "title": ["GME DD", "AMC?", None, None],
        "text": ["GME is undervalued. $GME", "", "gme and $AMC", "nothing here"],
    })


def test_extract_mentions(posts):
    m = extract_mentions(posts, UNIVERSE)
    assert sorted(zip(m["id"], m["ticker"])) == [
        ("c1", "AMC"), ("c1", "GME"), ("p1", "GME"), ("p2", "AMC")]
    assert set(m.columns) >= {"id", "trading_day", "author", "ticker", "source"}


def test_daily_counts(posts):
    counts = daily_mention_counts(extract_mentions(posts, UNIVERSE)).set_index(["ticker", "trading_day"])
    gme = counts.loc[("GME", pd.Timestamp("2021-01-27"))]
    assert gme["mentions"] == 2 and gme["unique_authors"] == 2
    amc = counts.loc[("AMC", pd.Timestamp("2021-01-27"))]
    assert amc["mentions"] == 2 and amc["unique_authors"] == 2


def test_labeling_roundtrip(posts):
    s = sample_for_labeling(posts, UNIVERSE, n=4, seed=1)
    assert {"id", "text", "predicted", "true_tickers"} <= set(s.columns)
    # simulate a labeler: c1 really mentions only GME, p1 is right
    labels = pd.DataFrame({
        "predicted": ["GME", "AMC GME", ""],
        "true_tickers": ["gme", "GME", "TSLA"],
    })
    r = score_labels(labels)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 1)
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(2 / 3)


def test_run_on_parquet(posts, tmp_path):
    from src.nlp.tickers import run_on_parquet
    path = tmp_path / "posts.parquet"
    posts.to_parquet(path, index=False)
    daily = run_on_parquet([path], UNIVERSE, tmp_path / "out", batch_size=2)
    assert (tmp_path / "out" / "mentions.parquet").exists()
    assert daily["mentions"].sum() == 4


# ---- contractions, possessives, quotes

@pytest.mark.parametrize("text, expected", [
    ("I DON'T KNOW", []),                     # DON'T is not DON
    ("WE'RE buying", []),                     # 'RE after an apostrophe
    ("GME's short interest is huge", ["GME"]),  # possessive still counts
    ("bought 'GME' today", ["GME"]),          # quoted ticker still counts
    ("AMC’s float", ["AMC"]),                 # curly apostrophe
])
def test_apostrophes(text, expected):
    u = build_universe(["DON", "RE", "GME", "AMC"])
    assert [m.ticker for m in find_tickers(text, u)] == expected


# ---- ambiguity filter

def make_mentions(spec):
    rows = []
    for ticker, source, n in spec:
        rows += [{"id": f"{ticker}{source}{i}", "trading_day": pd.Timestamp("2021-01-27"),
                  "author": f"u{i}", "ticker": ticker, "source": source} for i in range(n)]
    return pd.DataFrame(rows)


def test_drop_ambiguous_uses_cashtag_evidence():
    from src.nlp.tickers import drop_ambiguous
    m = make_mentions([
        ("JUST", "bare", 500),                         # word, never cashtagged -> drop
        ("HOOD", "bare", 400), ("HOOD", "cashtag", 60),   # word, 13% cashtags -> keep
        ("GAME", "bare", 490), ("GAME", "cashtag", 10),   # word, 2% -> cashtags only
        ("TD", "bare", 300),                           # two letters, no evidence -> drop
        ("SPY", "bare", 900), ("SPY", "cashtag", 10),     # word but allow-listed -> keep
        ("PLTR", "bare", 100),                         # not a word -> untouched
    ])
    words = frozenset({"JUST", "HOOD", "GAME", "SPY"})
    kept, dec = drop_ambiguous(m, common_words=words)
    counts = kept.groupby("ticker").size().to_dict()
    assert "JUST" not in counts and "TD" not in counts
    assert counts["HOOD"] == 460 and counts["SPY"] == 910 and counts["PLTR"] == 100
    assert counts["GAME"] == 10                         # its cashtags survive
    assert dec.loc["GAME", "decision"] == "cashtags only"
    assert dec.loc["HOOD", "decision"] == "keep (cashtag evidence)"
    assert dec.loc["JUST", "dropped"] == 500
    assert "PLTR" not in dec.index and "SPY" not in dec.index


def test_min_cashtags_guard():
    from src.nlp.tickers import drop_ambiguous
    m = make_mentions([("RARE", "bare", 10), ("RARE", "cashtag", 5)])  # 33% but only 5 cashtags
    kept, dec = drop_ambiguous(m, common_words=frozenset({"RARE"}))
    assert dec.loc["RARE", "decision"] == "cashtags only" and len(kept) == 5


def test_word_list_ships_and_excludes_tickers_we_rely_on():
    from src.nlp.tickers import load_common_words
    words = load_common_words()
    assert {"JUST", "TIME", "LINE", "KNOW"} <= words
    assert not {"GME", "PLTR", "TSLA", "NOK", "CLOV"} & words


# ---- exclude list, final predictions, label loading

def test_exclude_drops_joke_cashtags_entirely():
    from src.nlp.tickers import drop_ambiguous
    m = make_mentions([("ROPE", "cashtag", 80), ("ROPE", "bare", 10), ("GME", "bare", 5)])
    kept, dec = drop_ambiguous(m, common_words=frozenset({"ROPE"}))
    assert set(kept["ticker"]) == {"GME"}
    assert dec.loc["ROPE", "decision"] == "excluded (joke cashtag)"
    assert dec.loc["ROPE", "dropped"] == 90


def test_predict_matches_pipeline():
    from src.nlp.tickers import predict
    u = build_universe(["GME", "JUST", "ROPE", "HOOD"])
    text = "JUST bought GME, buying $ROPE, HOOD and $JUST"
    assert sorted(predict(text, u, cashtag_only=frozenset({"JUST"}))) == ["GME", "HOOD", "JUST"]


def test_load_labels_xlsx_and_csv(tmp_path):
    from src.nlp.tickers import load_labels
    df = pd.DataFrame({
        "id": ["EXAMPLE", "a1", "a2", "a3"],
        "text": ["ex", "GME up", "nothing", "not done"],
        "predicted": ["GME", "GME", None, "AMC"],
        "true_tickers": ["GME", "gme", None, None],
        "checked": ["x", "x", "X ", None],
    })
    for path in (tmp_path / "l.xlsx", tmp_path / "l.csv"):
        if path.suffix == ".xlsx":
            df.to_excel(path, sheet_name="Label", index=False)
        else:
            df.to_csv(path, index=False)
        out = load_labels(path)
        assert out["id"].tolist() == ["a1", "a2"]          # example and unchecked rows skipped
        assert score_labels(out)["tp"] == 1 and score_labels(out)["fp"] == 0
