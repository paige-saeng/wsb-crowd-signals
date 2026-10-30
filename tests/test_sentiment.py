import pandas as pd
import pytest

from src.nlp.sentiment import (classify, confusion, evaluate, load_sentiment_labels, preprocess,
                               vader_wsb_score, wsb_score)


@pytest.mark.parametrize("text, expected", [
    ("GME to the moon 🚀🚀🚀", "bullish"),
    ("loading up on calls, diamond hands", "bullish"),
    ("buying puts, this will drill", "bearish"),
    ("I am a bagholder", "bearish"),
    ("sold everything, its dead", "bearish"),
    ("what time does the market open?", "neutral"),
])
def test_directional_slang(text, expected):
    assert classify(wsb_score(text)) == expected
    assert classify(vader_wsb_score(text)) == expected


def test_negation_flips_direction():
    assert wsb_score("selling") < 0 < wsb_score("not selling")
    assert wsb_score("buying") > 0 > wsb_score("don't buy this")
    assert wsb_score("never shorting this") > 0


def test_phrases_and_emoji_become_tokens():
    t = preprocess("Diamond hands 💎🙌 to the moon 🌈🐻 https://x.com/a")
    assert "diamondhands" in t and "tothemoon" in t and "rainbowbear" in t
    assert "x.com" not in t
    assert " bear " not in t.replace("rainbowbear", "")          # 🌈🐻 is not two separate emoji


def test_scores_bounded():
    s = wsb_score("🚀" * 50 + " calls " * 50)
    assert 0.99 < s <= 1.0


def test_classify_thresholds():
    assert classify(0.05) == "bullish" and classify(-0.05) == "bearish" and classify(0.049) == "neutral"


def test_evaluate_metrics():
    true = pd.Series(["bullish", "bullish", "bearish", "neutral", "neutral"])
    pred = pd.Series(["bullish", "neutral", "bearish", "neutral", "bullish"])
    r = evaluate(true, pred)
    assert r["accuracy"] == pytest.approx(0.6)
    assert r["bullish_precision"] == pytest.approx(0.5) and r["bullish_recall"] == pytest.approx(0.5)
    assert r["bearish_precision"] == 1.0 and r["bearish_recall"] == 1.0
    cm = confusion(true, pred)
    assert cm.loc["bullish", "neutral"] == 1 and cm.loc["neutral", "bullish"] == 1


def test_load_labels(tmp_path):
    df = pd.DataFrame({"#": ["", "1", "2", "3"], "id": ["EXAMPLE", "a", "b", "c"], "text": ["x", "y", "z", "w"],
                       "label": ["bullish", "Bearish ", "neutral", None], "checked": ["x", "x", "X", None]})
    p = tmp_path / "s.xlsx"
    df.to_excel(p, sheet_name="Label", index=False)
    out = load_sentiment_labels(p)
    assert out["id"].tolist() == ["a", "b"] and out["label"].tolist() == ["bearish", "neutral"]


def test_load_labels_rejects_missing_label(tmp_path):
    df = pd.DataFrame({"#": ["1"], "id": ["a"], "text": ["y"], "label": [None], "checked": ["x"]})
    p = tmp_path / "s.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="no valid label"):
        load_sentiment_labels(p)


def test_evaluate_file_reads_stratum_from_raw(tmp_path):
    from src.nlp.sentiment import evaluate_file
    lab = pd.DataFrame({"#": ["1", "2", "3", "4"], "id": ["a", "b", "c", "d"],
                        "text": ["calls 🚀", "buying puts", "market open?", "sold, its dead"],
                        "label": ["bullish", "bearish", "neutral", "bearish"], "checked": ["x"] * 4})
    lab.to_excel(tmp_path / "sentiment_sample.xlsx", sheet_name="Label", index=False)
    pd.DataFrame({"id": list("abcd"), "stratum": ["random", "bearish keyword", "random", "bearish keyword"]}) \
        .to_csv(tmp_path / "sentiment_sample_raw.csv", index=False)
    res = evaluate_file(tmp_path / "sentiment_sample.xlsx")
    assert set(res["subset"]) == {"all", "random part only"}
    best = res[(res.subset == "all") & (res.scorer == "vader_wsb")].iloc[0]
    assert best["accuracy"] == 1.0 and best["n"] == 4


# ------------------------------------------------------------------ training sample

from src.nlp.sentiment import (display_text, draw_training_sample, single_ticker_ids,  # noqa: E402
                               write_label_workbook)


def test_single_ticker_ids_keeps_one_stock_texts():
    m = pd.DataFrame({"id": ["a", "a", "b", "c", "c"], "kind": "comment", "trading_day": "2021-03-01",
                      "ticker": ["GME", "AMC", "TSLA", "PLTR", "XYZ"]})
    out = single_ticker_ids(m, {"GME", "AMC", "TSLA", "PLTR"})
    assert sorted(out["id"]) == ["b", "c"]          # a has two study stocks; XYZ is not a study stock


def test_display_text_post_and_comment():
    assert display_text("post", "Title", "[deleted]") == "Title\n \n[deleted]"
    assert display_text("comment", None, "hi") == "hi"
    assert len(display_text("comment", None, "x" * 5000)) == 1500


def _pool(n=400):
    words = ["buying calls", "loading puts", "what a day", "selling it all"]
    return pd.DataFrame({"id": [f"i{k}" for k in range(n)], "kind": "comment", "trading_day": "2021-05-03",
                         "ticker": "GME", "text": [f"{words[k % 4]} {k}" for k in range(n)]})


def test_training_sample_excludes_test_ids_and_texts():
    pool = _pool()
    pool.loc[5, "text"] = "same comment twice"
    s = draw_training_sample(pool, n=100, bearish_share=0.3, exclude_ids=["i1", "i2"],
                             exclude_texts=["same comment twice", "buying calls 8"], seed=0)
    assert not set(s["id"]) & {"i1", "i2", "i5", "i8"}
    assert s["id"].is_unique and len(s) == 100


def test_training_sample_oversamples_bearish_hints():
    s = draw_training_sample(_pool(), n=100, bearish_share=0.3, seed=0)
    assert (s["stratum"] == "bearish keyword").sum() == 30
    hinted = s[s["stratum"] == "bearish keyword"]["text"]
    assert hinted.str.contains("puts|selling").all()


def test_training_sample_drops_empty_bodies():
    pool = _pool(50)
    pool.loc[:9, "text"] = "[deleted]"
    s = draw_training_sample(pool, n=40, bearish_share=0.0, seed=0)
    assert not (s["text"] == "[deleted]").any()


def test_workbook_roundtrip(tmp_path):
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "How to label"
    for r in range(1, 28):
        ws.cell(r, 1, f"line {r}")
    lab = wb.create_sheet("Label")
    lab.append(["#", "id", "kind", "date", "ticker", "text", "label", "checked", "notes"])
    lab.append([None, "EXAMPLE", "comment", "2021-01-27", "GME", "example", "bullish", "x", "not scored"])
    lab.append([1, "old", "comment", "2021-01-01", "GME", "old text", None, None, None])
    wb.create_sheet("Progress")
    tpl = tmp_path / "tpl.xlsx"
    wb.save(tpl)

    s = draw_training_sample(_pool(), n=20, bearish_share=0.3, seed=0)
    out = tmp_path / "train.xlsx"
    write_label_workbook(s.drop(columns="stratum"), tpl, out, "title", "summary")
    labels = pd.read_excel(out, sheet_name="Label", dtype=str)
    assert list(labels.columns) == ["#", "id", "kind", "date", "ticker", "text", "label", "notes"]
    assert len(labels) == 21 and "old" not in set(labels["id"])          # example + 20 new rows

    # a labeled training file loads without a `checked` column
    labels.loc[1:3, "label"] = ["bullish", "neutral", "bearish"]
    labels.to_csv(tmp_path / "t.csv", index=False)
    assert len(load_sentiment_labels(tmp_path / "t.csv")) == 3
