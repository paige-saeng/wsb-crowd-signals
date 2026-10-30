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
