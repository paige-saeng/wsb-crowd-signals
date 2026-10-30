"""Sentiment of WSB text toward a stock: bullish, bearish, or neutral.

Three scorers, compared on a hand-labeled sample:

    vader      VADER, a general-purpose sentiment model. Blind to WSB slang:
               "GME to the moon 🚀🚀🚀", "buying puts, this will drill" and
               "I am a bagholder" all score 0.0.
    wsb        A WSB lexicon alone: slang, options and emoji terms with
               bullish (+) or bearish (-) weights, with negation handling
               ("not selling" is bullish, "don't buy" is bearish).
    vader_wsb  VADER with the WSB lexicon added to its dictionary, so it keeps
               VADER's grammar rules (negation, intensifiers, caps, "but")
               and gains the slang.
    finbert    Optional: FinBERT, a finance-tuned language model. Too slow for
               13M comments on a laptop CPU; scored on the labeled sample only.

Every scorer returns a compound score in [-1, 1]; `classify` maps it to a
label with VADER's standard cutoffs (>= 0.05 bullish, <= -0.05 bearish).

Usage
    python -m src.nlp.sentiment evaluate --labels data/labels/sentiment_sample.xlsx
    python -m src.nlp.sentiment score    # after choosing a scorer
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd

THRESHOLD = 0.05
LABELS = ("bullish", "bearish", "neutral")

# ------------------------------------------------------------------ WSB lexicon

# Phrases become single tokens before scoring, so the lexicon can weight them.
PHRASES = {
    r"to the moon": " tothemoon ", r"diamond hands?": " diamondhands ", r"paper hands?": " paperhands ",
    r"buy(ing)? the dip": " buythedip ", r"rug ?pull(ed)?": " rugpull ", r"short squeeze": " shortsqueeze ",
    r"gamma squeeze": " shortsqueeze ", r"bag ?holder(s|ing)?": " bagholder ", r"holding the bag": " bagholder ",
    r"loading up": " loadingup ", r"load up": " loadingup ", r"all in": " allin ",
    r"stop loss": " stoploss ", r"took profits?|taking profits?": " takeprofit ",
    r"(this|it)('s|s| is) dead": " itsdead ", r"to zero": " tozero ",
}
EMOJI = {
    "🚀": " rocket ", "💎": " diamond ", "🙌": " hodlhands ", "🦍": " ape ", "📈": " chartup ",
    "📉": " chartdown ", "🐂": " bull ", "🐻": " bear ", "🌈🐻": " rainbowbear ", "🌕": " moon ",
    "🍗": " tendies ", "💀": " skull ", "🩸": " blood ", "🔥": " fire ", "🤡": " clown ",
}

# Weights on VADER's scale (-4 .. +4). Directional trading language, not mood.
WSB_LEXICON = {
    # bullish
    "tothemoon": 3.0, "moon": 2.0, "mooning": 2.5, "rocket": 2.5, "rockets": 2.5, "chartup": 2.0,
    "diamondhands": 2.5, "diamond": 1.5, "hodlhands": 1.0, "hodl": 2.0, "hodling": 2.0, "hold": 1.0,
    "holding": 1.0, "tendies": 2.5, "calls": 2.0, "call": 1.0, "buy": 1.5, "buying": 1.5, "bought": 1.2,
    "long": 1.0, "bull": 2.0, "bullish": 2.5, "squeeze": 2.0, "shortsqueeze": 2.5, "loadingup": 2.5,
    "allin": 2.0, "buythedip": 2.5, "undervalued": 2.5, "breakout": 2.0, "rip": 0.0, "ripping": 2.0,
    "printing": 2.0, "lambo": 2.0, "gains": 1.5, "green": 1.0, "ape": 1.0, "apes": 1.0, "yolo": 1.0,
    "fire": 0.5,
    # bearish
    "puts": -2.0, "put": -1.0, "short": -1.5, "shorting": -2.0, "shorted": -1.5, "sell": -1.5,
    "selling": -1.5, "sold": -1.2, "dump": -2.5, "dumping": -2.5, "dumped": -2.0, "crash": -2.5,
    "crashing": -2.5, "drill": -2.5, "drilling": -2.5, "tank": -2.0, "tanking": -2.5, "bear": -2.0,
    "bearish": -2.5, "rainbowbear": -2.0, "chartdown": -2.0, "overvalued": -2.5, "bagholder": -2.5,
    "bags": -1.5, "paperhands": -1.5, "rugpull": -3.0, "itsdead": -3.0, "tozero": -3.0, "red": -1.0,
    "bleeding": -2.0, "blood": -1.5, "guh": -2.5, "loss": -1.5, "losses": -1.5, "fraud": -3.0,
    "scam": -3.0, "bubble": -2.0, "stoploss": -1.0, "takeprofit": -0.5, "clown": -1.0, "skull": -1.0,
}
NEGATORS = {"not", "no", "never", "dont", "don't", "didnt", "didn't", "isnt", "isn't", "wont", "won't",
            "cant", "can't", "stop", "aint", "ain't", "nobody", "neither", "nor", "without"}
TOKEN_RE = re.compile(r"[a-z']+")


def preprocess(text: str | None) -> str:
    """Lowercase, map emojis and phrases to lexicon tokens, drop links."""
    if not isinstance(text, str):
        return ""
    t = re.sub(r"https?://\S+|www\.\S+", " ", text)
    for e, w in sorted(EMOJI.items(), key=lambda kv: -len(kv[0])):   # 🌈🐻 before 🐻
        t = t.replace(e, w)
    t = t.lower()
    for pat, rep in PHRASES.items():
        t = re.sub(rf"\b(?:{pat})\b", rep, t)
    return t


def wsb_score(text: str | None, window: int = 3) -> float:
    """Lexicon-only compound score in [-1, 1], with negation flipping a term
    when a negator appears within `window` tokens before it."""
    toks = TOKEN_RE.findall(preprocess(text))
    total = 0.0
    for i, tok in enumerate(toks):
        w = WSB_LEXICON.get(tok)
        if not w:
            continue
        if any(t in NEGATORS for t in toks[max(0, i - window):i]):
            w = -0.74 * w                       # VADER's negation factor
        total += w
    return total / math.sqrt(total * total + 15) if total else 0.0   # VADER's normalization


# ------------------------------------------------------------------ VADER

_vader = {}


def _analyzer(with_wsb: bool):
    if with_wsb not in _vader:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        a = SentimentIntensityAnalyzer()
        if with_wsb:
            a.lexicon.update(WSB_LEXICON)
        _vader[with_wsb] = a
    return _vader[with_wsb]


def vader_score(text: str | None) -> float:
    return _analyzer(False).polarity_scores(text if isinstance(text, str) else "")["compound"]


def vader_wsb_score(text: str | None) -> float:
    return _analyzer(True).polarity_scores(preprocess(text))["compound"]


def finbert_scores(texts: list[str], batch_size: int = 32) -> list[float]:
    """P(positive) - P(negative) from FinBERT. Needs `pip install transformers torch`
    and downloads the model (~440 MB) on first use."""
    from transformers import pipeline

    clf = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)
    out = []
    for i in range(0, len(texts), batch_size):
        for res in clf([t if isinstance(t, str) else "" for t in texts[i:i + batch_size]]):
            p = {r["label"].lower(): r["score"] for r in res}
            out.append(p.get("positive", 0.0) - p.get("negative", 0.0))
    return out


SCORERS = {"vader": vader_score, "wsb": wsb_score, "vader_wsb": vader_wsb_score}


def classify(score: float, threshold: float = THRESHOLD) -> str:
    if score >= threshold:
        return "bullish"
    if score <= -threshold:
        return "bearish"
    return "neutral"


# ------------------------------------------------------------------ evaluation

def evaluate(true: pd.Series, pred: pd.Series) -> dict:
    """Accuracy, macro-F1 and per-class precision/recall for 3-way labels."""
    true = true.str.strip().str.lower()
    pred = pred.str.strip().str.lower()
    out = {"n": int(len(true)), "accuracy": float((true == pred).mean())}
    f1s = []
    for c in LABELS:
        tp = int(((pred == c) & (true == c)).sum())
        p_den, r_den = int((pred == c).sum()), int((true == c).sum())
        prec = tp / p_den if p_den else float("nan")
        rec = tp / r_den if r_den else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if p_den and r_den and (prec + rec) else 0.0
        out[f"{c}_precision"], out[f"{c}_recall"], out[f"{c}_n"] = prec, rec, r_den
        f1s.append(f1)
    out["macro_f1"] = sum(f1s) / len(f1s)
    return out


def confusion(true: pd.Series, pred: pd.Series) -> pd.DataFrame:
    return pd.crosstab(true.str.strip().str.lower().rename("labeled"),
                       pred.str.strip().str.lower().rename("predicted")).reindex(
        index=LABELS, columns=LABELS, fill_value=0)


def load_sentiment_labels(path: str | Path, sheet: str = "Label") -> pd.DataFrame:
    """Checked rows of a filled-in sentiment labeling file (.xlsx or .csv)."""
    path = Path(path)
    df = pd.read_excel(path, sheet_name=sheet, dtype=str) if path.suffix == ".xlsx" else pd.read_csv(path, dtype=str)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df[df["id"].fillna("").str.upper() != "EXAMPLE"]
    df = df[df["checked"].fillna("").str.strip().str.lower() == "x"]
    df["label"] = df["label"].fillna("").str.strip().str.lower()
    bad = df[~df["label"].isin(LABELS)]
    if len(bad):
        raise ValueError(f"{len(bad)} checked rows have no valid label (#{', #'.join(bad['#'].astype(str))})")
    return df


def evaluate_file(path: str | Path, with_finbert: bool = False) -> pd.DataFrame:
    """Score every scorer on the labeled file. The sampling stratum is kept out
    of the spreadsheet (so it cannot sway labeling) and read from
    sentiment_sample_raw.csv in the same folder, when present."""
    lab = load_sentiment_labels(path)
    raw = Path(path).with_name("sentiment_sample_raw.csv")
    if "stratum" not in lab.columns and raw.exists():
        lab = lab.merge(pd.read_csv(raw, dtype=str)[["id", "stratum"]], on="id", how="left")
    scores = {name: lab["text"].map(f) for name, f in SCORERS.items()}
    if with_finbert:
        scores["finbert"] = pd.Series(finbert_scores(lab["text"].tolist()), index=lab.index)
    rows = []
    subsets = {"all": lab.index}
    if "stratum" in lab.columns:
        subsets["random part only"] = lab.index[lab["stratum"] == "random"]
    for subset, idx in subsets.items():
        for name, s in scores.items():
            rows.append({"subset": subset, "scorer": name,
                         **evaluate(lab.loc[idx, "label"], s.loc[idx].map(classify))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate", help="Compare scorers on the hand-labeled sample")
    e.add_argument("--labels", default="data/labels/sentiment_sample.xlsx")
    e.add_argument("--finbert", action="store_true", help="Also score with FinBERT (slow, large download)")
    e.add_argument("--out", default="reports/tables/sentiment_eval.csv")
    args = p.parse_args(argv)

    if args.cmd == "evaluate":
        res = evaluate_file(args.labels, with_finbert=args.finbert)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.out, index=False)
        cols = ["subset", "scorer", "n", "accuracy", "macro_f1", "bullish_recall", "bearish_recall",
                "bullish_precision", "bearish_precision"]
        print(res[cols].round(3).to_string(index=False))
        lab = load_sentiment_labels(args.labels)
        best = res[res.subset == "all"].sort_values("macro_f1").iloc[-1]["scorer"]
        f = SCORERS.get(best)
        if f:
            print(f"\nConfusion matrix, {best} (rows = your labels):")
            print(confusion(lab["label"], lab["text"].map(f).map(classify)).to_string())
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
