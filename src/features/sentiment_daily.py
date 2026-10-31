"""Daily WSB sentiment per stock, for H3.

Two steps:

    1. score   Every post or comment that mentions a stock gets P(bullish),
               P(bearish), P(neutral) from the sentiment model trained on the
               hand labels (models/sentiment_logreg.joblib, chosen by
               cross-validation on the training labels, not on the test set).
               -> data/processed/text_sentiment.parquet  (one row per text)

    2. daily   Ticker extraction says which stock(s) each text is about
               (data/processed/mentions.parquet). Joining the two gives a
               sentiment for each (text, stock) pair, averaged per stock per
               trading day.
               -> data/processed/daily_sentiment.parquet (one row per stock-day)

Measures per stock-day
    sent_net         mean of P(bullish) - P(bearish) over the day's texts, in
                     [-1, 1]; the main H3 measure. Probabilities rather than
                     hard labels, so an unsure text counts for little.
    bullishness      log((1 + n bullish) / (1 + n bearish)) from hard labels
                     (Antweiler and Frank 2004), a standard alternative.
    sent_net_single  sent_net using only texts that mention just this stock,
                     since a text naming two stocks gets one sentiment for both.
    n_scored         texts behind the average; few texts = a noisy average.

A text that mentions several stocks gets the same sentiment for each of them;
the single-stock measure is the robustness check for that.

Usage
    python -m src.features.sentiment_daily score    # slow: ~2 million texts
    python -m src.features.sentiment_daily daily    # seconds
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.nlp.sentiment import display_text

PARTS = Path("data/processed/text_sentiment_parts")
SCORES = Path("data/processed/text_sentiment.parquet")
DAILY = Path("data/processed/daily_sentiment.parquet")


def predict_probs(model, texts: list[str]) -> pd.DataFrame:
    """P(label) columns from a fitted scikit-learn pipeline, in a fixed order."""
    p = model.predict_proba(texts)
    cols = {c: f"p_{c}" for c in model.classes_}
    return pd.DataFrame(p, columns=[cols[c] for c in model.classes_])[["p_bullish", "p_bearish", "p_neutral"]]


def score_texts(model, ids, interim: str | Path = "data/interim", parts: Path = PARTS,
                batch_size: int = 50_000, log=print) -> pd.DataFrame:
    """Score every post and comment whose id is in `ids`.

    Streams the interim parquet files in batches and saves each batch's scores
    as its own file, so an interrupted run resumes where it stopped."""
    import pyarrow.dataset as ds

    parts.mkdir(parents=True, exist_ok=True)
    want = list(set(ids))
    total, start = len(want), time.time()
    done = 0
    for name in ("posts", "comments"):
        d = ds.dataset(Path(interim) / f"{name}.parquet")
        batches = d.to_batches(columns=["id", "kind", "title", "text"], filter=ds.field("id").isin(want),
                               batch_size=batch_size)
        for k, b in enumerate(batches):
            f = parts / f"{name}_{k:04d}.parquet"
            t = b.to_pandas()
            if t.empty:
                continue
            if f.exists():
                done += len(t)
                continue
            texts = [display_text(kd, ti, tx) for kd, ti, tx in zip(t["kind"], t["title"], t["text"])]
            out = pd.concat([t[["id"]].reset_index(drop=True), predict_probs(model, texts)], axis=1)
            out.to_parquet(f, index=False)
            done += len(t)
            rate = done / max(time.time() - start, 1e-9)
            log(f"  {name} batch {k}: {done:,}/{total:,} texts scored "
                f"(~{(total - done) / max(rate, 1e-9) / 60:.0f} min left)")
    return pd.concat([pd.read_parquet(f) for f in sorted(parts.glob("*.parquet"))], ignore_index=True)


def daily_sentiment(mentions: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Average text sentiment per (ticker, trading_day).

    mentions: id, ticker, trading_day (one row per text-stock pair)
    scores:   id, p_bullish, p_bearish, p_neutral (one row per text)"""
    n_tickers = mentions.groupby("id")["ticker"].nunique()
    m = mentions[["id", "ticker", "trading_day"]].merge(scores, on="id", how="inner")
    m["net"] = m["p_bullish"] - m["p_bearish"]
    m["label"] = m[["p_bullish", "p_bearish", "p_neutral"]].to_numpy().argmax(axis=1)
    m["is_bull"], m["is_bear"] = (m["label"] == 0), (m["label"] == 1)
    m["single"] = m["id"].map(n_tickers).eq(1)
    m["net_single"] = m["net"].where(m["single"])

    g = m.groupby(["ticker", "trading_day"])
    out = pd.DataFrame({
        "n_scored": g.size(),
        "sent_net": g["net"].mean(),
        "p_bullish": g["p_bullish"].mean(),
        "p_bearish": g["p_bearish"].mean(),
        "bull_share": g["is_bull"].mean(),
        "bear_share": g["is_bear"].mean(),
        "n_single": g["single"].sum(),
        "sent_net_single": g["net_single"].mean(),
    })
    nb, nr = g["is_bull"].sum(), g["is_bear"].sum()
    out["bullishness"] = np.log((1 + nb) / (1 + nr))
    return out.reset_index()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="Step 1: score every text that mentions a stock")
    s.add_argument("--model", default="models/sentiment_logreg.joblib")
    s.add_argument("--mentions", default="data/processed/mentions.parquet")
    s.add_argument("--study-only", action="store_true", default=True,
                   help="Only texts mentioning a study stock (default; the panel uses only those)")
    s.add_argument("--batch-size", type=int, default=50_000)
    d = sub.add_parser("daily", help="Step 2: average to one sentiment per stock per day")
    d.add_argument("--mentions", default="data/processed/mentions.parquet")
    args = p.parse_args(argv)

    if args.cmd == "score":
        import joblib
        model = joblib.load(args.model)
        m = pd.read_parquet(args.mentions, columns=["id", "ticker"])
        if args.study_only:
            study = set(Path("data/processed/study_tickers.txt").read_text().split())
            m = m[m["ticker"].isin(study)]
        ids = m["id"].unique()
        print(f"scoring {len(ids):,} texts with {args.model} (resumes if interrupted)")
        scores = score_texts(model, ids, batch_size=args.batch_size)
        scores.to_parquet(SCORES, index=False)
        print(f"-> {SCORES}: {len(scores):,} texts\nnext: python -m src.features.sentiment_daily daily")
    else:
        mentions = pd.read_parquet(args.mentions, columns=["id", "ticker", "trading_day"])
        daily = daily_sentiment(mentions, pd.read_parquet(SCORES))
        daily.to_parquet(DAILY, index=False)
        print(f"-> {DAILY}: {len(daily):,} stock-days, {daily['ticker'].nunique()} tickers; "
              f"median {daily['n_scored'].median():.0f} texts per stock-day")
        print(daily[["sent_net", "bullishness", "bull_share", "bear_share"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
