"""Supervised sentiment model: trained on the hand-labeled training sample,
graded on the 300 hand-labeled test texts it never sees.

Features
    word TF-IDF   1-2 word n-grams of the preprocessed text (emojis and WSB
                  phrases already mapped to tokens, so "to the moon" and the
                  rocket emoji become words), plus
    char TF-IDF   2-5 character n-grams, which catch misspellings and slang
                  variants ("tendies", "tendiez", "sqqueeze"), plus
    wsb_score     the WSB lexicon score, as one extra numeric feature.

Models
    logreg  logistic regression, class_weight="balanced" so the rare bearish
            class is not ignored. C is chosen by 5-fold CV on the TRAINING
            labels only (macro-F1).
    lgbm    LightGBM on the same features, for comparison.

Learning curve
    The model is refit on random subsets of the training labels (stratified,
    5 repeats per size) and scored on the test labels, to show whether more
    labels would still help.

Usage
    python -m src.nlp.sentiment_model train            # TF-IDF models only
    python -m src.nlp.sentiment_model embed            # step 1: sentence embeddings, saved to disk
    python -m src.nlp.sentiment_model train --embed    # step 2: also train on the saved embeddings
    python -m src.nlp.sentiment_model finance-models   # finance-specific pretrained models, zero-shot

Embeddings are computed in their own run because on macOS PyTorch and
LightGBM/scikit-learn in one Python process can crash (segmentation fault).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.nlp.sentiment import (SCORERS, classify, confusion, evaluate, load_sentiment_labels, preprocess,
                               read_label_sheet, wsb_score)

C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # ~90 MB, runs on a laptop CPU

# Sentence embeddings are slow to compute, so each text is encoded once per
# session and reused across CV folds and learning-curve refits.
_EMB_CACHE: dict[tuple[str, str], np.ndarray] = {}
_ENCODER = {"fn": None, "name": EMBED_MODEL, "offline": False}
EMB_FILE = Path("data/processed/sentiment_embeddings.npz")


def set_encoder(fn=None, name: str = EMBED_MODEL) -> None:
    """Use `fn(list[str]) -> array` as the encoder (tests pass a fake one);
    None = load `name` with sentence-transformers on first use."""
    _ENCODER.update(fn=fn, name=name, offline=False)


def encode(texts) -> np.ndarray:
    name = _ENCODER["name"]
    todo = [t for t in dict.fromkeys(texts) if (name, t) not in _EMB_CACHE]
    if todo:
        if _ENCODER["fn"] is None and _ENCODER["offline"]:
            raise RuntimeError(f"{len(todo)} texts have no saved embedding. Run "
                               "`python -m src.nlp.sentiment_model embed` first (again if you labeled new files).")
        if _ENCODER["fn"] is None:
            # macOS: PyTorch and LightGBM/scikit-learn each ship an OpenMP library;
            # loading both in one process can crash Python. One thread, CPU only.
            import os
            for k, v in {"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1",
                         "TOKENIZERS_PARALLELISM": "false"}.items():
                os.environ.setdefault(k, v)
            from sentence_transformers import SentenceTransformer   # pip install sentence-transformers
            st = SentenceTransformer(name, device="cpu")
            _ENCODER["fn"] = lambda xs: st.encode(xs, batch_size=64, normalize_embeddings=True,
                                                  show_progress_bar=len(xs) > 500)
        vecs = _ENCODER["fn"]([t if isinstance(t, str) else "" for t in todo])
        for t, v in zip(todo, vecs):
            _EMB_CACHE[(name, t)] = np.asarray(v, dtype=np.float32)
    return np.vstack([_EMB_CACHE[(name, t)] for t in texts])


def save_embeddings(texts, path: str | Path = EMB_FILE) -> int:
    """Encode texts and save them, so training can run in a separate Python
    process that never loads PyTorch (on macOS, PyTorch and LightGBM/scikit-learn
    in one process can crash with a segmentation fault)."""
    texts = list(dict.fromkeys(t for t in texts if isinstance(t, str)))
    vecs = encode(texts)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, model=np.array(_ENCODER["name"]), texts=np.array(texts, dtype=object),
                        vecs=vecs)
    return len(texts)


def load_embeddings(path: str | Path = EMB_FILE) -> int:
    """Load saved embeddings into the cache and stop encode() from loading PyTorch."""
    d = np.load(path, allow_pickle=True)
    name = str(d["model"])
    for t, v in zip(d["texts"], d["vecs"]):
        _EMB_CACHE[(name, str(t))] = v
    _ENCODER.update(name=name, offline=True)
    return len(d["texts"])


class Embeddings:
    """Sentence embedding of each text: a pretrained model's vector for its meaning."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return encode(list(X))

    def get_feature_names_out(self, input_features=None):
        dim = next(iter(_EMB_CACHE.values())).shape[0] if _EMB_CACHE else 0
        return np.array([f"embedding_{i}" for i in range(dim)])

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class LexiconScore:
    """The WSB lexicon score as a one-column sparse feature (sklearn transformer)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from scipy import sparse
        return sparse.csr_matrix(np.array([[wsb_score(t)] for t in X]))

    def get_feature_names_out(self, input_features=None):
        return np.array(["wsb_lexicon_score"])

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


def make_features():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    return FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=preprocess, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                 token_pattern=r"(?u)\b[\w']+\b")),
        ("char", TfidfVectorizer(preprocessor=preprocess, analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                 sublinear_tf=True, max_features=50_000)),
        ("lex", LexiconScore()),
    ])


def make_model(kind: str = "logreg", C: float = 1.0, seed: int = 0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    if kind in ("embed", "embed+tfidf"):
        from sklearn.pipeline import FeatureUnion
        feats = Embeddings() if kind == "embed" else FeatureUnion([("tfidf", make_features()),
                                                                   ("embed", Embeddings())])
        return Pipeline([("features", feats),
                         ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=5000))])
    if kind == "logreg":
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    elif kind == "lgbm":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=5,
                             class_weight="balanced", colsample_bytree=0.3, random_state=seed, verbose=-1)
    else:
        raise ValueError(kind)
    return Pipeline([("features", make_features()), ("clf", clf)])


def choose_C(texts: pd.Series, y: pd.Series, grid=C_GRID, folds: int = 5, seed: int = 0,
             kind: str = "logreg") -> tuple[float, dict]:
    """Pick C by stratified CV macro-F1 on the training labels only."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    scores = {C: float(cross_val_score(make_model(kind, C), texts, y, cv=cv, scoring="f1_macro").mean())
              for C in grid}
    return max(scores, key=scores.get), scores


def learning_curve(train: pd.DataFrame, test: pd.DataFrame, sizes, C: float, reps: int = 5,
                   kind: str = "logreg") -> pd.DataFrame:
    """Test macro-F1 when training on stratified random subsets of each size."""
    from sklearn.model_selection import train_test_split

    rows = []
    for n in sizes:
        for r in range(reps):
            if n >= len(train):
                sub = train
            else:
                sub, _ = train_test_split(train, train_size=n, stratify=train["label"], random_state=r)
            m = make_model(kind, C, seed=r).fit(sub["text"], sub["label"])
            res = evaluate(test["label"], pd.Series(m.predict(test["text"]), index=test.index))
            rows.append({"n_train": len(sub), "rep": r, "macro_f1": res["macro_f1"], "accuracy": res["accuracy"]})
            if n >= len(train):
                break
    return pd.DataFrame(rows)


# Finance-specific pretrained classifiers (Hugging Face), scored zero-shot.
HF_MODELS = {
    "fintwitbert": "StephanAkkerman/FinTwitBERT-sentiment",          # financial tweets
    "stocktwits-roberta": "zhayunduo/roberta-base-stocktwits-finetuned",  # StockTwits, bullish vs bearish
}


def _map_label(name: str) -> str | None:
    n = name.lower()
    if "bull" in n or "pos" in n:
        return "bullish"
    if "bear" in n or "neg" in n:
        return "bearish"
    if "neut" in n:
        return "neutral"
    return None


def hf_probs(texts, model: str, batch_size: int = 32) -> pd.DataFrame:
    """P(bullish), P(bearish), P(neutral) from a Hugging Face text classifier.

    Label names differ by model (BULLISH, Positive, LABEL_1...), so they are
    mapped by name; a model with only two classes gets p_neutral = NaN."""
    import os
    for k, v in {"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1",
                 "TOKENIZERS_PARALLELISM": "false"}.items():
        os.environ.setdefault(k, v)
    from transformers import pipeline

    clf = pipeline("text-classification", model=model, top_k=None, truncation=True, max_length=128, device="cpu")
    id2label = clf.model.config.id2label
    mapping = {lab: _map_label(lab) for lab in id2label.values()}
    if None in mapping.values():
        raise ValueError(f"cannot map labels {id2label} of {model}; edit _map_label")
    rows = []
    texts = [t if isinstance(t, str) else "" for t in texts]
    for i in range(0, len(texts), batch_size):
        for res in clf(texts[i:i + batch_size]):
            p = {"bullish": 0.0, "bearish": 0.0, "neutral": np.nan if "neutral" not in mapping.values() else 0.0}
            for r in res:
                p[mapping[r["label"]]] += r["score"]
            rows.append(p)
    out = pd.DataFrame(rows).rename(columns=lambda c: f"p_{c}")
    out.attrs["labels"] = mapping
    return out


def probs_to_label(p: pd.DataFrame, band: float = 0.0) -> pd.Series:
    """Most likely label; for two-class models, 'neutral' when P(bullish) is
    within `band` of 0.5 (the model is unsure either way)."""
    if p["p_neutral"].isna().all():
        d = p["p_bullish"] - 0.5
        return pd.Series(np.where(d > band, "bullish", np.where(d < -band, "bearish", "neutral")), index=p.index)
    return p[["p_bullish", "p_bearish", "p_neutral"]].idxmax(axis=1).str.replace("p_", "", regex=False)


def choose_band(p: pd.DataFrame, y: pd.Series, grid=np.arange(0.0, 0.5, 0.025)) -> float:
    """Neutral band for a two-class model, tuned on the TRAINING labels."""
    return float(max(grid, key=lambda b: evaluate(y.reset_index(drop=True),
                                                   probs_to_label(p.reset_index(drop=True), b))["macro_f1"]))


def top_features(model, n: int = 20, kinds=("word", "lex")) -> pd.DataFrame:
    """The words and phrases a trained logistic regression leans on most.

    For each label, the features with the largest positive weight: seeing them
    in a text pushes the prediction toward that label. Character n-grams are
    left out by default because pieces like "endi" are hard to read."""
    names = np.array([str(x) for x in model.named_steps["features"].get_feature_names_out()])
    clf = model.named_steps["clf"]
    keep = np.array([nm.split("__", 1)[0] in kinds for nm in names])
    rows = []
    for k, label in enumerate(clf.classes_):
        w = clf.coef_[k]
        idx = [i for i in np.argsort(-w) if keep[i]][:n]
        for rank, i in enumerate(idx, start=1):
            rows.append({"label": label, "rank": rank, "feature": names[i].split("__", 1)[1],
                         "weight": float(w[i])})
    return pd.DataFrame(rows)


def run(train_path, test_path, out_dir="reports/tables", model_dir="models", curve_reps: int = 5,
        embed: bool = False):
    """Fit every model on the training labels, score all of them on the test labels.

    With embed=True also fits logistic regression on sentence embeddings, alone
    and combined with the TF-IDF features. C is tuned separately per model,
    always by CV on the training labels."""
    train = load_sentiment_labels(train_path)[["id", "text", "label"]].reset_index(drop=True)
    test = load_sentiment_labels(test_path)[["id", "text", "label"]].reset_index(drop=True)
    leak = set(train["id"]) & set(test["id"])
    if leak:
        raise ValueError(f"{len(leak)} test texts are in the training labels")
    if embed:
        encode(list(train["text"]) + list(test["text"]))     # encode once, reused by every fit

    kinds = ["logreg", "lgbm"] + (["embed", "embed+tfidf"] if embed else [])
    Cs, cv_scores, models, preds = {}, {}, {}, {}
    for name, f in SCORERS.items():                       # rule-based baselines, same test set
        preds[name] = test["text"].map(f).map(classify)
    for kind in kinds:
        tune = "logreg" if kind == "lgbm" else kind       # LightGBM has no C; it reuses the TF-IDF features
        if tune not in Cs:
            Cs[tune], cv_scores[tune] = choose_C(train["text"], train["label"], kind=tune)
        models[kind] = make_model(kind, Cs[tune]).fit(train["text"], train["label"])
        preds[kind] = pd.Series(models[kind].predict(test["text"]), index=test.index)
    preds["always neutral"] = pd.Series("neutral", index=test.index)
    results = pd.DataFrame([{"scorer": n, **evaluate(test["label"], p)} for n, p in preds.items()])
    results = results.sort_values("macro_f1", ascending=False)

    # learning curve for the best trained model (picked on CV, not on the test set)
    best = max((k for k in kinds if k != "lgbm"), key=lambda k: max(cv_scores[k].values()))
    sizes = [n for n in (100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000) if n < len(train)] + [len(train)]
    curve = learning_curve(train, test, sizes, Cs[best], reps=curve_reps, kind=best).assign(model=best)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "sentiment_model_eval.csv", index=False)
    curve.to_csv(out / "sentiment_learning_curve.csv", index=False)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(models[best], Path(model_dir) / f"sentiment_{best.replace('+', '_')}.joblib")
    return {"train": train, "test": test, "C": Cs, "cv_scores": cv_scores, "results": results,
            "curve": curve, "preds": preds, "best": best}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="Train on the training labels, grade on the test labels")
    t.add_argument("--train", default="data/labels/sentiment_train.numbers")
    t.add_argument("--test", default="data/labels/sentiment_sample.xlsx")
    t.add_argument("--reps", type=int, default=5)
    t.add_argument("--embed", action="store_true",
                   help="Also try sentence embeddings (pip install sentence-transformers; ~90 MB download once)")
    x = sub.add_parser("explain", help="List the words the trained logistic regression relies on")
    x.add_argument("--model", default="models/sentiment_logreg.joblib")
    x.add_argument("--n", type=int, default=20)
    x.add_argument("--with-char", action="store_true", help="Also show character n-grams")
    h = sub.add_parser("finance-models", help="Score the test labels with finance-specific pretrained models")
    h.add_argument("--models", nargs="+", default=list(HF_MODELS))
    h.add_argument("--train", default="data/labels/sentiment_train.numbers")
    h.add_argument("--test", default="data/labels/sentiment_sample.xlsx")
    e = sub.add_parser("embed", help="Step 1: encode every labeling-file text and save the vectors")
    e.add_argument("--files", nargs="+", default=["data/labels/sentiment_train.numbers",
                                                   "data/labels/sentiment_sample.xlsx"])
    e.add_argument("--embed-model", default=EMBED_MODEL)
    args = p.parse_args(argv)

    if args.cmd == "explain":
        import joblib
        from src.nlp import sentiment_model as sm
        tf = sm.top_features(joblib.load(args.model), args.n,
                             kinds=("word", "lex", "char") if args.with_char else ("word", "lex"))
        out = Path("reports/tables/sentiment_top_features.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        tf.to_csv(out, index=False)
        wide = tf.assign(cell=tf["feature"] + " (" + tf["weight"].round(2).astype(str) + ")").pivot(
            index="rank", columns="label", values="cell")
        print(wide.to_string())
        print(f"\n-> {out}")
        return

    if args.cmd == "finance-models":
        # Runs alone: PyTorch only, no scikit-learn or LightGBM in this process.
        train = load_sentiment_labels(args.train)[["text", "label"]].reset_index(drop=True)
        test = load_sentiment_labels(args.test)[["text", "label"]].reset_index(drop=True)
        rows = []
        for key in args.models:
            name = HF_MODELS.get(key, key)
            print(f"\n{name}")
            pte = hf_probs(test["text"], name)
            print(f"  labels: {pte.attrs['labels']}")
            row = {"scorer": key, "model": name, "band": np.nan}
            if pte["p_neutral"].isna().all():        # two-class: tune a neutral band on the training labels
                ptr = hf_probs(train["text"], name)
                row["band"] = choose_band(ptr, train["label"])
                print(f"  two-class model: neutral when |P(bullish) - 0.5| < {row['band']:.3f} "
                      f"(tuned on {len(train)} training labels)")
            pred = probs_to_label(pte, 0.0 if np.isnan(row["band"]) else row["band"])
            row.update(evaluate(test["label"], pred))
            rows.append(row)
            print(confusion(test["label"], pred).to_string())
        res = pd.DataFrame(rows)
        out = Path("reports/tables/sentiment_finance_models.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out, index=False)
        cols = ["scorer", "macro_f1", "accuracy", "bullish_recall", "bearish_recall", "neutral_recall",
                "bullish_precision", "bearish_precision"]
        print("\nScored on the test labels (compare: LightGBM 0.55, logistic regression 0.51):")
        print(res[cols].round(3).to_string(index=False))
        print(f"\n-> {out}")
        return

    if args.cmd == "embed":
        from src.nlp import sentiment_model as sm
        sm.set_encoder(None, args.embed_model)
        texts = [t for f in args.files for t in read_label_sheet(f)["text"].dropna()]
        n = sm.save_embeddings(texts)
        print(f"saved {n:,} embeddings ({args.embed_model}) -> {EMB_FILE}\n"
              f"next: python -m src.nlp.sentiment_model train --embed")
        return

    # import by module name so the saved model pickles as src.nlp.sentiment_model.*, not __main__.*
    from src.nlp import sentiment_model as sm
    if args.embed:
        n = sm.load_embeddings()
        print(f"loaded {n:,} saved embeddings from {EMB_FILE}")
    r = sm.run(args.train, args.test, curve_reps=args.reps, embed=args.embed)
    tr = r["train"]["label"].value_counts()
    print(f"train: {len(r['train'])} labels {tr.to_dict()}   test: {len(r['test'])} labels")
    for k, sc in r["cv_scores"].items():
        print(f"  {k}: C = {r['C'][k]} chosen by 5-fold CV on train (best CV macro-F1 {max(sc.values()):.3f})")
    cols = ["scorer", "macro_f1", "accuracy", "bullish_recall", "bearish_recall", "neutral_recall",
            "bullish_precision", "bearish_precision"]
    print("\nScored on the test labels:")
    print(r["results"][cols].round(3).to_string(index=False))
    print(f"\nLearning curve, {r['best']} (test macro-F1, mean of repeats):")
    print(r["curve"].groupby("n_train")["macro_f1"].agg(["mean", "std", "count"]).round(3).to_string())
    print(f"\nConfusion matrix, {r['best']} (rows = your labels):")
    print(confusion(r["test"]["label"], r["preds"][r["best"]]).to_string())


if __name__ == "__main__":
    main()
