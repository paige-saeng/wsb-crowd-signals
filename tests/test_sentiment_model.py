import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("lightgbm")

from src.nlp.sentiment_model import choose_C, learning_curve, make_model, run  # noqa: E402


def _data(n, seed):
    import random
    rng = random.Random(seed)
    bull = ["loading calls on {t}", "{t} to the moon", "bought more {t}, holding", "{t} undervalued, buying"]
    bear = ["buying {t} puts", "{t} is overvalued, selling", "{t} going to zero", "shorting {t} tomorrow"]
    neu = ["what happened to {t} today", "{t} earnings are thursday", "anyone watching {t}?", "lol {t} memes"]
    rows = []
    for i in range(n):
        lab = rng.choice(["bullish"] * 3 + ["bearish"] * 2 + ["neutral"] * 5)
        tpl = rng.choice({"bullish": bull, "bearish": bear, "neutral": neu}[lab])
        rows.append({"id": f"{seed}-{i}", "text": tpl.format(t=rng.choice(["GME", "AMC", "PLTR", "BB"])),
                     "label": lab})
    return pd.DataFrame(rows)


def test_model_learns_planted_signal():
    tr, te = _data(300, 1), _data(120, 2)
    m = make_model("logreg", 1.0).fit(tr["text"], tr["label"])
    assert (m.predict(te["text"]) == te["label"]).mean() > 0.9


def test_choose_C_uses_training_only():
    tr = _data(150, 3)
    C, scores = choose_C(tr["text"], tr["label"], grid=(0.3, 3.0), folds=3)
    assert C in scores and set(scores) == {0.3, 3.0}


def test_learning_curve_sizes():
    tr, te = _data(200, 4), _data(80, 5)
    c = learning_curve(tr, te, [50, 200], C=1.0, reps=2)
    assert sorted(c["n_train"].unique()) == [50, 200]
    assert (c[c.n_train == 200]["rep"] == 0).all()        # full set is fit once


def test_run_rejects_leak(tmp_path):
    tr, te = _data(100, 6), _data(50, 7)
    te.loc[0, "id"] = tr.loc[0, "id"]
    tr.to_csv(tmp_path / "tr.csv", index=False)
    te.to_csv(tmp_path / "te.csv", index=False)
    with pytest.raises(ValueError, match="test texts"):
        run(tmp_path / "tr.csv", tmp_path / "te.csv", out_dir=tmp_path, model_dir=tmp_path)


def test_embedding_model_with_fake_encoder(tmp_path):
    import numpy as np
    from src.nlp import sentiment_model as sm

    def fake(texts):     # a stand-in "meaning" vector: which word family the text uses
        return np.array([[("call" in t or "moon" in t or "bought" in t or "undervalued" in t) * 1.0,
                          ("put" in t or "overvalued" in t or "zero" in t or "short" in t) * 1.0,
                          len(t) / 50] for t in texts])

    sm.set_encoder(fake, name="fake")
    try:
        tr, te = _data(200, 8), _data(80, 9)
        tr.to_csv(tmp_path / "tr.csv", index=False)
        te.to_csv(tmp_path / "te.csv", index=False)
        r = sm.run(tmp_path / "tr.csv", tmp_path / "te.csv", out_dir=tmp_path, model_dir=tmp_path,
                   curve_reps=1, embed=True)
        assert {"embed", "embed+tfidf"} <= set(r["results"]["scorer"])
        emb = r["results"].set_index("scorer").loc["embed", "macro_f1"]
        assert emb > 0.8
        assert r["best"] in {"logreg", "embed", "embed+tfidf"}
    finally:
        sm._EMB_CACHE.clear()
        sm.set_encoder(None)


def test_saved_embeddings_train_without_encoder(tmp_path):
    import numpy as np
    from src.nlp import sentiment_model as sm

    def fake(texts):
        return np.array([[("call" in t or "moon" in t) * 1.0, ("put" in t or "zero" in t) * 1.0] for t in texts])

    tr, te = _data(150, 10), _data(60, 11)
    f = tmp_path / "emb.npz"
    sm._EMB_CACHE.clear()
    sm.set_encoder(fake, name="fake-2d")
    try:
        n = sm.save_embeddings(list(tr["text"]) + list(te["text"]), f)
        assert n == len(set(tr["text"]) | set(te["text"]))
        sm._EMB_CACHE.clear()
        sm.set_encoder(None)                      # no encoder: must rely on the file
        assert sm.load_embeddings(f) == n
        assert sm.encode(list(te["text"][:3])).shape == (3, 2)
        with pytest.raises(RuntimeError, match="embed"):
            sm.encode(["a text that was never embedded"])
    finally:
        sm._EMB_CACHE.clear()
        sm.set_encoder(None)


def test_top_features_finds_planted_words():
    from src.nlp.sentiment_model import top_features
    tr = _data(300, 12)
    m = make_model("logreg", 1.0).fit(tr["text"], tr["label"])
    tf = top_features(m, n=10)
    assert set(tf["label"]) == {"bullish", "bearish", "neutral"}
    bear = " ".join(tf[tf.label == "bearish"]["feature"])
    bull = " ".join(tf[tf.label == "bullish"]["feature"])
    assert "puts" in bear or "overvalued" in bear or "zero" in bear
    assert "calls" in bull or "moon" in bull or "bought" in bull or "undervalued" in bull
