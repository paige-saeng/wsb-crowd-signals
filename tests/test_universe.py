"""Universe builder tests on a tiny local git repo (no network)."""
import subprocess

import pandas as pd

from src.data.universe import build, load_universe_csv


def commit(repo, date, nasdaq, nyse):
    (repo / "nasdaq").mkdir(exist_ok=True)
    (repo / "nyse").mkdir(exist_ok=True)
    (repo / "nasdaq" / "nasdaq_tickers.txt").write_text("\n".join(nasdaq))
    (repo / "nyse" / "nyse_tickers.txt").write_text("\n".join(nyse))
    env = {"GIT_AUTHOR_DATE": f"{date}T12:00:00", "GIT_COMMITTER_DATE": f"{date}T12:00:00",
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin:/usr/local/bin"}
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", date], check=True, env=env)


def test_union_keeps_delisted_and_dates(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    commit(repo, "2020-12-15", ["OLD"], ["X"])            # before window
    commit(repo, "2021-02-01", ["GME", "WISH"], ["BB"])
    commit(repo, "2021-06-01", ["GME", "WISH"], ["BB", "DWAC"])
    commit(repo, "2021-12-01", ["GME", "LOGC"], ["BB"])    # WISH renamed, DWAC gone
    commit(repo, "2023-01-01", ["NEW"], ["BB"])            # after window
    df = build(repo, "2021-01-01", "2022-01-31").set_index("ticker")
    assert set(df.index) == {"GME", "WISH", "BB", "DWAC", "LOGC"}
    assert df.loc["WISH", "first_seen"] == "2021-02-01" and df.loc["WISH", "last_seen"] == "2021-06-01"
    assert df.loc["DWAC", "first_seen"] == "2021-06-01"
    out = tmp_path / "u.csv"
    df.reset_index().to_csv(out, index=False)
    assert load_universe_csv(out) == {"GME", "WISH", "BB", "DWAC", "LOGC"}
