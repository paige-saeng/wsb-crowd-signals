"""H5: who leads whom? Does WSB attention follow price moves, or do price
moves follow WSB attention?

Two directions, each with the past 5 trading days of both variables
(a panel version of a Granger causality test):

    attention leads price:   |ret|_t  = sum_k c_k |ret|_{t-k} + sum_k d_k attn_{t-k} + stock + date
    price leads attention:   attn_t   = sum_k a_k attn_{t-k}  + sum_k b_k X_{t-k}    + stock + date
                             with X = |ret| (any big move) and, separately, signed ret
                             (do rises draw more attention than falls?)

"X leads Y" = the lags of X jointly help predict Y beyond Y's own past
(a Wald test that all five lag coefficients are zero, using standard errors
clustered by stock and date). Only lags enter, never same-day values, so
neither direction can borrow from the future.

Coefficients are standardized: the effect of a 1 SD change in the leading
variable k days ago, in SDs of the outcome, so the two directions are
comparable.

Usage
    python -m src.analysis.leadlag
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.attention import winsorize

LAGS = 5


def add_lags(panel: pd.DataFrame, cols, lags: int = LAGS) -> pd.DataFrame:
    df = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = df.groupby("ticker", group_keys=False)
    for c in cols:
        for k in range(1, lags + 1):
            df[f"{c}_l{k}"] = g[c].shift(k)
    return df


def granger(df: pd.DataFrame, y: str, cause: str, lags: int = LAGS, extra: tuple[str, ...] = ()) -> dict:
    """Does `cause` (lags 1..L) predict `y` beyond y's own lags 1..L?

    Two-way fixed effects, errors clustered by stock and date, variables
    winsorized at 1%/99% (as in H1-H4)."""
    from linearmodels.panel import PanelOLS

    own = [f"{y}_l{k}" for k in range(1, lags + 1)]
    cz = [f"{cause}_l{k}" for k in range(1, lags + 1)]
    ex = [f"{c}_l{k}" for c in extra for k in range(1, lags + 1)]
    cols = list(dict.fromkeys([y, *own, *cz, *ex]))
    d = df[["ticker", "date", *cols]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    for c in cols:
        d[c] = winsorize(d[c])
    d = d.set_index(["ticker", "date"])
    res = PanelOLS(d[y], d[[*own, *cz, *ex]], entity_effects=True, time_effects=True,
                   drop_absorbed=True, check_rank=False).fit(
        cov_type="clustered", cluster_entity=True, cluster_time=True)
    b = res.params[cz].to_numpy()
    V = res.cov.loc[cz, cz].to_numpy()
    wald = float(b @ np.linalg.solve(V, b))
    scale = d[cz[0]].std() / d[y].std()
    out = {"outcome": y, "leader": cause, "wald_chi2": wald, "df": len(cz),
           "p_joint": float(stats.chi2.sf(wald, len(cz))),
           "sum_std": float(b.sum() * scale), "n_obs": int(res.nobs)}
    for k, c in enumerate(cz, start=1):
        out[f"lag{k}_std"] = float(res.params[c] * scale)
        out[f"lag{k}_t"] = float(res.tstats[c])
    return out


TESTS = [
    # (label, outcome, leader, extra controls lagged too)
    ("attention -> next |return|", "abs_ret", "abn_attn", ()),
    ("|return| -> next attention", "abn_attn", "abs_ret", ()),
    ("signed return -> next attention", "abn_attn", "ret", ("abs_ret",)),
    ("attention -> next signed return", "ret", "abn_attn", ()),
]


def run(panel: pd.DataFrame, lags: int = LAGS) -> pd.DataFrame:
    df = add_lags(panel, ["abs_ret", "ret", "abn_attn"], lags)
    rows = []
    for label, y, x, extra in TESTS:
        rows.append({"test": label, **granger(df, y, x, lags, extra)})
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame, lags: int = LAGS) -> str:
    lines = [f"{'test':<34} {'joint p':>8}  " + "  ".join(f"lag{k:>2}" for k in range(1, lags + 1)) + "   (std effect, t)"]
    for r in res.itertuples():
        star = "***" if r.p_joint < 0.01 else "**" if r.p_joint < 0.05 else "*" if r.p_joint < 0.1 else ""
        cells = "  ".join(f"{getattr(r, f'lag{k}_std'):+.3f}({getattr(r, f'lag{k}_t'):+.1f})" for k in range(1, lags + 1))
        lines.append(f"{r.test:<34} {r.p_joint:8.4f}{star:<3} {cells}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", default="data/processed/panel.parquet")
    p.add_argument("--lags", type=int, default=LAGS)
    p.add_argument("--out", default="reports/tables/leadlag_results.csv")
    args = p.parse_args(argv)
    res = run(pd.read_parquet(args.panel), args.lags)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(summarize(res, args.lags))
    print("\nstd effect = change in the outcome (in SDs) per 1 SD of the leader k trading days earlier")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
