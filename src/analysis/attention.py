"""H1 and H2: does abnormal WSB attention on day t predict day t+1 volatility
and trading volume, beyond what the stock's own recent history explains?

Model (two-way fixed effects, standard errors clustered by stock and by date):

    y_{i,t+1} = b * abn_attn_{i,t} + controls_{i,t} + stock_i + date_t + e

    - stock effects absorb each stock's normal level (GME is always volatile)
    - date effects absorb market-wide days (a Fed day moves everything), so
      the market return needs no separate control
    - clustering by date handles all stocks moving together on one day;
      by stock handles each stock's persistence over time

H5 preview (reverse direction): does today's price move predict tomorrow's
attention? If attention mostly FOLLOWS moves, H1 must be read with care.

Usage:
    python -m src.analysis.attention --panel data/processed/panel.parquet \
        --out reports/tables/attention_results.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

H1_CONTROLS = ["abs_ret", "rv5", "rv20", "ret", "ret5", "abn_vol"]
H2_CONTROLS = ["abn_vol", "abn_vol5", "abs_ret", "rv5", "ret"]
H5_CONTROLS = ["abn_attn", "abn_vol"]
WINSOR = (0.01, 0.99)


@dataclass
class Spec:
    name: str
    hypothesis: str
    y: str
    x: str
    controls: list[str]
    sample: str = "all"            # all | no_gme_amc | no_squeeze | h1_2021 | h2_2021
    extra: dict = field(default_factory=dict)


def default_specs() -> list[Spec]:
    s = []
    for sample in ["all", "no_gme_amc", "no_squeeze", "h1_2021", "h2_2021"]:
        s.append(Spec(f"H1 volatility [{sample}]", "H1", "abs_ret_next", "abn_attn", H1_CONTROLS, sample))
    s.append(Spec("H1 volatility, market-adjusted", "H1", "abs_xret_next", "abn_attn", H1_CONTROLS))
    s.append(Spec("H1 volatility, unique-author attention", "H1", "abs_ret_next", "abn_authors", H1_CONTROLS))
    s.append(Spec("H1 volatility, z-score attention", "H1", "abs_ret_next", "abn_attn_z", H1_CONTROLS))
    for sample in ["all", "no_gme_amc", "no_squeeze", "h1_2021", "h2_2021"]:
        s.append(Spec(f"H2 volume [{sample}]", "H2", "abn_vol_next", "abn_attn", H2_CONTROLS, sample))
    s.append(Spec("H2 volume, unique-author attention", "H2", "abn_vol_next", "abn_authors", H2_CONTROLS))
    s.append(Spec("H5 preview: |return| today -> attention tomorrow", "H5", "abn_attn_next", "abs_ret", H5_CONTROLS))
    return s


def earnings_specs() -> list[Spec]:
    """H1/H2 again with earnings controls (needs add_earnings_flags columns).

    'covered' = stocks with any earnings date, no earnings control: the same
    stocks as the controlled versions, so the change in the attention effect
    comes from the control, not from a different sample."""
    s = []
    for hyp, y, ctrl in (("H1", "abs_ret_next", H1_CONTROLS), ("H2", "abn_vol_next", H2_CONTROLS)):
        name = "volatility" if hyp == "H1" else "volume"
        s.append(Spec(f"{hyp} {name} [earnings-covered stocks]", hyp, y, "abn_attn", ctrl, "earnings_covered"))
        s.append(Spec(f"{hyp} {name} + earnings-day controls", hyp, y, "abn_attn",
                      ctrl + ["earn_today", "earn_next"], "earnings_covered"))
        s.append(Spec(f"{hyp} {name} [no earnings within 2 days]", hyp, y, "abn_attn", ctrl, "no_earnings"))
    return s


def select_sample(panel: pd.DataFrame, sample: str) -> pd.DataFrame:
    if sample == "all":
        return panel
    if sample == "earnings_covered":
        return panel[panel["earn_today"].notna()]
    if sample == "no_earnings":
        return panel[panel["earn_window"] == 0]
    if sample == "no_gme_amc":
        return panel[~panel["ticker"].isin(["GME", "AMC"])]
    if sample == "no_squeeze":   # drop the Jan-Feb 2021 squeeze months
        return panel[panel["date"] >= "2021-03-01"]
    if sample == "h1_2021":
        return panel[panel["date"] < "2021-07-01"]
    if sample == "h2_2021":
        return panel[panel["date"] >= "2021-07-01"]
    raise ValueError(sample)


def winsorize(s: pd.Series, lo: float = WINSOR[0], hi: float = WINSOR[1]) -> pd.Series:
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)


def fit(panel: pd.DataFrame, spec: Spec) -> dict:
    """Fit one spec with linearmodels PanelOLS; return a result row."""
    from linearmodels.panel import PanelOLS

    cols = list(dict.fromkeys([spec.y, spec.x, *spec.controls]))
    d = select_sample(panel, spec.sample)[["ticker", "date", *cols]].replace([np.inf, -np.inf], np.nan).dropna()
    d = d.copy()
    for c in cols:
        if d[c].nunique() > 2:            # leave 0/1 flags (earnings days) as they are
            d[c] = winsorize(d[c])
    d = d.set_index(["ticker", "date"])
    exog = d[list(dict.fromkeys([spec.x, *spec.controls]))]
    res = PanelOLS(d[spec.y], exog, entity_effects=True, time_effects=True,
                   drop_absorbed=True, check_rank=False).fit(
        cov_type="clustered", cluster_entity=True, cluster_time=True)
    b, se = res.params[spec.x], res.std_errors[spec.x]
    sd_x = d[spec.x].std()
    return {
        "spec": spec.name, "hypothesis": spec.hypothesis, "y": spec.y, "x": spec.x, "sample": spec.sample,
        "coef": b, "se": se, "t": res.tstats[spec.x], "p": res.pvalues[spec.x],
        "sd_x": sd_x, "effect_1sd": b * sd_x, "y_mean": d[spec.y].mean(),
        "effect_1sd_pct_of_mean": b * sd_x / d[spec.y].mean() if spec.y.startswith("abs") else np.nan,
        "n_obs": int(res.nobs), "n_tickers": int(d.index.get_level_values(0).nunique()),
        "n_days": int(d.index.get_level_values(1).nunique()), "r2_within": res.rsquared_within,
    }


def run(panel: pd.DataFrame, specs: list[Spec] | None = None) -> pd.DataFrame:
    return pd.DataFrame([fit(panel, s) for s in (specs or default_specs())])


def summarize(results: pd.DataFrame) -> str:
    lines = []
    for r in results.itertuples():
        star = "***" if r.p < 0.01 else "**" if r.p < 0.05 else "*" if r.p < 0.1 else ""
        if r.y.startswith("abs"):
            eff = (f"+1 SD attention -> {r.effect_1sd * 1e4:+.0f} bp |return| "
                   f"({r.effect_1sd_pct_of_mean:+.1%} of avg)")
        elif r.y == "abn_vol_next":
            eff = f"+1 SD attention -> {np.expm1(r.effect_1sd):+.1%} volume"
        else:
            eff = f"+1 SD {r.x} -> {r.effect_1sd:+.3f} next-day abn_attn"
        lines.append(f"{r.spec:<52} b={r.coef:+.4f} (t={r.t:5.1f}){star:<3} {eff}  "
                     f"[N={r.n_obs:,}, {r.n_tickers} stocks, {r.n_days} days]")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", default="data/processed/panel.parquet")
    p.add_argument("--out", default="reports/tables/attention_results.csv")
    p.add_argument("--earnings", default="data/raw/earnings_dates.csv",
                   help="Earnings dates from src.data.earnings; adds controlled specs when the file exists")
    args = p.parse_args(argv)
    panel = pd.read_parquet(args.panel)
    specs = default_specs()
    if Path(args.earnings).exists():
        from src.data.earnings import add_earnings_flags, reaction_days
        days = reaction_days(pd.read_csv(args.earnings))
        panel = add_earnings_flags(panel, days)
        specs += earnings_specs()
        print(f"earnings: {len(days):,} reaction days for {days['ticker'].nunique()} stocks")
    results = run(panel, specs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    print(summarize(results))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
