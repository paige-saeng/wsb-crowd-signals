"""H4 event study: what happens to a stock's price around a WSB hype spike?

Events
    A stock-day where abnormal attention (z-score) >= 3 and mentions >= 10.
    Spikes on the same stock within 20 trading days of its previous spike are
    merged into the same episode (only the first counts as the event), so one
    long hype episode is one event, not twenty.

Timing
    Day 0 = the trading day of the spike (posts made before that day's close).
    Days -10..-1 show what happened BEFORE the hype (did price lead attention?).
    Day 0 mixes cause and effect: posts and price moves on the same day.
    Days +1..+20 are what a trader could act on after seeing the spike at the
    close of day 0. This is the tradable part, and where a reversal would show.

Abnormal returns
    Main: beta-only model, AR = r - beta * r_market, with beta estimated per
    event on days -250..-30 (min 100 days) and alpha set to 0.
    Why not the full market model (alpha + beta)? WSB stocks rallied hard in
    late 2020, so their estimated alpha averages +0.26% per day. Subtracting
    that "normal" excess return on every day after the event manufactures a
    fake decline of about 5% over 20 days. Setting alpha to 0 is the standard
    fix when the estimation window itself contains abnormal performance.
    Robustness: full market model (reported, but biased for this sample) and
    market-adjusted returns (stock minus SPY, beta = 1).

Inference
    For each window, the CAR of every event is regressed on a constant with
    standard errors clustered by event date: many stocks spike on the same
    day (late January 2021), so events are not independent. Means are
    winsorized at 1% and 99% because a few meme-stock CARs exceed +100%;
    medians and the share of positive CARs are reported alongside.

Usage
    python -m src.analysis.event_study --panel data/processed/panel.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.finance.returns import abnormal_returns, market_model

PRE, POST = 10, 20
EST_START, EST_END = -250, -30
WINDOWS = {"pre (-10,-1)": (-10, -1), "day 0": (0, 0), "next day (+1)": (1, 1),
           "week (+1,+5)": (1, 5), "month (+1,+20)": (1, 20)}
WINSOR = (0.01, 0.99)


# ---------------------------------------------------------------- events

def find_events(panel: pd.DataFrame, z: float = 3.0, min_mentions: int = 10, gap: int = 20,
                attn_col: str = "abn_attn_z") -> pd.DataFrame:
    """One row per event: ticker, date, and the spike's attention stats."""
    rows = []
    for t, g in panel.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        hit = (g[attn_col] >= z) & (g["mentions"] >= min_mentions)
        last = -10**9
        for k in np.flatnonzero(hit.to_numpy()):
            if k - last > gap:
                rows.append({"ticker": t, "date": g.at[k, "date"], "mentions": g.at[k, "mentions"],
                             attn_col: g.at[k, attn_col]})
                last = k
            # spikes within `gap` days of an event extend it; they are not new events
            else:
                last = k
    return pd.DataFrame(rows, columns=["ticker", "date", "mentions", attn_col])


# ---------------------------------------------------------------- abnormal returns

def event_returns(panel: pd.DataFrame, events: pd.DataFrame, pre: int = PRE, post: int = POST,
                  est: tuple[int, int] = (EST_START, EST_END), min_est: int = 100) -> pd.DataFrame:
    """Long table: one row per (event, relative day) with AR (market model)
    and XR (market-adjusted). Events without a full window or enough
    estimation data are dropped."""
    out = []
    by_ticker = {t: g.sort_values("date").reset_index(drop=True) for t, g in panel.groupby("ticker")}
    for ev_id, e in enumerate(events.itertuples(index=False)):
        g = by_ticker.get(e.ticker)
        if g is None:
            continue
        pos = np.flatnonzero(g["date"].to_numpy() == np.datetime64(e.date))
        if not len(pos):
            continue
        k = int(pos[0])
        lo, hi = k - pre, k + post
        e0, e1 = k + est[0], k + est[1]
        if lo < 0 or hi >= len(g) or e1 < 0:
            continue
        r, m = g["ret"].to_numpy(float), g["mkt_ret"].to_numpy(float)
        fit = market_model(r[max(e0, 0):e1 + 1], m[max(e0, 0):e1 + 1], min_obs=min_est)
        if fit is None:
            continue
        a, b = fit
        win_r, win_m = r[lo:hi + 1], m[lo:hi + 1]
        if not (np.isfinite(win_r).all() and np.isfinite(win_m).all()):
            continue
        out.append(pd.DataFrame({
            "event_id": ev_id, "ticker": e.ticker, "event_date": e.date,
            "rel_day": np.arange(-pre, post + 1),
            "ar": abnormal_returns(win_r, win_m, 0.0, b),       # main: beta only
            "ar_mm": abnormal_returns(win_r, win_m, a, b),      # full market model
            "xr": win_r - win_m,                                # market-adjusted
            "alpha": a, "beta": b,
        }))
    if not out:
        return pd.DataFrame(columns=["event_id", "ticker", "event_date", "rel_day", "ar", "ar_mm",
                                     "xr", "alpha", "beta"])
    return pd.concat(out, ignore_index=True)


def cars(er: pd.DataFrame, windows: dict = WINDOWS, col: str = "ar") -> pd.DataFrame:
    """Cumulative abnormal return per event for each window (simple sum of ARs)."""
    res = {}
    for name, (a, b) in windows.items():
        res[name] = er[er["rel_day"].between(a, b)].groupby("event_id")[col].sum()
    meta = er.groupby("event_id")[["ticker", "event_date"]].first()
    return meta.join(pd.DataFrame(res))


# ---------------------------------------------------------------- inference

def _winsor(s: pd.Series) -> pd.Series:
    lo, hi = s.quantile(list(WINSOR))
    return s.clip(lo, hi)


def mean_test(values: pd.Series, clusters: pd.Series) -> dict:
    """Winsorized mean with a t-stat from standard errors clustered by `clusters`."""
    import statsmodels.api as sm

    v = _winsor(values.astype(float))
    ok = v.notna()
    v, c = v[ok], clusters[ok]
    if len(v) < 3:
        return {"mean": np.nan, "se": np.nan, "t": np.nan, "p": np.nan}
    res = sm.OLS(v.to_numpy(), np.ones(len(v))).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(c)[0]})
    return {"mean": float(res.params[0]), "se": float(res.bse[0]),
            "t": float(res.tvalues[0]), "p": float(res.pvalues[0])}


def summarize_cars(car: pd.DataFrame, group: str, windows: dict = WINDOWS) -> pd.DataFrame:
    rows = []
    for name in windows:
        s = car[name]
        t = mean_test(s, car["event_date"])
        rows.append({"group": group, "window": name, "n_events": int(s.notna().sum()),
                     "n_dates": int(car["event_date"].nunique()), **t,
                     "median": float(s.median()), "share_positive": float((s > 0).mean())})
    return pd.DataFrame(rows)


def caar_path(er: pd.DataFrame, group: str, col: str = "ar", start: int = -PRE) -> pd.DataFrame:
    """Average cumulative abnormal return from `start` to each day, with a 95%
    interval (clustered by event date) at each day."""
    rows = []
    days = sorted(er["rel_day"].unique())
    for d in days:
        if d < start:
            continue
        c = er[er["rel_day"].between(start, d)].groupby("event_id")[col].sum()
        dates = er.groupby("event_id")["event_date"].first().reindex(c.index)
        t = mean_test(c, dates)
        rows.append({"group": group, "rel_day": int(d), "caar": t["mean"],
                     "lo": t["mean"] - 1.96 * t["se"], "hi": t["mean"] + 1.96 * t["se"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- groups

def groups(car: pd.DataFrame) -> dict[str, pd.Index]:
    """Event subsets, as index labels of `car`."""
    d0 = car["day 0"]
    return {
        "all events": car.index,
        "price up on day 0": car.index[d0 > 0],
        "price down on day 0": car.index[d0 <= 0],
        "without GME and AMC": car.index[~car["ticker"].isin(["GME", "AMC"])],
        "without Jan-Feb squeeze": car.index[car["event_date"] >= "2021-03-01"],
    }


def run(panel: pd.DataFrame, z: float = 3.0, min_mentions: int = 10, gap: int = 20):
    events = find_events(panel, z=z, min_mentions=min_mentions, gap=gap)
    er = event_returns(panel, events)
    car = cars(er)
    tables, paths = [], []
    for name, idx in groups(car).items():
        tables.append(summarize_cars(car.loc[idx], name))
        paths.append(caar_path(er[er["event_id"].isin(idx)], name))
    tables.append(summarize_cars(cars(er, col="xr"), "all events, market-adjusted"))
    tables.append(summarize_cars(cars(er, col="ar_mm"), "all events, full market model (alpha-biased)"))
    return events, er, pd.concat(tables, ignore_index=True), pd.concat(paths, ignore_index=True)


def format_summary(table: pd.DataFrame) -> str:
    lines = []
    for g, t in table.groupby("group", sort=False):
        n = int(t["n_events"].max())
        lines.append(f"\n{g}  ({n:,} events on {int(t['n_dates'].max())} dates)")
        for r in t.itertuples():
            star = "***" if r.p < 0.01 else "**" if r.p < 0.05 else "*" if r.p < 0.1 else ""
            lines.append(f"  {r.window:<16} mean {r.mean:+7.2%} (t={r.t:5.1f}){star:<3} "
                         f"median {r.median:+7.2%}  positive {r.share_positive:5.1%}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", default="data/processed/panel.parquet")
    p.add_argument("--z", type=float, default=3.0, help="Attention z-score threshold for an event")
    p.add_argument("--min-mentions", type=int, default=10)
    p.add_argument("--gap", type=int, default=20, help="Trading days before a stock can have a new event")
    p.add_argument("--out-dir", default="reports/tables")
    args = p.parse_args(argv)

    panel = pd.read_parquet(args.panel)
    events, er, table, path = run(panel, args.z, args.min_mentions, args.gap)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "event_study_car.csv", index=False)
    path.to_csv(out / "event_study_path.csv", index=False)
    er.to_parquet(Path("data/processed") / "event_returns.parquet", index=False)
    ev = er.groupby("event_id")[["alpha", "beta"]].first()
    print(f"{len(events):,} spikes found; {len(ev):,} with a full window and enough history; "
          f"median beta {ev['beta'].median():.2f}; mean estimation-window alpha {ev['alpha'].mean():+.3%}/day "
          f"(= {20 * ev['alpha'].mean():+.1%} over 20 days, why the main model sets alpha to 0)")
    print(format_summary(table))
    print(f"\n-> {out / 'event_study_car.csv'}, {out / 'event_study_path.csv'}")


if __name__ == "__main__":
    main()
