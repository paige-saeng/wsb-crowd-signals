"""Map Reddit timestamps to the trading day they can first inform.

The rule (this is what prevents lookahead bias):

    A post belongs to the first trading session whose CLOSE is strictly
    after the post's timestamp.

So with a normal 4:00 pm ET close:
    - Monday 3:59 pm ET   -> Monday     (posted before Monday's close)
    - Monday 4:00 pm ET   -> Tuesday    (a trade at the close can't use it)
    - Friday 6:00 pm ET   -> Monday
    - Saturday            -> Monday
    - before a holiday    -> the next open day
    - early-close days (e.g. 1:00 pm the day after Thanksgiving) use the real close

Features for trading day d therefore use only posts made between the close
of day d-1 and the close of day d. Targets are measured on day d+1 onward.

All comparisons happen in UTC. Reddit stores `created_utc` as Unix seconds,
and the NYSE calendar handles daylight saving time, so 4:00 pm ET is
20:00 UTC in summer and 21:00 UTC in winter. Hard-coding either one is a
classic bug; tests/test_calendar.py checks both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

ET = "America/New_York"


def to_utc(timestamps) -> pd.DatetimeIndex:
    """Convert Unix seconds, strings, or datetimes to a tz-aware UTC index.

    Naive datetimes are assumed to already be UTC (Reddit's convention).
    """
    s = pd.Series(timestamps)
    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_datetime(s, unit="s", utc=True)
    else:
        out = pd.to_datetime(s, utc=True)
    return pd.DatetimeIndex(out)


def trading_sessions(start, end, exchange: str = "NYSE") -> pd.DataFrame:
    """Return one row per trading day with UTC market_open and market_close.

    Index is the trading day (a date). Weekends and exchange holidays are
    absent; early closes have their real close time.
    """
    cal = mcal.get_calendar(exchange)
    sched = cal.schedule(start_date=start, end_date=end)
    sched = sched[["market_open", "market_close"]].copy()
    for col in sched.columns:
        sched[col] = pd.to_datetime(sched[col], utc=True)
    sched.index = pd.DatetimeIndex(sched.index).normalize()
    sched.index.name = "trading_day"
    return sched


def _as_ns(values) -> np.ndarray:
    idx = pd.DatetimeIndex(values)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.to_numpy(dtype="datetime64[ns]")


def assign_trading_day(timestamps, sessions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Map each timestamp to the trading day it can first inform.

    Parameters
    ----------
    timestamps : array-like of Unix seconds, strings, or datetimes (UTC).
    sessions : output of `trading_sessions`. If None, built automatically to
        cover the timestamps' range.

    Returns
    -------
    DataFrame aligned with the input order, with columns:
        created_utc  : tz-aware UTC timestamp
        trading_day  : the session date (NaT if past the calendar's range)
        during_market_hours : True if posted while that session was open
    """
    ts = to_utc(timestamps)
    if sessions is None:
        if len(ts) == 0:
            return pd.DataFrame(columns=["created_utc", "trading_day", "during_market_hours"])
        start = (ts.min() - pd.Timedelta(days=7)).date()
        end = (ts.max() + pd.Timedelta(days=14)).date()
        sessions = trading_sessions(start, end)

    closes = _as_ns(sessions["market_close"])
    opens = _as_ns(sessions["market_open"])
    ts_ns = _as_ns(ts)

    # side="right": a timestamp exactly at the close goes to the NEXT session.
    pos = np.searchsorted(closes, ts_ns, side="right")
    in_range = pos < len(closes)
    safe_pos = np.where(in_range, pos, 0)

    days = sessions.index.to_numpy(dtype="datetime64[ns]")[safe_pos]
    trading_day = pd.DatetimeIndex(np.where(in_range, days, np.datetime64("NaT", "ns")))
    during = in_range & (ts_ns >= opens[safe_pos]) & (ts_ns < closes[safe_pos])

    return pd.DataFrame(
        {
            "created_utc": ts,
            "trading_day": trading_day,
            "during_market_hours": during,
        }
    )
