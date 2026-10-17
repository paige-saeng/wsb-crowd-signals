"""Timestamp alignment tests. If any of these fail, every result downstream
is suspect, so run them before trusting an analysis."""

import pandas as pd
import pytest

from src.data.calendar import assign_trading_day, trading_sessions

ET = "America/New_York"


@pytest.fixture(scope="module")
def sessions():
    return trading_sessions("2020-12-01", "2022-01-31")


def day_for(et_string, sessions):
    """Trading day for a wall-clock time in New York."""
    ts = pd.Timestamp(et_string, tz=ET).tz_convert("UTC")
    out = assign_trading_day([ts], sessions)
    return out["trading_day"].iloc[0].date().isoformat()


@pytest.mark.parametrize(
    "posted_et, expected",
    [
        # Normal Wednesday (GME squeeze week)
        ("2021-01-27 09:00", "2021-01-27"),  # pre-market -> same day
        ("2021-01-27 15:59", "2021-01-27"),  # just before close
        ("2021-01-27 16:00", "2021-01-28"),  # exactly at close -> next day
        ("2021-01-27 16:01", "2021-01-28"),  # after close
        ("2021-01-27 23:59", "2021-01-28"),
        # Weekend rolls to Monday
        ("2021-01-29 17:00", "2021-02-01"),  # Friday evening
        ("2021-01-30 12:00", "2021-02-01"),  # Saturday
        ("2021-01-31 22:00", "2021-02-01"),  # Sunday night
        # Holiday: MLK Day, Mon 2021-01-18, market closed
        ("2021-01-15 18:00", "2021-01-19"),
        ("2021-01-18 12:00", "2021-01-19"),
        # Early close: day after Thanksgiving, close at 1:00 pm ET
        ("2021-11-26 12:59", "2021-11-26"),
        ("2021-11-26 13:30", "2021-11-29"),
    ],
)
def test_assignment_rules(posted_et, expected, sessions):
    assert day_for(posted_et, sessions) == expected


def test_daylight_saving_uses_real_close(sessions):
    """20:30 UTC is 3:30 pm EST in winter (before close) but 4:30 pm EDT in
    summer (after close). A hard-coded 21:00 UTC close would get one wrong."""
    winter = assign_trading_day([pd.Timestamp("2021-03-12 20:30", tz="UTC")], sessions)
    summer = assign_trading_day([pd.Timestamp("2021-03-15 20:30", tz="UTC")], sessions)
    assert winter["trading_day"].iloc[0] == pd.Timestamp("2021-03-12")
    assert summer["trading_day"].iloc[0] == pd.Timestamp("2021-03-16")


def test_accepts_unix_seconds_like_reddit(sessions):
    # 2021-01-27 15:00 ET = 20:00 UTC
    epoch = int(pd.Timestamp("2021-01-27 20:00", tz="UTC").timestamp())
    out = assign_trading_day([epoch], sessions)
    assert out["trading_day"].iloc[0] == pd.Timestamp("2021-01-27")


def test_market_hours_flag(sessions):
    times = [
        pd.Timestamp("2021-01-27 09:00", tz=ET),  # pre-market
        pd.Timestamp("2021-01-27 10:00", tz=ET),  # open
        pd.Timestamp("2021-01-27 17:00", tz=ET),  # after hours
    ]
    out = assign_trading_day(times, sessions)
    assert out["during_market_hours"].tolist() == [False, True, False]


def test_preserves_input_order(sessions):
    times = [
        pd.Timestamp("2021-02-01 10:00", tz=ET),
        pd.Timestamp("2021-01-27 10:00", tz=ET),
    ]
    out = assign_trading_day(times, sessions)
    assert out["trading_day"].dt.date.astype(str).tolist() == ["2021-02-01", "2021-01-27"]


def test_beyond_calendar_is_nat(sessions):
    out = assign_trading_day([pd.Timestamp("2030-01-01", tz="UTC")], sessions)
    assert pd.isna(out["trading_day"].iloc[0])


def test_auto_sessions():
    out = assign_trading_day([pd.Timestamp("2021-01-30 12:00", tz=ET)])
    assert out["trading_day"].iloc[0] == pd.Timestamp("2021-02-01")
