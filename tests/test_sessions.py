import pandas as pd

from basis_bt.sessions import BARS_PER_DAY, TradingClock, bar_label, in_session, minute_grid


def _labels(*stamps):
    return list(bar_label(pd.Series(pd.to_datetime(stamps))).dt.strftime("%H:%M"))


def test_bar_labels_are_end_stamped_and_closing_snapshots_fold_back():
    assert _labels("2025-03-03 09:30:00", "2025-03-03 10:00:59", "2025-03-03 10:01:00") == ["09:31", "10:01", "10:02"]
    assert _labels("2025-03-03 11:30:00", "2025-03-03 11:30:02", "2025-03-03 15:00:03") == ["11:30", "11:30", "15:00"]


def test_session_filter_keeps_closing_grace_and_drops_auction_and_junk():
    ts = pd.Series(pd.to_datetime(["2025-03-03 09:29:00", "2025-03-03 09:30:00", "2025-03-03 11:30:03",
                                   "2025-03-03 11:31:00", "2025-03-03 15:00:02", "2025-03-03 15:23:52"]))
    assert list(in_session(ts)) == [False, True, True, False, True, False]


def test_minute_grid_has_240_bars_per_day():
    grid = minute_grid(pd.to_datetime(["2025-03-03 10:00", "2025-03-04 14:00", "2025-03-04 09:45"]))
    assert len(grid) == 2 * BARS_PER_DAY
    assert grid[0].strftime("%H:%M") == "09:31" and grid[119].strftime("%H:%M") == "11:30"
    assert grid[120].strftime("%H:%M") == "13:01" and grid[239].strftime("%H:%M") == "15:00"


def test_trading_clock_skips_lunch_and_overnight():
    clock = TradingClock(minute_grid(pd.to_datetime(["2025-03-03", "2025-03-04"])))
    assert clock.seconds_between(pd.Timestamp("2025-03-03 11:29:30"), pd.Timestamp("2025-03-03 13:00:30")) == 60
    assert clock.seconds_between(pd.Timestamp("2025-03-03 14:59:00"), pd.Timestamp("2025-03-04 09:31:00")) == 120
