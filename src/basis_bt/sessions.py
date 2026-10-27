"""China A-share / CFFEX continuous-trading calendar.

Bar convention (matches the vendor's 1-minute bars, verified in
``validation.compare_with_vendor``): a bar is **labelled by its end time** and
covers ``[label - 1min, label)``.  The 10:01 bar therefore holds ticks stamped
10:00:00 .. 10:00:59.  The closing snapshots stamped 11:30:00 / 15:00:00 (plus a
few seconds of publication delay) belong to the last bar of their session
instead of opening a new one.

Consequence for the backtest: a signal computed on bar ``t`` uses information up
to ``t`` and may only be executed on ticks at or after ``t``.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

SESSIONS = ((dt.time(9, 30), dt.time(11, 30)), (dt.time(13, 0), dt.time(15, 0)))
BARS_PER_DAY = 240
# The session's final snapshot (closing auction on SSE) is stamped up to ~3s
# after 11:30:00 / 15:00:00 and carries real volume; it belongs to the last bar.
CLOSE_GRACE = dt.timedelta(seconds=5)


def in_session(ts: pd.Series) -> pd.Series:
    """True for ticks inside a continuous-trading session (start inclusive,
    end inclusive plus ``CLOSE_GRACE``)."""
    t = ts.dt.time
    mask = pd.Series(False, index=ts.index)
    for start, end in SESSIONS:
        end_grace = (dt.datetime.combine(dt.date.min, end) + CLOSE_GRACE).time()
        mask |= (t >= start) & (t <= end_grace)
    return mask


def bar_label(ts: pd.Series) -> pd.Series:
    """End-stamped 1-minute bar label for in-session tick timestamps."""
    label = ts.dt.floor("min") + pd.Timedelta(minutes=1)
    # closing snapshots (11:30:00.x / 15:00:00.x) fold into the session's last bar
    for _, end in SESSIONS:
        overflow = label.dt.time == (dt.datetime.combine(dt.date.min, end)
                                     + dt.timedelta(minutes=1)).time()
        label = label.where(~overflow, label - pd.Timedelta(minutes=1))
    return label


def minute_grid(days) -> pd.DatetimeIndex:
    """Canonical 240-bar grid (09:31..11:30, 13:01..15:00) for the given days."""
    offsets = []
    for start, end in SESSIONS:
        first = pd.Timedelta(hours=start.hour, minutes=start.minute + 1)
        n = (end.hour * 60 + end.minute) - (start.hour * 60 + start.minute)
        offsets.append(first + pd.to_timedelta(np.arange(n), unit="min"))
    offsets = offsets[0].append(offsets[1])
    assert len(offsets) == BARS_PER_DAY
    days = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(days).normalize())))
    return pd.DatetimeIndex((days.values[:, None] + offsets.values[None, :]).ravel(),
                            name="datetime")


class TradingClock:
    """Measures elapsed *trading* seconds, skipping lunch, nights and holidays.

    Built from the minute grid of the backtest: every bar contributes 60 seconds.
    """

    def __init__(self, grid: pd.DatetimeIndex):
        # bar k covers [grid[k] - 60s, grid[k])
        self._starts = (grid - pd.Timedelta(minutes=1)).to_numpy("datetime64[ns]")

    def _cum(self, t) -> float:
        t = np.datetime64(pd.Timestamp(t), "ns")
        k = np.searchsorted(self._starts, t, side="right") - 1
        if k < 0:
            return 0.0
        within = (t - self._starts[k]) / np.timedelta64(1, "s")
        return k * 60.0 + min(max(within, 0.0), 60.0)

    def seconds_between(self, t0, t1) -> float:
        if pd.isna(t0) or pd.isna(t1):
            return np.nan
        return self._cum(t1) - self._cum(t0)
