import numpy as np
import pandas as pd

from basis_bt.align import asof_align_ticks, build_minute_panel
from basis_bt.bars import make_bars
from basis_bt.cleaning import QualityReport
from basis_bt.instruments import IH

from conftest import make_ticks


def _bars(rows):
    return make_bars(make_ticks(rows), IH, QualityReport())


def test_open_comes_from_first_trade_not_from_stale_last_price():
    bars = _bars([
        dict(datetime="2025-03-03 09:30:10", last=2600.0, volume=1, amount=780_000.0, bid=2599.8, ask=2600.0),
        # 09:31:00 snapshot without a trade still shows last=2600.0
        dict(datetime="2025-03-03 09:31:00", last=2600.0, volume=0, amount=0.0, bid=2601.0, ask=2601.2),
        dict(datetime="2025-03-03 09:31:20", last=2601.2, volume=2, amount=1_560_720.0, bid=2601.0, ask=2601.2),
    ])
    assert bars.loc["2025-03-03 09:32", "open"] == 2601.2


def test_quiet_minute_is_an_explicit_zero_volume_bar():
    bars = _bars([
        dict(datetime="2025-03-03 09:30:10", last=2600.0, volume=1, amount=780_000.0, bid=2599.8, ask=2600.0),
        dict(datetime="2025-03-03 09:33:10", last=2600.4, volume=1, amount=780_120.0, bid=2600.2, ask=2600.4),
    ])
    quiet = bars.loc["2025-03-03 09:32"]
    assert quiet["volume"] == 0 and quiet["close"] == 2600.0 and quiet["open"] == 2600.0
    assert quiet["quote_age_s"] == 110.0          # last quote 09:30:10, bar ends 09:32:00
    assert len(bars) == 240


def test_panel_uses_common_days_only_and_flags_stale_quotes():
    day1 = [dict(datetime="2025-03-03 09:30:10", last=2600.0, volume=1, amount=780_000.0, bid=2599.8, ask=2600.0)]
    day2 = [dict(datetime="2025-03-04 09:30:10", last=2600.0, volume=1, amount=780_000.0, bid=2599.8, ask=2600.0)]
    ih = _bars(day1 + day2)
    etf = _bars(day2)
    rep = QualityReport()
    panel = build_minute_panel(ih, etf, rep)
    assert set(panel.index.normalize()) == {pd.Timestamp("2025-03-04")}
    assert panel.loc["2025-03-04 09:31", "stale"] == False      # noqa: E712
    assert panel.loc["2025-03-04 15:00", "stale"] == True       # noqa: E712


def test_tick_asof_alignment_never_looks_ahead_and_does_not_cross_lunch():
    ih = make_ticks([dict(datetime=t, last=1, volume=0, amount=0, bid=1, ask=1)
                     for t in ("2025-03-03 10:00:01", "2025-03-03 10:00:02", "2025-03-03 13:00:00")])
    etf = make_ticks([dict(datetime=t, last=1, volume=0, amount=0, bid=1, ask=1)
                      for t in ("2025-03-03 10:00:00", "2025-03-03 10:00:02", "2025-03-03 11:30:00")])
    out = asof_align_ticks(ih, etf, tolerance="10s")
    assert list(out["etf_lag_s"].iloc[:2]) == [1.0, 0.0]
    assert np.isnan(out["etf_lag_s"].iloc[2])     # 13:00 must not pair with the 11:30 quote
