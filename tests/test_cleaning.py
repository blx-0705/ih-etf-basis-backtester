import pandas as pd
import pytest

from basis_bt.cleaning import (DataQualityError, QualityReport, clean_etf_ticks, clean_ih_ticks,
                               combine_raw_files)


def _ms(stamp):
    return pd.Timestamp(stamp).tz_localize("Asia/Shanghai").tz_convert("UTC").value // 1_000_000


def _etf_raw(rows):
    lv = lambda p: "[" + ", ".join(f"{p - 0.001 * k:.3f}" for k in range(5)) + "]"
    lu = lambda p: "[" + ", ".join(f"{p + 0.001 * k:.3f}" for k in range(5)) + "]"
    return pd.DataFrame({
        "time": [_ms(t) for t, *_ in rows], "lastPrice": [p for _, p, *_ in rows],
        "volume": [v for *_, v, _ in rows], "amount": [a for *_, a in rows],
        "bidPrice": [lv(2.700)] * len(rows), "askPrice": [lu(2.701)] * len(rows),
        "bidVol": ["[100, 100, 100, 100, 100]"] * len(rows), "askVol": ["[100, 100, 100, 100, 100]"] * len(rows),
    })


def test_etf_cumulative_volume_is_differenced_before_session_filter():
    raw = _etf_raw([("2025-03-03 09:25:00", 2.700, 5000, 1_350_000.0),     # opening auction
                    ("2025-03-03 09:30:00", 2.701, 5010, 1_352_701.0),
                    ("2025-03-03 09:30:03", 2.701, 5030, 1_358_103.0)])
    out = clean_etf_ticks(raw, QualityReport())
    # the auction's 5000 lots must not leak into the first continuous snapshot
    assert list(out["volume"]) == [10.0, 20.0]


def test_etf_decreasing_cumulative_volume_fails_fast():
    raw = _etf_raw([("2025-03-03 09:30:00", 2.7, 100, 27_000.0), ("2025-03-03 09:30:03", 2.7, 90, 24_300.0)])
    with pytest.raises(DataQualityError):
        clean_etf_ticks(raw, QualityReport())


def _ih_raw(rows):
    return pd.DataFrame(rows, columns=["时间", "最新", "成交量", "成交额", "买一价", "买一量", "卖一价", "卖一量"]).astype(str)


def test_ih_corrupt_values_dropped_and_same_second_order_preserved():
    raw = _ih_raw([("2025-03-03 09:30:00", 2600.0, 1, 780000.0, 2599.8, 3, 2600.0, 2),
                   ("2025-03-03 09:30:00", 2600.2, 2, 1560120.0, 2600.0, 1, 2600.2, 4),
                   ("2025-03-03 09:30:01", 2600.2, 0, 0.0, 2600.0, 1, 2600.2, "2<")])
    rep = QualityReport()
    out = clean_ih_ticks(raw, rep)
    assert list(out["last"]) == [2600.0, 2600.2]           # vendor order within the second
    assert list(out["seq"]) == [0, 1]
    assert rep.to_frame().set_index("check").loc["unparsable numeric values -> row dropped", "count"] == 1


def test_crossed_quote_is_replaced_by_last_valid_quote():
    raw = _ih_raw([("2025-03-03 09:30:00", 2600.0, 0, 0.0, 2599.8, 3, 2600.0, 2),
                   ("2025-03-03 09:30:01", 2600.0, 0, 0.0, 2600.4, 3, 2600.2, 2)])
    out = clean_ih_ticks(raw, QualityReport())
    assert out.loc[1, "bid_px_1"] == 2599.8 and out.loc[1, "ask_px_1"] == 2600.0


def test_overlapping_raw_files_keep_each_day_once():
    a = pd.DataFrame({"时间": ["2025-03-03 09:30:00", "2025-03-04 09:30:00"], "x": [1, 2]})
    b = pd.DataFrame({"时间": ["2025-03-04 09:30:00", "2025-03-05 09:30:00"], "x": [2, 3]})
    out = combine_raw_files({"a": a, "b": b}, "时间", "IH raw", QualityReport())
    assert sorted(out["x"]) == [1, 2, 3]
