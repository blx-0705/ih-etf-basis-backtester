import numpy as np
import pandas as pd

from basis_bt.execution import BUY, SELL, AggressiveExecutor, MarketTape, PassiveQueueExecutor
from basis_bt.instruments import ETF, IH
from basis_bt.orderbook import SideBook, build_books, parse_side
from basis_bt.sessions import TradingClock, minute_grid

from conftest import make_ticks

T0 = pd.Timestamp("2025-03-03 10:00:00")


def test_side_book_consume_then_refresh():
    ask = SideBook(is_ask=True)
    ask.refresh(np.array([100.0, 101.0]), np.array([10.0, 10.0]), max_depth=10)
    ask.consume(vwap_ticks=101.0, volume=15)          # eats level 100 fully, 5 of level 101
    assert ask.levels == {101: 5.0}
    ask.refresh(np.array([102.0]), np.array([7.0]), max_depth=10)
    assert ask.levels == {102: 7.0}                   # 101 traded through: removed


def test_etf_book_vwap_uses_board_lots():
    """Regression: the prototype divided ETF amount by lots instead of shares, so
    the tick VWAP was 100x the price and every trade 'consumed' the ask side."""
    ticks = make_ticks([
        dict(datetime="2025-03-03 10:00:00", last=2.700, volume=0, amount=0, bid=2.700, ask=2.701,
             bid_qty=50, ask_qty=50),
        # 80 lots sold into the bid at 2.700; deeper ask level 2.702 is not re-published
        dict(datetime="2025-03-03 10:00:03", last=2.700, volume=80, amount=80 * 100 * 2.700, bid=2.699,
             ask=2.701, bid_qty=50, ask_qty=50),
    ], n_levels=5)
    ticks.loc[0, ["ask_px_2", "ask_qty_2"]] = [2.702, 50]
    books = build_books(ticks, ETF, n_levels=5)
    assert parse_side(books.loc[1, "book_ask"], ETF.tick_size) == {2701: 50.0, 2702: 50.0}
    assert parse_side(books.loc[1, "book_bid"], ETF.tick_size) == {2699: 50.0}


def _tape(inst, rows, book_bid, book_ask):
    t = make_ticks(rows)
    t["book_bid"], t["book_ask"] = book_bid, book_ask
    return MarketTape(inst, t)


def _other():
    return _tape(ETF, [dict(datetime=T0, last=2.7, volume=0, amount=0, bid=2.7, ask=2.701)],
                 ["2.7000:100"], ["2.7010:100"])


def _passive(tape, price_rule="join"):
    return PassiveQueueExecutor(tape, _other(), price_rule, events=[], trace=None,
                                clock=TradingClock(minute_grid([T0])))


def _ih_rows(vols, bid=2600.0, ask=2600.2, px=2600.0):
    return [dict(datetime=T0 + pd.Timedelta(seconds=k), last=px, volume=v, amount=v * px * 300,
                 bid=bid, ask=ask) for k, v in enumerate(vols)]


def test_passive_fill_waits_for_queue_and_does_not_double_count():
    # queue ahead at 2600.0 is 5 lots; 4 trade, then 3 trade -> 1 left for me
    tape = _tape(IH, _ih_rows([9, 4, 3]), ["2600.0000:5"] * 3, ["2600.2000:5"] * 3)
    res = _passive(tape).execute(BUY, 1.0, T0, T0 + pd.Timedelta(minutes=1), T0)
    assert res.filled == 1.0
    assert res.fills[0]["time"] == T0 + pd.Timedelta(seconds=2)   # 9 lots on the placement tick don't count
    assert res.fills[0]["price"] == 2600.0


def test_passive_queue_not_consumed_by_trades_away_from_my_price():
    rows = _ih_rows([0, 5, 1], px=2600.2)            # trades at the ask, never reach my bid
    tape = _tape(IH, rows, ["2600.0000:2"] * 3, ["2600.2000:5"] * 3)
    ex = _passive(tape)
    assert ex.execute(BUY, 1.0, T0, T0 + pd.Timedelta(minutes=1), T0).filled == 0
    assert ex.order.queue_ahead == 2


def test_cancellations_shrink_queue_ahead():
    tape = _tape(IH, _ih_rows([0, 0, 1]), ["2600.0000:20", "2600.0000:1", "2600.0000:1"], ["2600.2000:5"] * 3)
    res = _passive(tape).execute(BUY, 1.0, T0, T0 + pd.Timedelta(minutes=1), T0)
    assert res.filled == 0                            # 1 lot traded, 1 was still ahead of me
    tape = _tape(IH, _ih_rows([0, 0, 2]), ["2600.0000:20", "2600.0000:1", "2600.0000:1"], ["2600.2000:5"] * 3)
    assert _passive(tape).execute(BUY, 1.0, T0, T0 + pd.Timedelta(minutes=1), T0).filled == 1


def test_mid_rule_snaps_to_the_tick_grid():
    tape = _tape(IH, _ih_rows([0, 2]), ["2600.0000:0"] * 2, ["2600.2000:5"] * 2)
    ex = _passive(tape, "mid")
    ex.execute(SELL, 1.0, T0, T0 + pd.Timedelta(minutes=1), T0)
    assert ex.order.price == 2600.2                   # mid 2600.1 is not a valid limit price


def test_aggressive_capacity_cap():
    rows = [dict(datetime=T0 + pd.Timedelta(seconds=3 * k), last=2.701, volume=100, amount=100 * 100 * 2.701,
                 bid=2.700, ask=2.701, ask_qty=1000) for k in range(3)]
    tape = _tape(ETF, rows, ["2.7000:1"] * 3, ["2.7010:1000"] * 3)
    ex = AggressiveExecutor(tape, _tape(IH, _ih_rows([0]), ["2600.0000:1"], ["2600.2000:1"]), 0.7, 0.1)
    res = ex.execute(BUY, 150_000.0, T0, T0 + pd.Timedelta(minutes=1), T0)
    assert [f["qty"] for f in res.fills] == [70_000.0, 70_000.0, 10_000.0]
    assert all(f["price"] == 2.701 for f in res.fills)
