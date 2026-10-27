"""Tick-replay fill models: passive queue-position model and aggressive taker model.

Layering (the original prototype's main design lesson):

* **Market layer** – ``MarketTape``: read-only arrays from the feed.  Nothing the
  strategy does ever modifies them.
* **Queue-state layer** – ``PassiveQueueExecutor`` keeps one number per resting
  order, ``queue_ahead``: volume that must trade at my price before I can fill.
* **Observation layer** – optional trace rows explaining how ``queue_ahead``
  evolved.  It is write-only and cannot feed back into fills.

All quantities passed to and returned from executors are in base units
(IH contracts, ETF shares).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

from .instruments import Instrument
from .orderbook import parse_side

BUY, SELL = 1, -1


def _ns(t) -> np.datetime64:
    return np.datetime64(pd.Timestamp(t), "ns")


class MarketTape:
    """One instrument, one day, as numpy arrays (tick times are non-decreasing)."""

    def __init__(self, inst: Instrument, ticks: pd.DataFrame):
        self.inst = inst
        self.time = ticks["datetime"].to_numpy("datetime64[ns]")
        self.bid = ticks["bid_px_1"].to_numpy(float)
        self.ask = ticks["ask_px_1"].to_numpy(float)
        self.bid_qty = ticks["bid_qty_1"].to_numpy(float)
        self.ask_qty = ticks["ask_qty_1"].to_numpy(float)
        self.volume = ticks["volume"].to_numpy(float)
        self.amount = ticks["amount"].to_numpy(float)
        self.mid = (self.bid + self.ask) / 2
        self.book_bid = ticks["book_bid"].to_numpy(object)
        self.book_ask = ticks["book_ask"].to_numpy(object)

    def window(self, start, end) -> tuple[int, int]:
        """Index range of ticks with start <= time < end."""
        return (int(np.searchsorted(self.time, _ns(start), side="left")),
                int(np.searchsorted(self.time, _ns(end), side="left")))

    def mid_asof(self, t) -> float:
        i = int(np.searchsorted(self.time, _ns(t), side="right")) - 1
        return self.mid[i] if i >= 0 else np.nan

    def book(self, i: int, side: int) -> dict[int, float]:
        return parse_side(self.book_bid[i] if side == BUY else self.book_ask[i], self.inst.tick_size)


def _floor_to(qty: float, step: float) -> float:
    return math.floor(qty / step + 1e-9) * step


def _fill(tape: MarketTape, other: MarketTape, i: int, side: int, qty: float, price: float,
          order: "OrderState", role: str) -> dict:
    return {
        "time": pd.Timestamp(tape.time[i]), "symbol": tape.inst.symbol, "side": side, "qty": qty,
        "price": price, "fee": tape.inst.notional(price, qty) * tape.inst.fee_rate, "role": role,
        "order_id": order.order_id, "order_time": order.placed_time, "order_price": order.price,
        "mid_at_order": order.mid_at_order, "other_mid_at_order": order.other_mid_at_order,
        "mid_at_fill": tape.mid[i], "other_mid_at_fill": other.mid_asof(tape.time[i]),
    }


@dataclass
class OrderState:
    order_id: int
    side: int
    price: float
    placed_time: pd.Timestamp
    mid_at_order: float
    other_mid_at_order: float
    decision_time: pd.Timestamp
    price_key: int = 0
    qty: float = 0.0
    queue_ahead: float = 0.0          # in volume units
    filled: float = 0.0


@dataclass
class ExecResult:
    filled: float
    last_time: pd.Timestamp | None
    fills: list = field(default_factory=list)


class PassiveQueueExecutor:
    """Resting limit order filled only after the volume queued ahead of it trades.

    Price rule ``mid``: the mid snapped to the passive side of the tick grid
    (bid when the spread is one tick, inside the spread when it is wider).  The
    prototype quoted the raw mid, which is not a valid limit price for a
    one-tick spread.  Price rule ``join``: best price on my own side.

    Queue rules, per tick after placement:
      * trades only consume my queue when the tick VWAP reached my price;
      * of a tick's volume, ``min(queue_ahead, volume)`` goes to orders ahead of
        me and only the remainder can fill me (the prototype subtracted the
        same volume twice);
      * cancellations: the queue ahead can never exceed what the book still
        shows at my price; if my price is now the best or better and the level
        is gone, everyone ahead has left and ``queue_ahead = 0``.
    A resting order keeps its queue position across calls while side, price and
    decision are unchanged and the size is not increased; any other change
    cancels it and the new order joins the back of the queue.
    """

    def __init__(self, tape: MarketTape, other: MarketTape, price_rule: str, events: list,
                 trace: list | None, clock):
        self.tape, self.other, self.price_rule = tape, other, price_rule
        self.events, self.trace, self.clock = events, trace, clock
        self.order: OrderState | None = None
        self._next_id = 0

    def _limit_price_key(self, i: int, side: int) -> int:
        tick = self.tape.inst.tick_size
        if self.price_rule == "join":
            return round((self.tape.bid[i] if side == BUY else self.tape.ask[i]) / tick)
        mid_ticks = self.tape.mid[i] / tick
        return math.floor(mid_ticks + 1e-9) if side == BUY else math.ceil(mid_ticks - 1e-9)

    def _visible_queue(self, i: int, o: OrderState) -> float | None:
        book = self.tape.book(i, o.side)
        if o.price_key in book:
            return book[o.price_key]
        best = max(book) if o.side == BUY else min(book)
        at_or_better = o.price_key >= best if o.side == BUY else o.price_key <= best
        return 0.0 if at_or_better else None

    def cancel(self, t, status: str):
        if self.order is not None:
            self._emit(self.order, t, status)
            self.order = None

    def _emit(self, o: OrderState, t, status: str):
        self.events.append({
            "order_id": o.order_id, "symbol": self.tape.inst.symbol, "side": o.side, "price": o.price,
            "decision_time": o.decision_time, "placed_time": o.placed_time, "end_time": pd.Timestamp(t),
            "status": status, "qty": o.qty, "filled": o.filled,
            "trading_seconds": self.clock.seconds_between(o.placed_time, pd.Timestamp(t)),
        })

    def execute(self, side: int, qty: float, start, end, decision_time) -> ExecResult:
        tape, inst = self.tape, self.tape.inst
        i0, i1 = tape.window(start, end)
        if i0 >= i1:
            return ExecResult(0.0, None)

        key = self._limit_price_key(i0, side)
        o = self.order
        resume = (o is not None and o.side == side and o.price_key == key
                  and o.decision_time == decision_time and qty <= o.qty - o.filled + 1e-9)
        if resume:
            first = i0
            o.qty = o.filled + qty
        else:
            if o is not None:
                self.cancel(tape.time[i0], "CANCELLED" if o.side != side else "REPRICED")
            self._next_id += 1
            o = self.order = OrderState(
                order_id=self._next_id, side=side, price=round(key * inst.tick_size, 6), price_key=key,
                placed_time=pd.Timestamp(tape.time[i0]), mid_at_order=tape.mid[i0],
                other_mid_at_order=self.other.mid_asof(tape.time[i0]), decision_time=decision_time,
                qty=qty, queue_ahead=tape.book(i0, side).get(key, 0.0))
            # trades printed on the placement tick happened before the order arrived
            first = i0 + 1

        res = ExecResult(0.0, None)
        for i in range(first, i1):
            v = tape.volume[i]
            if v > 0:
                px = inst.vwap(tape.amount[i], v)
                if (px <= o.price + 1e-9) if side == BUY else (px >= o.price - 1e-9):
                    ahead = min(o.queue_ahead, v)
                    o.queue_ahead -= ahead
                    fill = _floor_to(min((v - ahead) * inst.volume_unit, qty - res.filled), inst.min_fill)
                    if fill > 0:
                        res.fills.append(_fill(tape, self.other, i, side, fill, o.price, o, "passive"))
                        res.filled += fill
                        o.filled += fill
                        res.last_time = pd.Timestamp(tape.time[i])
            visible = self._visible_queue(i, o)
            if visible is not None:
                o.queue_ahead = min(o.queue_ahead, visible)
            if self.trace is not None:
                self.trace.append({"order_id": o.order_id, "symbol": inst.symbol, "tick_time": tape.time[i],
                                   "price": o.price, "queue_ahead": o.queue_ahead, "visible": visible,
                                   "tick_volume": v, "filled": res.filled})
            if res.filled >= qty - 1e-9:
                self._emit(o, tape.time[i], "FILLED")
                self.order = None
                break
        return res


class AggressiveExecutor:
    """Crosses the spread; each tick can absorb a capped share of available size.

    capacity per tick = max(take_ratio * opposite L1 size, participation * tick volume)
    """

    def __init__(self, tape: MarketTape, other: MarketTape, take_ratio: float, participation: float):
        self.tape, self.other = tape, other
        self.take_ratio, self.participation = take_ratio, participation
        self._next_id = 0

    def execute(self, side: int, qty: float, start, end, decision_time) -> ExecResult:
        tape, inst = self.tape, self.tape.inst
        i0, i1 = tape.window(start, end)
        res = ExecResult(0.0, None)
        if i0 >= i1:
            return res
        self._next_id += 1
        order = OrderState(order_id=self._next_id, side=side, price=tape.mid[i0],
                           placed_time=pd.Timestamp(tape.time[i0]), mid_at_order=tape.mid[i0],
                           other_mid_at_order=self.other.mid_asof(tape.time[i0]), decision_time=decision_time)
        price = tape.ask if side == BUY else tape.bid
        opposite = tape.ask_qty if side == BUY else tape.bid_qty
        cap = np.maximum(self.take_ratio * opposite[i0:i1], self.participation * tape.volume[i0:i1])
        for k, i in enumerate(range(i0, i1)):
            fill = _floor_to(min(cap[k] * inst.volume_unit, qty - res.filled), inst.min_fill)
            if fill > 0:
                res.fills.append(_fill(tape, self.other, i, side, fill, price[i], order, "aggressive"))
                res.filled += fill
                res.last_time = pd.Timestamp(tape.time[i])
                if res.filled >= qty - 1e-9:
                    break
        return res
