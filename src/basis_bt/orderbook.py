"""Reconstruct a per-tick limit-order-book estimate from snapshot data.

The feed only shows the top ``N`` levels at each snapshot (N = 1 for IH, 5 for
the ETF).  The book kept here is the information the passive-fill queue model
needs: *how much resting volume is at my price right now*.

Per snapshot, in this order:

1. **Consume** – the snapshot's traded volume eats resting liquidity that the
   trade VWAP reached: ask levels priced <= VWAP from the best ask upward, bid
   levels priced >= VWAP from the best bid downward.
2. **Refresh** – levels better than the newly published best price have been
   traded through or cancelled and are removed; every published level overwrites
   the quantity at its price; deeper levels seen earlier but no longer published
   are kept (they are still the best available estimate).

Prices are stored as integer ticks so level matching is exact rather than a
float comparison with an epsilon.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .instruments import Instrument


class SideBook:
    """One side of the book as {price_in_ticks: qty}; ``is_ask`` sets priority order."""

    def __init__(self, is_ask: bool):
        self.is_ask = is_ask
        self.levels: dict[int, float] = {}

    def _better(self, a: int, b: int) -> bool:
        return a < b if self.is_ask else a > b

    def sorted_levels(self):
        return sorted(self.levels.items(), reverse=not self.is_ask)

    def consume(self, vwap_ticks: float, volume: float):
        budget = volume
        for price, qty in self.sorted_levels():
            if budget <= 0 or self._better(vwap_ticks, price):
                break
            take = min(budget, qty)
            budget -= take
            if qty - take > 0:
                self.levels[price] = qty - take
            else:
                del self.levels[price]

    def refresh(self, prices: np.ndarray, qtys: np.ndarray, max_depth: int):
        valid = ~np.isnan(prices)
        best = int(prices[valid][0])
        self.levels = {p: q for p, q in self.levels.items() if not self._better(p, best)}
        for p, q in zip(prices[valid].astype(int), qtys[valid]):
            self.levels[p] = q
        if len(self.levels) > max_depth:
            self.levels = dict(self.sorted_levels()[:max_depth])

    def serialize(self, tick_size: float) -> str:
        return "|".join(f"{p * tick_size:.4f}:{q:g}" for p, q in self.sorted_levels())


@lru_cache(maxsize=8192)
def parse_side(snapshot: str, tick_size: float) -> dict[int, float]:
    """Inverse of ``SideBook.serialize``: {price_in_ticks: qty}."""
    if not snapshot:
        return {}
    out = {}
    for level in snapshot.split("|"):
        p, q = level.split(":")
        out[round(float(p) / tick_size)] = float(q)
    return out


def build_books(ticks: pd.DataFrame, inst: Instrument, n_levels: int, max_depth: int = 10) -> pd.DataFrame:
    """Return the tick tape with the reconstructed ``book_bid`` / ``book_ask`` columns.

    The book is reset at the start of every trading day.  Remembered levels are
    capped at ``max_depth`` per side: anything deeper is too stale to matter for
    a queue position near the top of the book.
    """
    to_ticks = lambda c: np.round(ticks[c].to_numpy() / inst.tick_size)
    bid_px = np.column_stack([to_ticks(f"bid_px_{k}") for k in range(1, n_levels + 1)])
    ask_px = np.column_stack([to_ticks(f"ask_px_{k}") for k in range(1, n_levels + 1)])
    bid_q = np.column_stack([ticks[f"bid_qty_{k}"].to_numpy() for k in range(1, n_levels + 1)])
    ask_q = np.column_stack([ticks[f"ask_qty_{k}"].to_numpy() for k in range(1, n_levels + 1)])
    vol = ticks["volume"].to_numpy()
    vwap_ticks = np.where(vol > 0, inst.vwap(ticks["amount"].to_numpy(), np.where(vol > 0, vol, 1.0)),
                          np.nan) / inst.tick_size
    day = ticks["datetime"].dt.normalize().to_numpy()

    out_bid, out_ask = [], []
    bid, ask = SideBook(is_ask=False), SideBook(is_ask=True)
    for i in range(len(ticks)):
        if i == 0 or day[i] != day[i - 1]:
            bid, ask = SideBook(is_ask=False), SideBook(is_ask=True)
        if vol[i] > 0:
            ask.consume(vwap_ticks[i], vol[i])
            bid.consume(vwap_ticks[i], vol[i])
        bid.refresh(bid_px[i], bid_q[i], max_depth)
        ask.refresh(ask_px[i], ask_q[i], max_depth)
        out_bid.append(bid.serialize(inst.tick_size))
        out_ask.append(ask.serialize(inst.tick_size))

    cols = ["datetime", "seq", "last", "volume", "amount", "bid_px_1", "bid_qty_1", "ask_px_1", "ask_qty_1"]
    return ticks[cols].assign(book_bid=out_bid, book_ask=out_ask)
