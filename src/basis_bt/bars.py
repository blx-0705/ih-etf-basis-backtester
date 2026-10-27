"""Clean ticks -> 1-minute bars on the canonical 240-bar grid.

Two conventions here were the main sources of disagreement with the vendor's
official bars in the original prototype (see docs/data_quality_findings.md):

1. **OHLC from trades only.**  A snapshot's ``last`` is the last trade price,
   which is carried forward even when nothing traded in that snapshot.  Taking
   ``first`` over all snapshots makes the bar open equal to the *previous* bar's
   close whenever the minute starts with a quote-only update.
2. **Every minute exists.**  Bars are re-indexed onto the full session grid, so
   a quiet minute becomes an explicit zero-volume bar (OHLC = previous close,
   quotes carried forward within the day) instead of a missing row that makes
   a later inner join silently drop the other instrument's minute.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cleaning import QualityReport
from .instruments import Instrument
from .sessions import BARS_PER_DAY, bar_label, minute_grid


def make_bars(ticks: pd.DataFrame, inst: Instrument, report: QualityReport) -> pd.DataFrame:
    ds = f"{inst.symbol} bars"
    t = ticks.assign(label=bar_label(ticks["datetime"]))

    trades = t.loc[t["volume"] > 0]
    ohlc = trades.groupby("label")["last"].agg(open="first", high="max", low="min", close="last")
    ohlc["n_trades"] = trades.groupby("label").size()
    flow = t.groupby("label")[["volume", "amount"]].sum()
    quotes = t.groupby("label").agg(bid=("bid_px_1", "last"), ask=("ask_px_1", "last"),
                                    quote_time=("datetime", "last"))

    grid = minute_grid(t["datetime"])
    bars = pd.concat([ohlc, flow, quotes], axis=1).reindex(grid)
    day = bars.index.normalize()

    missing = bars["quote_time"].isna()
    report.add(ds, "grid minutes without any snapshot -> quotes carried forward", missing.sum(), len(bars))
    bars[["bid", "ask", "quote_time"]] = bars.groupby(day)[["bid", "ask", "quote_time"]].ffill()
    bars["quote_age_s"] = (bars.index.to_series() - bars["quote_time"]).dt.total_seconds()

    no_trade = bars["n_trades"].isna()
    report.add(ds, "grid minutes without trades -> OHLC = previous close, volume 0", no_trade.sum(), len(bars))
    bars[["volume", "amount", "n_trades"]] = bars[["volume", "amount", "n_trades"]].fillna(0)
    bars["close"] = bars.groupby(day)["close"].ffill()
    for c in ("open", "high", "low"):
        bars[c] = bars[c].fillna(bars["close"])
    report.add(ds, "minutes before the day's first trade (OHLC undefined)", bars["close"].isna().sum(), len(bars))

    bars["mid"] = (bars["bid"] + bars["ask"]) / 2
    bars["vwap"] = inst.vwap(bars["amount"], bars["volume"].where(bars["volume"] > 0))

    traded = bars["n_trades"] > 0
    tol = inst.tick_size / 2
    outside = traded & ((bars["vwap"] < bars["low"] - tol) | (bars["vwap"] > bars["high"] + tol))
    report.add(ds, "bar VWAP outside [low, high]", outside.sum(), traded.sum())

    assert bars.index.is_unique and bars.index.is_monotonic_increasing
    assert len(bars) == BARS_PER_DAY * day.nunique()
    assert (bars.loc[traded, "low"] <= bars.loc[traded, ["open", "close"]].min(axis=1)).all()
    assert (bars.loc[traded, "high"] >= bars.loc[traded, ["open", "close"]].max(axis=1)).all()
    return bars.drop(columns="quote_time")
