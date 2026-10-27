"""Post-trade alignment: fills -> minute positions, daily P&L, reconciliation, metrics.

Minute positions use the same end-stamped bar convention as the market data: a
fill at 10:00:30 is part of the position reported on the 10:01 bar.  (Flooring
fill times to the minute, as the prototype did, reports every position one bar
early and makes fills look as if they traded on the previous bar's prices.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .instruments import INSTRUMENTS, UNIT_SIZE
from .sessions import bar_label

TRADING_DAYS = 242


def _signed(fills: pd.DataFrame, sym: str) -> pd.Series:
    f = fills.loc[fills["symbol"] == sym]
    return f["side"] * f["qty"]


def minute_positions(fills: pd.DataFrame, signal: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=signal.index)
    labels = bar_label(fills["time"])
    for sym in ("IH", "ETF"):
        mask = fills["symbol"] == sym
        flow = _signed(fills, sym).groupby(labels[mask]).sum()
        assert flow.index.isin(out.index).all(), f"{sym} fills outside the minute grid"
        out[f"{sym.lower()}_pos"] = flow.reindex(out.index, fill_value=0.0).cumsum()
    out["target_ih_qty"] = signal["target_ih_qty"]
    out["ih_gap"] = out["target_ih_qty"] - out["ih_pos"]
    out["hedge_imbalance_units"] = out["ih_pos"] + out["etf_pos"] / UNIT_SIZE["ETF"]
    return out


def daily_pnl(fills: pd.DataFrame, panel: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """Holding + trading P&L per day, reconciled against cash accounting."""
    closes = panel.groupby(panel.index.normalize())[["ih_mid", "etf_mid"]].last()
    closes.columns = ["IH", "ETF"]
    day = fills["time"].dt.normalize()
    out = pd.DataFrame(index=closes.index)
    cash = pd.Series(0.0, index=closes.index)
    mtm = pd.Series(0.0, index=closes.index)
    for sym in ("IH", "ETF"):
        inst, f = INSTRUMENTS[sym], fills["symbol"] == sym
        signed = _signed(fills, sym)
        px, close = fills.loc[f, "price"], closes[sym]
        fee = fills.loc[f, "fee"].groupby(day[f]).sum().reindex(out.index, fill_value=0.0)
        trade = (signed * (close.reindex(day[f]).to_numpy() - px) * inst.multiplier).groupby(day[f]).sum()
        pos = signed.groupby(day[f]).sum().reindex(out.index, fill_value=0.0).cumsum()
        out[f"{sym.lower()}_pos"] = pos
        out[f"{sym.lower()}_holding_pnl"] = pos.shift(1, fill_value=0.0) * close.diff().fillna(0.0) * inst.multiplier
        out[f"{sym.lower()}_trading_pnl"] = trade.reindex(out.index, fill_value=0.0) - fee
        out[f"{sym.lower()}_fees"] = fee
        cash -= ((signed * px * inst.multiplier).groupby(day[f]).sum().reindex(out.index, fill_value=0.0) + fee)
        mtm += pos * close * inst.multiplier

    out["holding_pnl"] = out["ih_holding_pnl"] + out["etf_holding_pnl"]
    out["trading_pnl"] = out["ih_trading_pnl"] + out["etf_trading_pnl"]
    out["total_pnl"] = out["holding_pnl"] + out["trading_pnl"]
    out["equity"] = initial_capital + out["total_pnl"].cumsum()

    accounting_equity = initial_capital + cash.cumsum() + mtm
    assert np.allclose(out["equity"], accounting_equity, rtol=0, atol=1e-3), \
        "P&L decomposition does not reconcile with cash + mark-to-market"
    return out


def metrics(daily: pd.DataFrame, initial_capital: float) -> dict:
    eq = daily["equity"]
    ret = eq.pct_change().fillna(eq.iloc[0] / initial_capital - 1)
    n = len(daily)
    total = eq.iloc[-1] / initial_capital - 1
    ann = (1 + total) ** (TRADING_DAYS / n) - 1
    vol = ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
    dd = eq / eq.cummax().clip(lower=initial_capital) - 1
    return {"days": n, "total_return": total, "annual_return": ann, "annual_vol": vol,
            "sharpe": ret.mean() / ret.std(ddof=1) * np.sqrt(TRADING_DAYS) if ret.std() > 0 else np.nan,
            "max_drawdown": dd.min(), "calmar": ann / -dd.min() if dd.min() < 0 else np.nan,
            "total_fees": daily["ih_fees"].sum() + daily["etf_fees"].sum()}


def execution_quality(minute_pos: pd.DataFrame, events: pd.DataFrame) -> dict:
    passive = events["status"].value_counts()
    return {
        "minutes_on_target": float((minute_pos["ih_gap"] == 0).mean()),
        "mean_abs_gap_contracts": float(minute_pos["ih_gap"].abs().mean()),
        "max_hedge_imbalance_units": float(minute_pos["hedge_imbalance_units"].abs().max()),
        "passive_orders": int(len(events)),
        **{f"passive_{k.lower()}": int(v) for k, v in passive.items()},
        "passive_median_wait_s_filled": float(events.loc[events["status"] == "FILLED", "trading_seconds"].median()),
    }
