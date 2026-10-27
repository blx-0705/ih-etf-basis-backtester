"""Hedge-unit slippage attribution and FIFO round trips.

Five price anchors per hedge unit, each a pair of (IH price, ETF price):

    t0   signal      mids on the decision bar
    t1f  first quote leader limit price / follower mid when the leader first quoted
    t1w  weighted    the same, fill-weighted over every (re)quote that filled
    t2   fill tick   both mids on the ticks where the leader filled
    t3   execution   actual average prices of both legs

Consecutive anchors give four cost segments in index points (positive = cost):
``delay`` (t0->t1f), ``reprice`` (t1f->t1w), ``queue`` (t1w->t2) and ``chase``
(t2->t3), each split into IH and ETF legs.  By construction they telescope to
the signal-to-execution basis difference, which is asserted for every unit, as
is the passive-leg price lock (the leader always fills at its limit price, so
its own t3 equals t1w).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .instruments import BASIS_SCALE, IH, UNIT_SIZE
from .sessions import TradingClock

ANCHORS = ("t0", "t1f", "t1w", "t2", "t3")
SEGMENTS = {"delay": ("t0", "t1f"), "reprice": ("t1f", "t1w"), "queue": ("t1w", "t2"), "chase": ("t2", "t3")}
TOL = 1e-6


def _wavg(values: pd.Series, weights: pd.Series) -> float:
    return float((values * weights).sum() / weights.sum())


def hedge_units(fills: pd.DataFrame, pairs: pd.DataFrame, leader: str, clock: TradingClock) -> pd.DataFrame:
    """One row per COMPLETE hedge unit with its anchors and cost segments."""
    follower = "ETF" if leader == "IH" else "IH"
    tagged = fills.loc[fills["pair_id"] > 0]
    rows = []
    for pid, g in tagged.groupby("pair_id"):
        meta = pairs.loc[pairs["pair_id"] == pid].iloc[0]
        if meta["status"] != "COMPLETE":
            continue
        L, F = g.loc[g["symbol"] == leader], g.loc[g["symbol"] == follower]
        assert abs(L["qty"].sum() - UNIT_SIZE[leader]) < TOL and abs(F["qty"].sum() - UNIT_SIZE[follower]) < TOL, \
            f"hedge unit {pid} legs do not sum to one unit"
        first = L.loc[L["order_time"].idxmin()]
        a = {
            "t0": {"IH": meta["t0_ih"], "ETF": meta["t0_etf"]},
            "t1f": {leader: first["order_price"], follower: first["other_mid_at_order"]},
            "t1w": {leader: _wavg(L["order_price"], L["qty"]), follower: _wavg(L["other_mid_at_order"], L["qty"])},
            "t2": {leader: _wavg(L["mid_at_fill"], L["qty"]), follower: _wavg(L["other_mid_at_fill"], L["qty"])},
            "t3": {leader: _wavg(L["price"], L["qty"]), follower: _wavg(F["price"], F["qty"])},
        }
        assert abs(a["t3"][leader] - a["t1w"][leader]) < TOL, f"hedge unit {pid}: passive leg not filled at its limit"

        s = 1.0 if meta["direction"] == "OPEN" else -1.0      # OPEN sells the basis
        row = {"pair_id": pid, "direction": meta["direction"], "decision_time": meta["decision_time"],
               "leader_first_order": first["order_time"], "leader_last_fill": L["time"].max(),
               "follower_last_fill": F["time"].max(), "complete_time": meta["end_time"],
               "fees": g["fee"].sum()}
        for k in ANCHORS:
            row[f"ih_{k}"], row[f"etf_{k}"] = a[k]["IH"], a[k]["ETF"]
            row[f"basis_{k}"] = a[k]["IH"] - a[k]["ETF"] * BASIS_SCALE
        for name, (x, y) in SEGMENTS.items():
            row[f"{name}_ih"] = s * (a[x]["IH"] - a[y]["IH"])
            row[f"{name}_etf"] = s * (a[y]["ETF"] - a[x]["ETF"]) * BASIS_SCALE
            row[name] = row[f"{name}_ih"] + row[f"{name}_etf"]
        row["total_cost"] = sum(row[n] for n in SEGMENTS)
        assert abs(row["total_cost"] - s * (row["basis_t0"] - row["basis_t3"])) < TOL
        row["leader_wait_s"] = clock.seconds_between(row["leader_first_order"], row["leader_last_fill"])
        row["hedge_lag_s"] = clock.seconds_between(row["leader_last_fill"], row["follower_last_fill"])
        rows.append(row)
    return pd.DataFrame(rows)


def fifo_round_trips(units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match CLOSE units to the oldest OPEN unit.  Returns (round_trips, still_open).

    P&L per round trip in CNY: ``signal_pnl`` at t0 prices, ``gross_pnl`` at t3,
    ``net_pnl`` after fees.  ``signal_pnl - net_pnl`` equals the summed cost
    segments of both units times the IH multiplier plus fees (asserted).
    """
    queue, trips = [], []
    for _, u in units.sort_values("complete_time", kind="stable").iterrows():
        if u["direction"] == "OPEN":
            queue.append(u)
            continue
        assert queue, f"CLOSE unit {u['pair_id']} has no open unit to match"
        o = queue.pop(0)
        trip = {"open_pair": o["pair_id"], "close_pair": u["pair_id"],
                "open_time": o["complete_time"], "close_time": u["complete_time"],
                "open_basis_t0": o["basis_t0"], "close_basis_t0": u["basis_t0"],
                "open_basis_t3": o["basis_t3"], "close_basis_t3": u["basis_t3"]}
        trip["signal_pnl"] = (o["basis_t0"] - u["basis_t0"]) * IH.multiplier
        trip["gross_pnl"] = (o["basis_t3"] - u["basis_t3"]) * IH.multiplier
        trip["fees"] = o["fees"] + u["fees"]
        trip["net_pnl"] = trip["gross_pnl"] - trip["fees"]
        for name in SEGMENTS:
            for leg in ("", "_ih", "_etf"):
                trip[f"{name}{leg}_cost"] = (o[f"{name}{leg}"] + u[f"{name}{leg}"]) * IH.multiplier
        trip["shortfall"] = trip["signal_pnl"] - trip["net_pnl"]
        explained = sum(trip[f"{n}_cost"] for n in SEGMENTS) + trip["fees"]
        assert abs(trip["shortfall"] - explained) < 1e-4, "shortfall decomposition does not close"
        trips.append(trip)
    n_open = int((units["direction"] == "OPEN").sum())
    assert n_open == len(trips) + len(queue), "FIFO conservation: opened != closed + still open"
    return pd.DataFrame(trips), pd.DataFrame(queue)


def summarize_costs(units: pd.DataFrame) -> pd.DataFrame:
    """Average cost per hedge unit (index points) by direction and segment."""
    cols = [f"{n}{leg}" for n in SEGMENTS for leg in ("_ih", "_etf", "")] + ["total_cost"]
    out = units.groupby("direction")[cols + ["leader_wait_s", "hedge_lag_s"]].mean().T
    out["ALL"] = units[cols + ["leader_wait_s", "hedge_lag_s"]].mean()
    return out
