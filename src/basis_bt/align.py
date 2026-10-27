"""Futures-spot alignment at bar level (signal input) and tick level (diagnostics).

Bar level
    Both instruments are already on the same 240-bar grid, so alignment reduces
    to choosing the common trading days.  Days present on only one side are
    reported, never silently inner-joined away minute by minute.  A bar whose
    latest quote is older than ``max_quote_age_s`` on either leg is flagged
    ``stale``: its basis mixes prices from different moments.

Tick level
    Each futures snapshot is matched with the latest ETF snapshot at or before
    it (backward as-of join, so no future spot information leaks in).  The ETF
    feed is stamped to the exact second while IH stamps are truncated, i.e. an
    ETF snapshot with an equal stamp is never later than the IH snapshot, so
    exact matches are allowed.  A ``tolerance`` stops a 13:00 futures tick from
    being paired with an 11:30 spot quote across the lunch break.
"""
from __future__ import annotations

import pandas as pd

from .cleaning import QualityReport
from .instruments import BASIS_SCALE
from .sessions import BARS_PER_DAY

BAR_COLUMNS = ["open", "high", "low", "close", "volume", "amount", "vwap", "bid", "ask", "mid",
               "n_trades", "quote_age_s"]


def build_minute_panel(ih_bars: pd.DataFrame, etf_bars: pd.DataFrame, report: QualityReport,
                       max_quote_age_s: float = 60.0) -> pd.DataFrame:
    ds = "minute panel"
    ih_days = set(ih_bars.index.normalize())
    etf_days = set(etf_bars.index.normalize())
    common = ih_days & etf_days
    report.add(ds, "days with IH only -> excluded", len(ih_days - common), len(ih_days),
               ", ".join(sorted(d.strftime("%Y-%m-%d") for d in ih_days - common))[:200])
    report.add(ds, "days with ETF only -> excluded", len(etf_days - common), len(etf_days),
               ", ".join(sorted(d.strftime("%Y-%m-%d") for d in etf_days - common))[:200])

    ih = ih_bars.loc[ih_bars.index.normalize().isin(common), BAR_COLUMNS].add_prefix("ih_")
    etf = etf_bars.loc[etf_bars.index.normalize().isin(common), BAR_COLUMNS].add_prefix("etf_")
    assert ih.index.equals(etf.index), "both legs must sit on the identical minute grid"

    panel = pd.concat([ih, etf], axis=1)
    panel["basis"] = panel["ih_mid"] - panel["etf_mid"] * BASIS_SCALE
    panel["basis_close"] = panel["ih_close"] - panel["etf_close"] * BASIS_SCALE
    panel["stale"] = ((panel["ih_quote_age_s"] > max_quote_age_s)
                      | (panel["etf_quote_age_s"] > max_quote_age_s)
                      | panel["basis"].isna())
    report.add(ds, f"bars with a quote older than {max_quote_age_s:.0f}s on either leg (stale)",
               panel["stale"].sum(), len(panel))

    assert len(panel) == BARS_PER_DAY * len(common)
    return panel


def asof_align_ticks(ih_ticks: pd.DataFrame, etf_ticks: pd.DataFrame,
                     tolerance: str = "10s") -> pd.DataFrame:
    """Tick-level panel: every IH snapshot with the prevailing ETF quote."""
    etf = etf_ticks[["datetime", "last", "bid_px_1", "ask_px_1"]].rename(
        columns={"datetime": "etf_time", "last": "etf_last", "bid_px_1": "etf_bid", "ask_px_1": "etf_ask"})
    ih = ih_ticks[["datetime", "seq", "last", "volume", "bid_px_1", "ask_px_1"]].rename(
        columns={"last": "ih_last", "volume": "ih_volume", "bid_px_1": "ih_bid", "ask_px_1": "ih_ask"})
    out = pd.merge_asof(ih, etf, left_on="datetime", right_on="etf_time",
                        direction="backward", allow_exact_matches=True,
                        tolerance=pd.Timedelta(tolerance))
    out["etf_lag_s"] = (out["datetime"] - out["etf_time"]).dt.total_seconds()
    out["basis"] = (out["ih_bid"] + out["ih_ask"]) / 2 - (out["etf_bid"] + out["etf_ask"]) / 2 * BASIS_SCALE
    assert (out["etf_lag_s"].dropna() >= 0).all(), "look-ahead: spot quote later than futures tick"
    return out
