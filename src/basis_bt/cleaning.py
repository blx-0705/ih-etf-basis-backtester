"""Raw vendor ticks -> one clean, typed tick schema shared by both instruments.

Output schema (one row per snapshot, sorted by ``datetime`` then ``seq``)::

    datetime  seq  last  volume  amount  bid_px_1..N  bid_qty_1..N  ask_px_1..N  ask_qty_1..N

* ``volume`` / ``amount`` are per-snapshot increments, ``volume`` in the
  instrument's volume unit (IH contracts, ETF board lots).
* ``seq`` is the vendor's row order.  IH timestamps are truncated to the second
  while the exchange publishes two snapshots per second, so ``seq`` is the only
  valid intra-second order and every sort in the pipeline must be stable on it.
* Only continuous-session snapshots are kept (see ``sessions.in_session``).

Every rule that drops or edits data records a count in a ``QualityReport`` so a
run can be audited end to end instead of silently shrinking the data set.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .sessions import in_session


class DataQualityError(ValueError):
    """Raised when raw data violates an assumption the pipeline cannot repair."""


@dataclass
class QualityReport:
    rows: list = field(default_factory=list)

    def add(self, dataset: str, check: str, count: int, total: int | None = None, note: str = ""):
        self.rows.append({"dataset": dataset, "check": check, "count": int(count),
                          "total": None if total is None else int(total), "note": note})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["dataset", "check", "count", "total", "note"])


def level_columns(n_levels: int) -> list[str]:
    cols = []
    for side in ("bid", "ask"):
        cols += [f"{side}_px_{k}" for k in range(1, n_levels + 1)]
        cols += [f"{side}_qty_{k}" for k in range(1, n_levels + 1)]
    return cols


def _require(df: pd.DataFrame, columns, dataset: str):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataQualityError(f"{dataset}: missing raw columns {missing}")


def _to_numeric(df: pd.DataFrame, columns, dataset: str, report: QualityReport) -> pd.DataFrame:
    """Coerce to float; rows with unparsable values (e.g. '2<') are dropped and counted."""
    parsed = df[columns].apply(pd.to_numeric, errors="coerce")
    bad = parsed.isna() & df[columns].notna()
    report.add(dataset, "unparsable numeric values -> row dropped", bad.any(axis=1).sum(), len(df),
               ", ".join(f"{c}={n}" for c, n in bad.sum().items() if n))
    df = df.copy()
    df[columns] = parsed
    return df.loc[~bad.any(axis=1)]


def _session_filter(df: pd.DataFrame, dataset: str, report: QualityReport) -> pd.DataFrame:
    keep = in_session(df["datetime"])
    pre = (~keep) & (df["datetime"].dt.hour < 12)
    report.add(dataset, "outside continuous session (pre-open / auction) -> dropped", pre.sum(), len(df))
    report.add(dataset, "outside continuous session (lunch / post-close) -> dropped",
               ((~keep) & ~pre).sum(), len(df))
    return df.loc[keep]


def _clean_quotes(df: pd.DataFrame, n_levels: int, dataset: str, report: QualityReport) -> pd.DataFrame:
    """Empty depth levels are published as price 0 -> NaN.  A crossed or one-sided
    top of book is not tradable information: blank it and carry the last valid
    quote forward within the same day (never across days)."""
    df = df.copy()
    for side in ("bid", "ask"):
        for k in range(1, n_levels + 1):
            empty = (df[f"{side}_px_{k}"] <= 0) | (df[f"{side}_qty_{k}"] <= 0)
            df.loc[empty, [f"{side}_px_{k}", f"{side}_qty_{k}"]] = np.nan
    bad_top = (df["bid_px_1"].isna() | df["ask_px_1"].isna()
               | (df["bid_px_1"] >= df["ask_px_1"]))
    report.add(dataset, "invalid top of book (missing / crossed) -> forward-filled", bad_top.sum(), len(df))
    book = level_columns(n_levels)
    df.loc[bad_top, book] = np.nan
    df[book] = df.groupby(df["datetime"].dt.date)[book].ffill()
    still = df["bid_px_1"].isna() | df["ask_px_1"].isna()
    report.add(dataset, "no valid quote yet at start of day -> dropped", still.sum(), len(df))
    return df.loc[~still]


def _check_trades(df: pd.DataFrame, multiplier_per_volume: float, dataset: str, report: QualityReport):
    """A snapshot's implied trade price must lie close to its last price."""
    if (df["volume"] < 0).any():
        raise DataQualityError(f"{dataset}: negative per-snapshot volume")
    traded = df["volume"] > 0
    implied = df.loc[traded, "amount"] / (df.loc[traded, "volume"] * multiplier_per_volume)
    off = (implied / df.loc[traded, "last"] - 1).abs() > 0.01
    report.add(dataset, "snapshots with trades", traded.sum(), len(df))
    report.add(dataset, "implied trade price >1% away from last price", off.sum(), traded.sum())


def _finalize(df: pd.DataFrame, n_levels: int) -> pd.DataFrame:
    df = df.sort_values(["datetime", "seq"], kind="stable")
    cols = ["datetime", "seq", "last", "volume", "amount"] + level_columns(n_levels)
    return df[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# IH futures: Chinese column names, per-snapshot volume, second-resolution time
# ---------------------------------------------------------------------------
IH_RAW_COLUMNS = {"时间": "datetime", "最新": "last", "成交量": "volume", "成交额": "amount",
                  "买一价": "bid_px_1", "买一量": "bid_qty_1", "卖一价": "ask_px_1", "卖一量": "ask_qty_1"}


def combine_raw_files(frames: dict[str, pd.DataFrame], time_col: str, dataset: str,
                      report: QualityReport) -> pd.DataFrame:
    """Concatenate per-period vendor files of one contract.

    Monthly export folders can overlap at their boundaries, which would double
    every overlapping trading day.  Each day is taken from exactly one file (the
    one with the most rows for that day).
    """
    parts = []
    for name, f in frames.items():
        f = f.copy()
        f["_source"] = name
        f["_day"] = pd.to_datetime(f[time_col]).dt.normalize()
        parts.append(f)
    df = pd.concat(parts, ignore_index=True)
    rows = df.groupby(["_day", "_source"]).size().rename("n").reset_index()
    best = rows.sort_values(["_day", "n"], ascending=[True, False], kind="stable").drop_duplicates("_day")
    keep = df.set_index(["_day", "_source"]).index.isin(best.set_index(["_day", "_source"]).index)
    report.add(dataset, "rows from days duplicated across raw files -> dropped", (~keep).sum(), len(df))
    return df.loc[keep].drop(columns=["_source", "_day"]).reset_index(drop=True)


def clean_ih_ticks(raw: pd.DataFrame, report: QualityReport, multiplier: float = 300.0) -> pd.DataFrame:
    ds = "IH ticks"
    _require(raw, IH_RAW_COLUMNS, ds)
    df = raw[list(IH_RAW_COLUMNS)].rename(columns=IH_RAW_COLUMNS)
    df["seq"] = np.arange(len(df))
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")
    report.add(ds, "raw rows", len(df))
    report.add(ds, "rows sharing a timestamp with another row (ordered by seq)",
               df["datetime"].duplicated(keep=False).sum(), len(df))

    df = _to_numeric(df, [c for c in IH_RAW_COLUMNS.values() if c != "datetime"], ds, report)
    df = _session_filter(df, ds, report)
    df = _clean_quotes(df, 1, ds, report)
    _check_trades(df, multiplier, ds, report)
    df = _finalize(df, 1)
    report.add(ds, "clean rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 50 ETF: epoch-ms UTC time, cumulative volume, depth as "[p1, p2, ...]" strings
# ---------------------------------------------------------------------------
ETF_RAW_COLUMNS = ["time", "lastPrice", "volume", "amount", "bidPrice", "bidVol", "askPrice", "askVol"]


def _explode_levels(s: pd.Series, n_levels: int, name: str) -> pd.DataFrame:
    parts = s.str.strip("[] ").str.split(",", expand=True)
    if parts.shape[1] != n_levels:
        raise DataQualityError(f"{name}: expected {n_levels} depth levels, got {parts.shape[1]}")
    return parts.astype(float)


def clean_etf_ticks(raw: pd.DataFrame, report: QualityReport, n_levels: int = 5) -> pd.DataFrame:
    ds = "ETF ticks"
    _require(raw, ETF_RAW_COLUMNS, ds)
    df = pd.DataFrame({
        "datetime": pd.to_datetime(raw["time"], unit="ms", utc=True)
                      .dt.tz_convert("Asia/Shanghai").dt.tz_localize(None),
        "seq": np.arange(len(raw)),
        "last": raw["lastPrice"].astype(float),
        "cum_volume": raw["volume"].astype(float),
        "cum_amount": raw["amount"].astype(float),
    })
    for side, px, qty in (("bid", "bidPrice", "bidVol"), ("ask", "askPrice", "askVol")):
        p = _explode_levels(raw[px], n_levels, px)
        q = _explode_levels(raw[qty], n_levels, qty)
        for k in range(n_levels):
            df[f"{side}_px_{k + 1}"] = p[k].to_numpy()
            df[f"{side}_qty_{k + 1}"] = q[k].to_numpy()
    report.add(ds, "raw rows", len(df))

    if not df["datetime"].is_monotonic_increasing:
        raise DataQualityError(f"{ds}: timestamps are not in order")
    dups = df["datetime"].duplicated()
    report.add(ds, "duplicate timestamps -> later row kept", dups.sum(), len(df))
    df = df.loc[~df["datetime"].duplicated(keep="last")]

    # Cumulative -> per-snapshot increments.  Must run BEFORE the session filter:
    # the first continuous-session snapshot is differenced against the opening
    # auction snapshot, so auction volume is not attributed to the 09:31 bar.
    day = df["datetime"].dt.date
    for cum, inc in (("cum_volume", "volume"), ("cum_amount", "amount")):
        d = df.groupby(day)[cum].diff()
        if (d < 0).any():
            raise DataQualityError(f"{ds}: cumulative {inc} decreases intraday "
                                   f"({(d < 0).sum()} snapshots) - vendor reset not handled")
        df[inc] = d.fillna(df[cum])
    df = df.drop(columns=["cum_volume", "cum_amount"])

    df = _session_filter(df, ds, report)
    df = _clean_quotes(df, n_levels, ds, report)
    _check_trades(df, 100.0, ds, report)
    df = _finalize(df, n_levels)
    report.add(ds, "clean rows", len(df))
    return df
