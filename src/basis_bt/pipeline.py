"""End-to-end run: raw ticks -> clean ticks -> bars -> panel -> books -> signal
-> tick-replay execution -> attribution -> P&L.  Every stage writes its output
so each one can be inspected or re-run in isolation."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from .align import build_minute_panel
from .attribution import fifo_round_trips, hedge_units, summarize_costs
from .bars import make_bars
from .cleaning import QualityReport, clean_etf_ticks, clean_ih_ticks, combine_raw_files
from .engine import ExecConfig, PairBacktester
from .instruments import ETF, IH
from .orderbook import build_books
from .performance import daily_pnl, execution_quality, metrics, minute_positions
from .sessions import TradingClock
from .signals import ZScoreRatchet, validate_signal

BOOK_LEVELS = {"IH": 1, "ETF": 5}


def prepare_data(ih_raw_files: list[Path], etf_raw_file: Path, work: Path) -> QualityReport:
    """Stages 1-3: clean ticks, build bars + minute panel, reconstruct daily books."""
    work.mkdir(parents=True, exist_ok=True)
    report = QualityReport()
    ih_raw = combine_raw_files({p.name: pd.read_csv(p, dtype=str) for p in ih_raw_files}, "时间", "IH raw", report)
    ticks = {"IH": clean_ih_ticks(ih_raw, report),
             "ETF": clean_etf_ticks(pd.read_csv(etf_raw_file, dtype={c: str for c in
                                    ("askPrice", "bidPrice", "askVol", "bidVol")}), report)}
    bars = {"IH": make_bars(ticks["IH"], IH, report), "ETF": make_bars(ticks["ETF"], ETF, report)}
    panel = build_minute_panel(bars["IH"], bars["ETF"], report)
    panel.to_parquet(work / "panel.parquet")

    days = set(panel.index.normalize())
    for sym, inst in (("IH", IH), ("ETF", ETF)):
        bars[sym].to_parquet(work / f"bars_{sym}.parquet")
        (work / "books" / sym).mkdir(parents=True, exist_ok=True)
        for day, g in ticks[sym].groupby(ticks[sym]["datetime"].dt.normalize()):
            if day in days:
                build_books(g, inst, BOOK_LEVELS[sym]).to_parquet(
                    work / "books" / sym / f"{day:%Y-%m-%d}.parquet")
    report.to_frame().to_csv(work / "data_quality_report.csv", index=False)
    return report


def run_backtest(work: Path, out: Path, cfg: ExecConfig, signal_model: ZScoreRatchet,
                 initial_capital: float) -> dict:
    """Stages 4-6: signal, execution, attribution and P&L."""
    out.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(work / "panel.parquet")
    signal = signal_model.generate(panel)
    validate_signal(signal, panel)

    load_books = lambda sym, day: pd.read_parquet(work / "books" / sym / f"{day:%Y-%m-%d}.parquet")
    res = PairBacktester(cfg).run(signal, panel, load_books)

    clock = TradingClock(panel.index)
    units = hedge_units(res.fills, res.pairs, cfg.leader, clock)
    trips, still_open = fifo_round_trips(units)
    minute_pos = minute_positions(res.fills, signal)
    daily = daily_pnl(res.fills, panel, initial_capital)

    summary = {
        "config": asdict(cfg), "signal": asdict(signal_model) | {"max_units": signal.attrs["max_units"]},
        "performance": metrics(daily, initial_capital),
        "execution": execution_quality(minute_pos, res.order_events),
        "hedge_units": res.pairs["status"].value_counts().to_dict(),
        "round_trips": len(trips), "open_units_at_end": len(still_open),
        "mean_cost_per_round_trip_cny": trips[[c for c in trips if c.endswith("_cost")] + ["fees"]].mean().to_dict()
        if len(trips) else {},
    }
    signal.to_csv(out / "signal.csv")
    res.fills.to_csv(out / "fills.csv", index=False)
    res.order_events.to_csv(out / "passive_order_events.csv", index=False)
    res.pairs.to_csv(out / "hedge_units_status.csv", index=False)
    units.to_csv(out / "hedge_unit_attribution.csv", index=False)
    summarize_costs(units).to_csv(out / "cost_summary_points.csv")
    trips.to_csv(out / "round_trips_fifo.csv", index=False)
    minute_pos.to_csv(out / "minute_positions.csv")
    daily.to_csv(out / "daily_pnl.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
