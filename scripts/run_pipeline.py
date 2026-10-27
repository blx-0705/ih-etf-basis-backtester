"""Run the full pipeline on synthetic data (default) or on your own raw files.

    python scripts/make_synthetic_data.py
    python scripts/run_pipeline.py                 # F1: IH passive / ETF aggressive
    python scripts/run_pipeline.py --leader ETF    # F2: ETF passive / IH aggressive
"""
import argparse
import json
from pathlib import Path

from basis_bt.engine import ExecConfig
from basis_bt.pipeline import prepare_data, run_backtest
from basis_bt.signals import ZScoreRatchet

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--work", type=Path, default=ROOT / "data" / "interim")
    p.add_argument("--out", type=Path, default=ROOT / "output")
    p.add_argument("--leader", choices=["IH", "ETF"], default="IH")
    p.add_argument("--passive-price", choices=["mid", "join"], default=None)
    p.add_argument("--latency", type=float, default=1.0)
    p.add_argument("--window", type=int, default=240, help="z-score window in minutes")
    p.add_argument("--capital", type=float, default=20_000_000)
    p.add_argument("--skip-prepare", action="store_true")
    a = p.parse_args()

    if not a.skip_prepare:
        report = prepare_data(sorted(a.raw.glob("IH*.csv")), a.raw / "510050_SH_tick.csv", a.work)
        print(report.to_frame().to_string(index=False))

    cfg = ExecConfig(leader=a.leader, latency_s=a.latency,
                     passive_price=a.passive_price or ("mid" if a.leader == "IH" else "join"))
    summary = run_backtest(a.work, a.out / f"leader_{a.leader}", cfg,
                           ZScoreRatchet(capital=a.capital, window=a.window), a.capital)
    print(json.dumps({k: summary[k] for k in ("performance", "execution", "hedge_units", "round_trips")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
