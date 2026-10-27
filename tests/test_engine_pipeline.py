import json

import pandas as pd
import pytest

from basis_bt.engine import ExecConfig, plan_legs
from basis_bt.pipeline import prepare_data, run_backtest
from basis_bt.signals import ZScoreRatchet


def test_plan_legs_rounds_fractional_etf_towards_target():
    # short 4 IH, long 3.5 ETF units, target 2 -> IH back to -3, sell 0.5 ETF, then one unit
    assert plan_legs(-4, 3.5, -2, leader="IH") == [("IH", 1, "align"), ("ETF", -0.5, "round"), ("PAIR", 1, "pair")]
    # same state, leader ETF -> the passive leg goes first
    assert plan_legs(-4, 3.5, -2, leader="ETF")[0] == ("ETF", -0.5, "round")
    # increasing the short: the half unit is completed, not unwound
    assert plan_legs(-4, 3.5, -6, leader="IH") == [("ETF", 0.5, "round"), ("PAIR", -2, "pair")]
    assert plan_legs(-2, 2.0, -2, leader="IH") == []


@pytest.mark.parametrize("leader", ["IH", "ETF"])
def test_end_to_end_on_synthetic_data(synthetic_raw, tmp_path, leader):
    work = tmp_path / "work"
    prepare_data([synthetic_raw["ih_part1"], synthetic_raw["ih_part2"]], synthetic_raw["etf"], work)
    cfg = ExecConfig(leader=leader, passive_price="mid" if leader == "IH" else "join")
    summary = run_backtest(work, tmp_path / "out", cfg, ZScoreRatchet(window=120), 20_000_000)
    # every invariant (attribution closure, FIFO conservation, P&L reconciliation,
    # hedge imbalance) is asserted inside the run; here check it actually traded
    assert summary["hedge_units"].get("COMPLETE", 0) > 0
    assert summary["execution"]["max_hedge_imbalance_units"] <= 1.0 + 1e-9
    fills = pd.read_csv(tmp_path / "out" / "fills.csv", parse_dates=["time"])
    decision = pd.to_datetime(fills["decision_time"])
    assert (fills["time"] >= decision + pd.Timedelta(seconds=1)).all(), "fill before signal + latency"
    json.loads((tmp_path / "out" / "summary.json").read_text())
