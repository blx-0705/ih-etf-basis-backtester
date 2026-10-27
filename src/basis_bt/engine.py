"""Minute-driven, tick-executed pair backtester.

Each minute ``t`` the engine reads the target from the signal (computed on the
bar ending at ``t``) and works it on ticks in ``[t + latency, t + 1min)``:

1. ``plan_legs`` derives the trades from the **actual** holdings, never from the
   previous target, so partial fills simply carry over to the next minute.
2. Legs run in order: single-instrument legs that restore a clean integer
   hedge (leader instrument first), then whole hedge units (``PAIR`` legs), in
   which the leader is worked passively and every completed leader unit is
   immediately hedged aggressively on the follower.
3. Any leg that cannot complete inside the minute ends the minute.

``leader='IH'`` reproduces framework F1 (futures quoted passively, ETF taken);
``leader='ETF'`` reproduces F2.  Every fill is tagged with the hedge unit
(``pair_id``) it belongs to, which is what the slippage attribution consumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

from .execution import BUY, SELL, AggressiveExecutor, MarketTape, PassiveQueueExecutor
from .instruments import INSTRUMENTS, UNIT_SIZE
from .sessions import TradingClock

EPS = 1e-9
ORDER_CLIP = {"IH": 1.0, "ETF": UNIT_SIZE["ETF"]}   # IH is worked one contract per order


@dataclass
class ExecConfig:
    leader: str = "IH"
    passive_price: str = "mid"        # "mid" (snapped to grid) or "join"
    latency_s: float = 1.0
    take_ratio: float = 0.7
    participation: float = 0.1
    trace: bool = False

    @property
    def follower(self) -> str:
        return "ETF" if self.leader == "IH" else "IH"


def plan_legs(ih_pos: float, etf_units: float, target_ih: int, leader: str) -> list[tuple[str, float, str]]:
    """Ordered legs ``(symbol, signed qty, stage)``; qty in IH contracts / ETF hedge units.

    A fractional ETF holding is first rounded *towards the target* so it is never
    traded away and bought back; IH is aligned to that integer; the remainder of
    the target is traded as whole hedge units.
    """
    target_etf = -target_ih
    if target_etf > etf_units + EPS:
        rounded = math.ceil(etf_units - EPS)
    elif target_etf < etf_units - EPS:
        rounded = math.floor(etf_units + EPS)
    else:
        rounded = round(etf_units)
    legs = []
    if ih_pos != -rounded:
        legs.append(("IH", -rounded - ih_pos, "align"))
    if abs(rounded - etf_units) > EPS:
        legs.append(("ETF", rounded - etf_units, "round"))
    legs.sort(key=lambda leg: leg[0] != leader)
    if target_ih != -rounded:
        legs.append(("PAIR", target_ih + rounded, "pair"))
    return legs


@dataclass
class PendingPair:
    pair_id: int
    side: dict
    need: dict
    record: dict


@dataclass
class BacktestResult:
    fills: pd.DataFrame
    order_events: pd.DataFrame
    pairs: pd.DataFrame
    minute_state: pd.DataFrame
    trace: pd.DataFrame | None
    config: ExecConfig = field(repr=False, default=None)


class PairBacktester:
    def __init__(self, cfg: ExecConfig):
        self.cfg = cfg
        self.pos = {"IH": 0.0, "ETF": 0.0}
        self.fills, self.events, self.pairs, self.state = [], [], [], []
        self.trace = [] if cfg.trace else None
        self.pending: PendingPair | None = None
        self._pair_seq = 0

    # ------------------------------------------------------------------ fills
    def _book(self, fills: list, stage: str, decision_time):
        for f in fills:
            sym = f["symbol"]
            self.pos[sym] += f["side"] * f["qty"]
            f.update(stage=stage, decision_time=decision_time, pair_id=-1)
            p = self.pending
            if p is not None:
                if f["side"] == p.side[sym] and p.need[sym] > EPS:
                    assert f["qty"] <= p.need[sym] + EPS, "fill overshoots its hedge unit"
                    p.need[sym] -= f["qty"]
                    f["pair_id"] = p.pair_id
                elif f["side"] != p.side[sym]:
                    p.record.update(status="BROKEN", end_time=f["time"])
                    self.pending = None
                if self.pending is not None and max(p.need.values()) <= EPS:
                    p.record.update(status="COMPLETE", end_time=f["time"])
                    self.pending = None
            self.fills.append(f)

    def _trade(self, sym: str, side: int, qty: float, end, stage: str, decision_time) -> bool:
        """Work ``qty`` base units of ``sym``; True if fully done inside the window."""
        remaining = qty
        while remaining > EPS:
            clip = min(remaining, ORDER_CLIP[sym])
            res = self.exec[sym].execute(side, clip, self.cursor, end, decision_time)
            self._book(res.fills, stage, decision_time)
            remaining -= res.filled
            if res.filled < clip - EPS:
                return False
            self.cursor = res.last_time + self.latency
        return True

    # ----------------------------------------------------------------- minute
    def _run_minute(self, t: pd.Timestamp, target: int, decision_time, anchors: dict):
        end = t + pd.Timedelta(minutes=1)
        self.cursor = max(self.cursor, t + self.latency)
        lead, follow = self.cfg.leader, self.cfg.follower
        for sym, qty, stage in plan_legs(self.pos["IH"], self.pos["ETF"] / UNIT_SIZE["ETF"], target, lead):
            if sym != "PAIR":
                if not self._trade(sym, BUY if qty > 0 else SELL, abs(qty) * UNIT_SIZE[sym], end,
                                   stage, decision_time):
                    return
                continue
            ih_side = BUY if qty > 0 else SELL
            side = {"IH": ih_side, "ETF": -ih_side}
            for _ in range(int(round(abs(qty)))):
                assert self.pending is None, "a new hedge unit started before the previous one completed"
                self._pair_seq += 1
                record = {"pair_id": self._pair_seq, "direction": "OPEN" if ih_side == SELL else "CLOSE",
                          "decision_time": decision_time, "start_time": max(self.cursor, t),
                          "status": "OPEN_AT_END", "end_time": pd.NaT, **anchors}
                self.pairs.append(record)
                self.pending = PendingPair(self._pair_seq, side, dict(UNIT_SIZE), record)
                if not self._trade(lead, side[lead], UNIT_SIZE[lead], end, stage, decision_time):
                    if self.pending.need == UNIT_SIZE:      # nothing traded: the unit never started
                        self.pairs.pop()
                        self.pending = None
                        self._pair_seq -= 1
                    return
                if not self._trade(follow, side[follow], UNIT_SIZE[follow], end, stage, decision_time):
                    return

    # -------------------------------------------------------------------- run
    def run(self, signal: pd.DataFrame, panel: pd.DataFrame, load_books) -> BacktestResult:
        """``load_books(symbol, day) -> DataFrame`` returns that day's book tape."""
        cfg = self.cfg
        self.latency = pd.Timedelta(seconds=cfg.latency_s)
        clock = TradingClock(panel.index)
        for day, sig_day in signal.groupby(signal.index.normalize()):
            tapes = {s: MarketTape(INSTRUMENTS[s], load_books(s, day)) for s in ("IH", "ETF")}
            lead, follow = cfg.leader, cfg.follower
            passive = PassiveQueueExecutor(tapes[lead], tapes[follow], cfg.passive_price,
                                           self.events, self.trace, clock)
            self.exec = {lead: passive,
                         follow: AggressiveExecutor(tapes[follow], tapes[lead], cfg.take_ratio, cfg.participation)}
            self.cursor = day
            prev_target, decision_time = None, None
            pan_day = panel.loc[sig_day.index]
            for t, target in sig_day["target_ih_qty"].items():
                if target != prev_target:          # a new decision cancels the resting order
                    decision_time = t
                    passive.cancel(t, "CANCELLED")
                    prev_target = target
                anchors = {"t0_ih": pan_day.at[decision_time, "ih_mid"],
                           "t0_etf": pan_day.at[decision_time, "etf_mid"]}
                self._run_minute(t, int(target), decision_time, anchors)
                imbalance = self.pos["IH"] + self.pos["ETF"] / UNIT_SIZE["ETF"]
                assert abs(imbalance) < 1 + 1e-6, f"hedge imbalance {imbalance:.4f} units at {t}"
                self.state.append({"window_start": t, "target_ih_qty": target, "ih_pos": self.pos["IH"],
                                   "etf_pos": self.pos["ETF"], "imbalance_units": imbalance})
            passive.cancel(tapes[lead].time[-1], "EOD")

        fills = pd.DataFrame(self.fills)
        return BacktestResult(fills=fills, order_events=pd.DataFrame(self.events),
                              pairs=pd.DataFrame(self.pairs), minute_state=pd.DataFrame(self.state),
                              trace=None if self.trace is None else pd.DataFrame(self.trace), config=cfg)
