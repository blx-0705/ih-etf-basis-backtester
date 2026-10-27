"""Signal layer: minute panel -> target futures position.

The execution engine only consumes ``target_ih_qty`` (IH contracts, <= 0: short
futures / long ETF).  Any research signal can be plugged in as long as it passes
``validate_signal``.  ``ZScoreRatchet`` is a deliberately simple placeholder used
to exercise the execution stack; it is not a tuned strategy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .instruments import ETF_SHARES_PER_IH, IH


@dataclass
class ZScoreRatchet:
    """Short the basis when it is rich, unwind as it normalizes.

    The z-score of the mid basis over a rolling window is mapped to a target
    exposure in [-1, 0] with hysteresis: in the open zone (z > open_z) the
    position may only get deeper, in the close zone (z < -close_z) only
    shallower, and in between it is frozen, so noise around a threshold does not
    churn the position.
    """
    capital: float = 20_000_000
    window: int = 600
    open_z: float = 0.5
    full_open_z: float = 2.0
    close_z: float = 0.5
    full_close_z: float = 1.5
    curvature: float = 4.0
    margin_ratio: float = 0.15
    use_ratio: float = 0.95

    def _curve(self, x: float) -> float:
        a = self.curvature
        return (1 - math.exp(-a * x)) / (1 - math.exp(-a))

    def _step(self, z: float, pos: float) -> float:
        if z > self.open_z:
            x = min((z - self.open_z) / (self.full_open_z - self.open_z), 1.0)
            return min(pos, -self._curve(x))
        if z < -self.close_z:
            x = min((-z - self.close_z) / (self.full_close_z - self.close_z), 1.0)
            return max(pos, -(1 - self._curve(x)))
        return pos

    def max_units(self, ih_price: float, etf_price: float) -> int:
        unit_cost = etf_price * ETF_SHARES_PER_IH + ih_price * IH.multiplier * self.margin_ratio
        return int(self.capital * self.use_ratio // unit_cost)

    def generate(self, panel: pd.DataFrame) -> pd.DataFrame:
        basis = panel["basis"].where(~panel["stale"])
        mu = basis.rolling(self.window, min_periods=self.window).mean()
        sd = basis.rolling(self.window, min_periods=self.window).std()
        z = (basis - mu) / sd.where(sd > 0)

        pos, path = 0.0, []
        for zi in z.to_numpy():
            if not np.isnan(zi):
                pos = self._step(zi, pos)
            path.append(0.0 if abs(pos) < 1e-4 else pos)

        first = panel.loc[~panel["stale"]].iloc[0]
        units = self.max_units(first["ih_mid"], first["etf_mid"])
        out = pd.DataFrame({"basis": panel["basis"], "z": z, "target_pct": path}, index=panel.index)
        out["target_ih_qty"] = (out["target_pct"] * units).round().astype(int)
        out.attrs["max_units"] = units
        return out


def validate_signal(signal: pd.DataFrame, panel: pd.DataFrame):
    """Contract between any signal and the execution engine."""
    assert signal.index.equals(panel.index), "signal must be defined on every panel minute"
    q = signal["target_ih_qty"]
    assert q.dtype.kind == "i", "target_ih_qty must be integer contracts"
    assert (q <= 0).all(), "convention: IH leg is short (<= 0), ETF leg long"
