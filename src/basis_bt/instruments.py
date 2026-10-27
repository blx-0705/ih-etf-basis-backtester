"""Contract specifications for the two legs of the basis trade.

Units used throughout the package
---------------------------------
* **base unit**  – what a position is counted in: IH contracts, ETF shares.
* **volume unit** – what the exchange feed reports volume / queue size in:
  IH contracts, ETF board lots (1 lot = 100 shares).
* **price**      – IH in index points, ETF in CNY per share.

``amount / (volume * volume_unit * multiplier)`` is therefore a price for both
instruments, which is the single formula every VWAP in the package goes through.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str
    multiplier: float      # cash per 1.0 price move per base unit
    volume_unit: float     # base units per reported volume unit
    tick_size: float
    fee_rate: float        # proportional commission on notional
    min_fill: float        # smallest tradable quantity, in base units

    def vwap(self, amount, volume):
        """Trade VWAP of a tick (scalar or array) from its amount and volume."""
        return amount / (volume * self.volume_unit * self.multiplier)

    def notional(self, price, qty):
        return price * qty * self.multiplier


# SSE 50 index futures (CFFEX): 300 CNY per index point, tick 0.2
IH = Instrument(symbol="IH", multiplier=300.0, volume_unit=1.0, tick_size=0.2,
                fee_rate=0.3e-4, min_fill=1.0)

# SSE 50 ETF (510050.SH): volume reported in board lots of 100 shares, tick 0.001
ETF = Instrument(symbol="ETF", multiplier=1.0, volume_unit=100.0, tick_size=0.001,
                 fee_rate=0.5e-4, min_fill=100.0)

# One hedge unit: 1 IH contract against the ETF shares of equal notional.
# IH notional = F * 300; ETF notional = S * 300_000; F ~= 1000 * S.
ETF_SHARES_PER_IH = 300_000
BASIS_SCALE = ETF_SHARES_PER_IH / IH.multiplier     # = 1000, ETF price -> index points


def basis(ih_price, etf_price):
    """Futures-minus-spot basis in index points."""
    return ih_price - etf_price * BASIS_SCALE


INSTRUMENTS = {"IH": IH, "ETF": ETF}
UNIT_SIZE = {"IH": 1.0, "ETF": float(ETF_SHARES_PER_IH)}   # base units per hedge unit
