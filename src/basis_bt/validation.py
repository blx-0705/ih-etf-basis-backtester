"""Cross-validation of locally built bars against a vendor's official 1-minute bars.

The vendor file needs ``datetime, open, high, low, close, volume`` with the same
end-of-bar labels; volume in the same unit as the local bars.  Proprietary
vendor SDK code and credentials are deliberately not part of this repository.
"""
from __future__ import annotations

import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")


def compare_with_vendor(local: pd.DataFrame, vendor: pd.DataFrame, price_tol: float,
                        volume_tol: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Field-by-field agreement on the common minutes.

    Returns (summary per field, mismatching rows) so disagreements can be
    inspected rather than just counted.
    """
    v = vendor.set_index("datetime")[list(FIELDS)]
    common = local.index.intersection(v.index)
    joined = local.loc[common, list(FIELDS)].join(v.loc[common], rsuffix="_vendor")
    rows, bad = [], pd.Series(False, index=common)
    for f in FIELDS:
        tol = volume_tol if f == "volume" else price_tol
        diff = (joined[f] - joined[f"{f}_vendor"]).abs()
        miss = diff > tol + 1e-9
        bad |= miss
        rows.append({"field": f, "minutes": len(common), "match_rate": 1 - miss.mean(),
                     "max_abs_diff": diff.max(), "mean_abs_diff": diff.mean()})
    by_time = joined.loc[bad].assign(time_of_day=lambda d: d.index.strftime("%H:%M"))
    return pd.DataFrame(rows), by_time
