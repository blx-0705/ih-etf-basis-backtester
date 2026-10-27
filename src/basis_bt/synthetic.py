"""Synthetic raw ticks in the exact vendor layouts the pipeline consumes.

The real market data used during the internship is licensed and is not
redistributed.  This generator reproduces the *formats and defects* the
cleaning layer has to handle, on top of a simple latent model (GBM index level,
Ornstein-Uhlenbeck basis), so every stage runs end to end:

IH futures (Chinese headers, two snapshots per second, stamps truncated to 1s)
  * opening-auction snapshot at 09:29:00 and post-close junk snapshots ~15:2x
  * corrupted depth values such as ``'2<'``
  * a feed gap of ~2.5 minutes on one day (exercises grid filling and stale flags)
  * export split into two files that overlap by one trading day

50 ETF (epoch-ms UTC, cumulative volume/amount, depth as ``"[p1, ..., p5]"``)
  * pre-open snapshots with zero prices, opening auction at 09:25
  * session-closing snapshots stamped 2s after 11:30 / 15:00 carrying volume
  * empty depth levels published as 0 and one crossed quote
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

IH_TICK, ETF_TICK = 0.2, 0.001


def _session_times(day: pd.Timestamp, step_s: float, start: str, end: str) -> pd.DatetimeIndex:
    t0, t1 = day + pd.Timedelta(start), day + pd.Timedelta(end)
    n = int(round((t1 - t0).total_seconds() / step_s)) + 1
    return t0 + pd.to_timedelta(np.arange(n) * step_s, unit="s")


def _latent(rng, times: pd.DatetimeIndex, s0: float, b0: float, vol_daily=0.012,
            basis_mean=-8.0, basis_sd=3.0, half_life_min=25.0):
    """Index level S (GBM) and basis b (OU) sampled at ``times`` (seconds apart)."""
    dt = np.diff(times.values).astype("timedelta64[ms]").astype(float) / 1000.0
    dt = np.minimum(np.concatenate([[0.0], dt]), 2.0)         # lunch gap: no extra diffusion
    sig_s = vol_daily / np.sqrt(4 * 3600)
    s = s0 * np.exp(np.cumsum(sig_s * np.sqrt(dt) * rng.standard_normal(len(dt))))
    theta = np.log(2) / (half_life_min * 60)
    sig_b = basis_sd * np.sqrt(2 * theta)
    b = np.empty(len(dt))
    b[0] = b0
    eps = rng.standard_normal(len(dt))
    for k in range(1, len(dt)):
        b[k] = b[k - 1] + theta * (basis_mean - b[k - 1]) * dt[k] + sig_b * np.sqrt(dt[k]) * eps[k]
    return s, b


def _ih_day(rng, day, s0, b0):
    times = _session_times(day, 0.5, "09:30:00", "11:30:00").append(
        _session_times(day, 0.5, "13:00:00", "15:00:00"))
    s, b = _latent(rng, times, s0, b0)
    fair = s + b
    bid = np.floor(fair / IH_TICK)
    ask = bid + np.where(rng.random(len(times)) < 0.8, 1, 2)
    vol = rng.poisson(0.9, len(times)) * (rng.random(len(times)) < 0.45)
    buyer = rng.random(len(times)) < 0.5
    trade_px = np.where(buyer, ask, bid) * IH_TICK
    last = pd.Series(np.where(vol > 0, trade_px, np.nan)).ffill().bfill().to_numpy()
    df = pd.DataFrame({
        "dt": times, "最新": last.round(1), "成交量": vol.astype(float),
        "成交额": (trade_px * vol * 300).round(1),
        "买一价": (bid * IH_TICK).round(1), "卖一价": (ask * IH_TICK).round(1),
        "买一量": rng.integers(1, 12, len(times)), "卖一量": rng.integers(1, 12, len(times)),
    })
    auction = pd.DataFrame({"dt": [day + pd.Timedelta("09:29:00")], "最新": [round(fair[0] / IH_TICK) * IH_TICK],
                            "成交量": [12.0], "买一价": [bid[0] * IH_TICK], "卖一价": [ask[0] * IH_TICK],
                            "买一量": [3], "卖一量": [2]})
    auction["成交额"] = auction["最新"] * 12 * 300
    junk_t = day + pd.to_timedelta(["15:23:52", "15:24:04", "15:25:07"])
    junk = pd.DataFrame({"dt": junk_t, "最新": last[-1], "成交量": 0.0, "成交额": 0.0,
                         "买一价": bid[-1] * IH_TICK, "卖一价": ask[-1] * IH_TICK, "买一量": 1, "卖一量": 1})
    df = pd.concat([auction, df, junk], ignore_index=True)
    return df, times, s, b[-1]


def _ih_format(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "合约代码": "IH2509",
        "时间数": (df["dt"] - pd.Timestamp("1899-12-30")) / pd.Timedelta(days=1),
        "时间": df["dt"].dt.strftime("%Y-%m-%d %H:%M:%S"),       # vendor truncates to the second
        "最新": df["最新"], "成交量": df["成交量"], "成交额": df["成交额"],
        "持仓": 0, "增仓": 0.0, "买卖标识": 0,
        "买一价": df["买一价"].round(1), "卖一价": df["卖一价"].round(1),
        "买一量": df["买一量"].astype(str), "卖一量": df["卖一量"].astype(str),
        "最高价": df["最新"], "最低价": df["最新"],
    })
    return out


def _levels_str(mat: np.ndarray, fmt: str) -> list[str]:
    return ["[" + ", ".join(fmt.format(v) for v in row) + "]" for row in mat]


def _etf_day(rng, day, s_open, s_path_times, s_path):
    pre = _session_times(day, 3, "09:15:00", "09:24:57")
    cont = _session_times(day, 3, "09:30:00", "11:30:00").append(_session_times(day, 3, "13:00:00", "15:00:00"))
    final = day + pd.to_timedelta(["11:30:02", "15:00:02"])
    times = pre.append(pd.DatetimeIndex([day + pd.Timedelta("09:25:00")])).append(cont).append(final).sort_values()
    # spot tracks the same latent index level as the future
    idx = np.clip(np.searchsorted(s_path_times.values, times.values, side="right") - 1, 0, len(s_path) - 1)
    s = np.where(times < day + pd.Timedelta("09:30:00"), s_open, s_path[idx])
    fair = s / 1000.0
    bid1 = np.floor(fair / ETF_TICK)
    ask1 = bid1 + np.where(rng.random(len(times)) < 0.95, 1, 2)
    k = np.arange(5)
    bid_px = (bid1[:, None] - k) * ETF_TICK
    ask_px = (ask1[:, None] + k) * ETF_TICK
    bid_q = rng.integers(300, 6000, (len(times), 5)).astype(float)
    ask_q = rng.integers(300, 6000, (len(times), 5)).astype(float)
    thin = rng.random(len(times)) < 0.005                       # empty deep levels published as 0
    bid_px[thin, 3:] = 0.0
    bid_q[thin, 3:] = 0.0

    is_pre = times < day + pd.Timedelta("09:25:00")
    is_auction = times == day + pd.Timedelta("09:25:00")
    vol = np.round(rng.lognormal(np.log(900), 0.8, len(times)))
    vol[is_pre] = 0
    vol[is_auction] = 25_000
    share_buy = rng.random(len(times))
    amount = vol * 100 * (share_buy * ask1 + (1 - share_buy) * bid1) * ETF_TICK
    last = np.where(share_buy > 0.5, ask1, bid1) * ETF_TICK
    last[is_pre] = 0.0
    bid_px[is_pre, 1:], ask_px[is_pre, 1:] = 0.0, 0.0           # indicative price only before the auction

    crossed = np.flatnonzero(~is_pre & ~is_auction)[len(times) // 3]
    bid_px[crossed, 0] = ask_px[crossed, 0] + ETF_TICK

    ms = times.tz_localize("Asia/Shanghai").tz_convert("UTC").as_unit("ms").asi8
    return pd.DataFrame({
        "time": ms, "lastPrice": last.round(3), "open": 0.0, "high": 0.0, "low": 0.0, "lastClose": 0.0,
        "amount": np.cumsum(amount).round(1), "volume": np.cumsum(vol).astype(np.int64),
        "pvolume": 0, "tickvol": 0, "stockStatus": 0, "openInt": 12, "lastSettlementPrice": 0.0,
        "askPrice": _levels_str(ask_px, "{:.3f}"), "bidPrice": _levels_str(bid_px, "{:.3f}"),
        "askVol": _levels_str(ask_q, "{:.0f}"), "bidVol": _levels_str(bid_q, "{:.0f}"),
        "settlementPrice": 0.0, "transactionNum": 0, "pe": 0.0, " stock_code": "510050.SH",
    })


def generate(out_dir: str | Path, n_days: int = 6, start: str = "2025-03-03", seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    days = pd.bdate_range(start, periods=n_days)
    s, b = 2700.0, -8.0
    ih_days, etf_days = [], []
    for d, day in enumerate(days):
        s_open = s * np.exp(0.004 * rng.standard_normal())       # overnight gap
        ih, times, s_path, b = _ih_day(rng, day, s_open, b)
        s = s_path[-1]
        if d == 1:                                               # feed outage of ~2.5 minutes
            gap = (ih["dt"] >= day + pd.Timedelta("10:15:10")) & (ih["dt"] < day + pd.Timedelta("10:17:40"))
            ih = ih.loc[~gap]
        etf_days.append(_etf_day(rng, day, s_open, times, s_path))   # spot sees the same index level
        ih_days.append(_ih_format(ih))

    ih_all = [f.reset_index(drop=True) for f in ih_days]
    ih_all[0].loc[500, "卖一量"] = "2<"
    ih_all[-1].loc[900, "卖一量"] = "2p"
    split = n_days // 2
    paths = {
        "ih_part1": out / "IH2509_part1.csv",
        "ih_part2": out / "IH2509_part2.csv",
        "etf": out / "510050_SH_tick.csv",
    }
    pd.concat(ih_all[: split + 1]).to_csv(paths["ih_part1"], index=False, encoding="utf-8")
    pd.concat(ih_all[split:]).to_csv(paths["ih_part2"], index=False, encoding="utf-8")   # overlaps one day
    pd.concat(etf_days).to_csv(paths["etf"], index=False, encoding="utf-8-sig")
    return paths
