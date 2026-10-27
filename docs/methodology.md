# Methodology

## 1. Units and the hedge unit

| | IH future | 50 ETF |
|---|---|---|
| position unit | contract | share |
| reported volume / queue unit | contract | board lot (100 shares) |
| price | index points | CNY / share |
| multiplier | 300 CNY / point | 1 |

One **hedge unit** = 1 IH contract against 300,000 ETF shares. Equal notional needs `F ≈ 1000 · S`, so the basis in
index points is `basis = F_mid − 1000 · S_mid`. All VWAPs go through one formula,
`amount / (volume · volume_unit · multiplier)`. A unit mismatch in this formula is what broke the prototype's ETF
order book.

## 2. Time conventions

- Bars are **end-stamped**: bar `t` covers `[t−1min, t)`. The closing snapshots (published up to ~3 s after
  11:30 / 15:00) fold into the last bar of their session.
- A signal on bar `t` can trade from `t + latency` to `t + 1min`. Trades printed on the tick where an order is
  placed are treated as having happened before the order arrived.
- Positions reported on bar `t` include every fill stamped before `t`. This matches the bars, so a fill is never
  compared against the bar before the one it traded in.
- The tick as-of join is backward with exact matches allowed. ETF stamps are exact seconds and IH stamps are
  truncated, so an ETF snapshot with an equal stamp is never later than the IH snapshot. A tolerance prevents
  matches across the lunch break.

## 3. Order-book reconstruction

The feed shows only the top N levels (IH: 1, ETF: 5). For every snapshot:

1. **consume**: the snapshot's traded volume removes resting size on levels its VWAP reached, starting from the
   best price;
2. **refresh**: levels better than the newly published best are dropped (traded through or cancelled). Published
   levels overwrite quantities. Deeper levels no longer shown are kept as the best estimate, capped at 10 per side.

Prices are integer ticks, so level matching is exact.

## 4. Fill models

**Passive** (the *leader* instrument). When placed, `queue_ahead` is set to the size shown at the limit price. On
each later tick:

```
if tick VWAP reached my price:
    ahead       = min(queue_ahead, volume)      # orders in front of me trade first
    queue_ahead -= ahead
    fill        = floor_to_lot(min(volume − ahead, remaining))
queue_ahead = min(queue_ahead, size still shown at my price)    # cancellations
              (0 if my price is now at or better than the best and the level is gone)
```

A resting order keeps its place in the queue across minutes while side, price and decision are unchanged and its
size is not increased. Any other change cancels it, and the replacement joins the back of the queue. Every order
emits a lifecycle event (FILLED / REPRICED / CANCELLED / EOD) with its duration in trading seconds.

**Aggressive** (the *follower*). Each tick fills up to `max(0.7 · opposite L1 size, 0.1 · tick volume)` at the
opposite best price, rounded down to whole lots.

## 5. Minute loop

```
for each minute t:
    if target changed: new decision, cancel the resting passive order
    legs = plan_legs(actual IH, actual ETF units, target)
        1. round the fractional ETF holding towards the target
        2. align IH to that integer            (single legs, leader instrument first)
        3. remaining change as whole hedge units:
           leader passive -> +latency -> follower aggressive -> +latency -> next unit
    stop the minute at the first leg that cannot finish; it carries over via actual holdings
```

A hedge unit lives in a `PendingPair` until both legs reach unit size. A fill in the opposite direction on either
leg marks it `BROKEN`. A unit with no fills is discarded. The engine asserts that a new unit never starts while
another is pending, and that hedge imbalance stays below one unit.

## 6. Attribution

For each complete hedge unit, with `s = +1` when opening (selling the basis) and `−1` when closing:

| anchor | IH | ETF |
|---|---|---|
| t0 signal | mid on decision bar | mid on decision bar |
| t1f first quote | leader limit / follower mid at the first placement | |
| t1w weighted quote | the same, weighted by fill size over the orders that filled | |
| t2 fill tick | mids on the leader's fill ticks (weighted by fill size) | |
| t3 execution | average execution prices | |

Segment cost on each leg: `s·(IH_a − IH_b)` for IH and `s·1000·(ETF_b − ETF_a)` for the ETF. Positive values are
costs. Delay = t0→t1f, reprice = t1f→t1w, queue = t1w→t2, chase = t2→t3.

The checks below hold by construction and are asserted:
- `Σ segments = s · (basis_t0 − basis_t3)`
- leader `t3 = t1w` (passive price lock): the leader's queue and chase costs cancel on its own leg
- in a FIFO round trip, `signal_pnl − net_pnl = 300 · Σ segment costs + fees`

## 7. P&L and reconciliation

Daily P&L = holding P&L (yesterday's position × change in close) + trading P&L (fills marked to the close, minus
fees). It is checked independently against cash accounting: equity = initial capital + cumulative cash flows +
positions × close.
