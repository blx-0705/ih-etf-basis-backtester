# Data-quality findings on the real data set

These are aggregate diagnostics from running the pipeline on the licensed data used during the internship:
IH2509 snapshots for 164 trading days and 510050 ETF snapshots for 239 trading days, with 140 days in common. The
raw data is not redistributed. Only counts and rates are reported here.

## Tick level

| finding | count |
|---|---|
| IH snapshots sharing a timestamp with another snapshot (2 per second, stamps truncated) | 2,268,446 of 2,884,740 |
| IH opening-auction snapshots (09:29:00) | 164 |
| IH junk snapshots after the close (15:2x) | 164 |
| IH corrupted depth values (`'2<'`, `'2p'`) | 2 |
| ETF pre-open / auction snapshots (09:15–09:25, zero prices) | 49,023 |
| ETF snapshots stamped 11:30:01–03 / 15:00:01–03 that **carry volume** | 202 |
| ETF snapshots whose implied trade price is > 1 % from last price | 71 of 920,133 |
| ETF intraday decreases in cumulative volume | 0 (would fail the run) |
| days present for only one instrument (excluded from the panel and reported) | 24 IH-only, 99 ETF-only |

## Comparison with the vendor's 1-minute bars (33,600 bars)

| field | prototype | this pipeline | what is left |
|---|---|---|---|
| IH open | 59.66 % | 99.99 % | |
| IH high | 91.80 % | 99.18 % | trades between two 500 ms snapshots are not visible in snapshot data |
| IH low | 91.87 % | 99.26 % | same |
| IH close | 99.86 % | 99.99 % | |
| IH volume (exact) | 99.76 % | 99.99 % | |
| ETF open | 91.93 % | 99.88 % | |
| ETF high / low | 97.79 / 98.07 % | 98.51 / 98.98 % | intra-snapshot extremes (3 s snapshots) |
| ETF close | 99.56 % | 99.78 % | |
| ETF volume (±1 lot) | 98.96 % | 99.54 % | a 3 s snapshot spanning a minute boundary; the two adjacent bars offset each other exactly |

The vendor reports ETF volume in fractional board lots (odd-lot shares / 100), so ±1 lot is the honest tolerance.

### Diagnosis process

1. A 59.7 % open-price agreement is too low to be noise. The mismatching opens mostly equalled the **previous
   bar's close**. That pointed to quote-only snapshots that still carry the last trade price.
2. Rebuilding OHLC from trade snapshots only took IH open agreement to 99.99 %.
3. The remaining ETF volume gaps clustered at **11:30 and 15:00**. The raw feed showed final snapshots stamped
   1–3 s after the close that carry real volume (including the closing call auction). A 5-second close grace fixed
   them.
4. What still mismatches comes in **offsetting pairs of adjacent minutes** (identical daily totals), or in highs and
   lows that snapshot data cannot observe. These are limits of the data, not bugs.
