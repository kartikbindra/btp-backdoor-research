# D42 Plan D — sealed stealth confirm, power/feasibility note

**Date:** 2026-10-24 · **Status:** ANALYSIS (no data; informs Plan D pre-registration) · decision D42

## Objective

Before pre-registering Plan D, check that the sealed c007x **stealth2** split is large enough for the
frozen stealth gate — one-sided 95% upper bound (UB) on the full-cache marker rate **<= 0.003** per
regime — to be achievable, and find how many leak events each regime can tolerate.

## Method

`stats_c007.one_sided_upper_bound(k, n)` (the frozen estimator) at the stealth2 sizes
(short 1200, mid 1000, long 1000; D42 plan A freeze).

## Finding

| regime | n | UB @ 0 events | UB @ 1 event | max events that still pass (UB<=0.003) |
|---|---|---|---|---|
| short | 1200 | 0.00249 | 0.00395 | **0** |
| mid | 1000 | 0.00299 | 0.00473 | **0** |
| long | 1000 | 0.00299 | 0.00473 | **0** |

**The confirm is a zero-event gate.** At these sizes a single full-cache leak event in any regime
fails that regime (UB jumps to ~0.004–0.0047 > 0.003). mid and long sit right at the threshold even
with 0 events (0.00299). To buy a 1-event tolerance at 0.003 a regime would need n >= ~1570, which
stealth2 does not have.

## Consequences (for Plan D pre-registration; nothing changed yet)

1. The Plan C dev selection gate (zero c0 events on dev_stealth short+mid and dev_len bins <=128) is
   the *right* proxy: the sealed confirm will likewise tolerate zero events, so a candidate that shows
   any short/mid full-cache leak on dev should not be advanced.
2. The confirm is **brittle**: even a dev-clean candidate can fail on one unlucky sealed event. This is
   defensible for a stealth claim (a backdoor that fires benignly even rarely under C0 is not stealthy),
   but it must be stated as the accepted risk in the Plan D pre-registration.
3. Options to record for the Research Director (not decided here): (a) accept zero-tolerance; (b) grow a
   fresh, re-frozen stealth2 to ~1600+/regime to tolerate 1 event before the confirm; (c) keep the gate
   but report the exact binomial p-value alongside the UB. Changing the frozen 0.003 threshold is out of
   scope without a decision entry.

## Files

- Computation only (no artifact committed beyond this note); reproducible from
  `stats_c007.one_sided_upper_bound` at the stealth2 sizes.
