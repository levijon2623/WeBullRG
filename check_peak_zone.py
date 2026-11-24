"""
check_peak_zone.py -- PRE-REGISTRATION (user, 2026-11-24): what happens when price ARRIVES at the peak-gamma zone?

BACKGROUND. check_magnet (2026-09-03) and check_peak_gamma (2026-10-22) settled that price is NOT drawn to peak gamma
(reached at or below a random level's rate; the node follows price). This asks a different, mechanism-level question
the user raised on 2026-11-24 after SPY / SPX found the 713 ~ 715 / SPX 7140 zone of the weekly GEX map: once price is
THERE, does the zone behave differently from an ordinary level -- price, the zone's own contracts, and the futures?
Dealer-hedging theory gives signed predictions: a POSITIVE net-gamma node is hedged against the move (sell rallies /
buy dips -> stall, smaller moves, IV bleeds); a NEGATIVE node is hedged with it (-> faster through). The 11-24 zone
was NEGATIVE (713: net -547.9M on the map, puts -655M).

THE ZONE (frozen; built to match weekly_gex.html's 0DTE column with "+ index", the map the user reads)
  map      09:31 ET, same-day expiry only. Per contract gamma (last gamma_close <= 09:31; Black-Scholes from the last
           iv_close <= 09:31 when missing) x start-of-day OI x spot^2, calls +, puts -, summed per strike; the index
           (SPXW on SPY, NDXP on QQQ, RUTW on IWM) added on the ETF grid by index_gex.blend_rows at that minute's
           index/ETF ratio (NDXP bars carry no index price: forward from put-call parity, 5 strikes nearest C = P).
           Strikes within +-5% of spot (live_state.WEEKLY_BAND). Peak = max |net gamma| strike, peak_alt = the
           runner-up > 1 step away holding >= 85% of it (index_gex.walls_with_ties, the page's "713 ~ 715").
  zone     [min(P, A), max(P, A)] when the alt A is within $3 of the peak P, else [P - 1, P + 1].
  sign     sign of the summed net gamma over the zone's strikes. + zone / - zone.
  MACHINERY CHECK (done before this file was committed; scratchpad pz_map.py vs the saved live maps
  live/weekly_gex 2026-09-28..11-13, 32 SPY sessions): blended peak within $1 of the page's 94% (same strike 66%;
  ETF-only 97%). QQQ blend 78% (NDX gamma agrees less), IWM 91%. -> SPY primary, QQQ / IWM secondary.

ARRIVALS (5-second underlying path, check_sixpack_lake.price_path)
  An ARRIVAL is price entering [lo, hi] after having been at least R = 5 bp beyond the edge it enters from (each
  re-entry after such an excursion counts again). 09:32 <= t0 <= 15:30 ET. Direction = the side it came from.
  PLACEBO ZONES the same day: every window of the zone's width (same number of $1 strikes) on the $1 grid that does not
  overlap [lo - 2, hi + 2], within the +-5% band; their arrivals by the same rule. MATCHED CONTROLS of a zone arrival
  = placebo arrivals on the same session within +-30 min of t0. A zone arrival with no control is dropped (counted).

OUTCOMES (from t0, 15 minutes)
  THROUGH  price reaches the far edge + R before the entry edge - R; REVERSE the opposite; STALL neither in 15 min.
  MOVE15   |S(t0 + 15m) - S(t0)| in bp.
  IV15     the zone's ATM straddle (0DTE call + put at the zone strike nearest S(t0)): mean of the two legs' iv_close
           change t0-minute -> t0 + 15, in % of itself (lake iv is daily vol; a ratio, so the convention cancels).
  CONTRACT FLOW AT ARRIVAL (the user's addition, 2026-11-24), the zone's 0DTE contracts (calls and puts at every
           strike in [lo, hi]), the arrival minute and the minute before:
             SURGE    their volume / (their mean per-minute volume over the prior 30 min x 2); >= 1 = busier than usual
             PUSH_IMB (ask - bid) / (ask + bid) volume on the PUSH side (calls when arriving from below, puts from above)
           a control arrival's are its own placebo zone's contracts, same construction.

PRIMARY QUESTIONS, SPY (+ SPX) -- estimand = mean over zone arrivals of (zone outcome - mean of its matched controls)
  P1  + zones: THROUGH rate lower than matched (predicted < 0).
  P2  - zones: THROUGH rate higher than matched (predicted > 0).
  P3a + zones: IV15 lower than matched (predicted < 0).
  P3b - zones: IV15 higher than matched (predicted > 0).
  P4a SURGE at arrival differs from matched (two-sided) -- is the node busier when price gets there?
  P4b PUSH_IMB at arrival differs from matched (two-sided).
  P4c does PUSH_IMB predict THROUGH differently at the zone? slope of THROUGH on PUSH_IMB (rank-scaled 0..1) at zone
      arrivals minus the same slope at control arrivals (two-sided).
PASS (each)  holdout estimate with the predicted sign (two-sided: either) AND its day-block bootstrap 95% CI (2,000
  resamples of sessions) excluding 0 AND Benjamini-Hochberg q < 0.10 across the 7, AND the discovery estimate has the
  same sign. Scored ONCE on the holdout.
SECONDARY (reported, no pass/fail): MOVE15 and STALL / REVERSE rates by sign; P1-P4 on QQQ (+NDX) and IWM (+RUT); the
  ETF-only map; arrivals in the first 30 minutes vs later (the shootout window is fast regardless of gamma).
FUTURES (descriptive only -- futures-1s covers 2026-07-15..11-13, ~85 sessions, too few for a test): ES for SPY (NQ
  QQQ, RTY IWM), seconds aligned to t0. Signed aggressor imbalance (buy - sell)/(buy + sell) in the arrival direction,
  60 s before and 60 s after; prints in the 60 s after / the prior 10-minute mean per 60 s; ABSORPTION = signed futures
  volume in the 60 s after / the SPY move in bp over it (pushing without moving = absorbed). Zone vs matched, + and -
  zones, with day-block CIs; prior result to compare with: aggressive futures flow into ANY strike -> stall (6-pack).

SAMPLE  sessions = _breach_cache/events (2024-08-20..2026-11-13, 552). DISCOVERY = sessions in ODD calendar months,
  HOLDOUT = EVEN months. Exploration and code checks on discovery only; the holdout is read once. Not spent for this
  question: check_peak_gamma tested REACH, the six-pack tested strike touches without a gamma label.
NULL CHECK before scoring: on discovery, relabel a random placebo window as the "zone" (global shuffle across sessions,
  200 draws); the P1-P4 estimands must centre on 0.

    python check_peak_zone.py --machinery      (rebuild-vs-live map check; needs the live/weekly_gex archives in --wk)
    python check_peak_zone.py --discovery
    python check_peak_zone.py --holdout        (once)
"""
from __future__ import annotations

import math
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

IDX = {"SPY": "SPXW", "QQQ": "NDXP", "IWM": "RUTW"}
FUT = {"SPY": "ES", "QQQ": "NQ", "IWM": "RTY"}
SNAP_MOD = 9 * 60 + 31
BAND = 0.05
R_BP = 5.0
T_OUT_S = 15 * 60
MATCH_S = 30 * 60
ALT_JOIN = 3.0


def bs_gamma(S, K, T, v):
    if S <= 0 or K <= 0 or T <= 0 or not v or v <= 0:
        return None
    sd = v * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sd * sd) / sd
    return math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (S * sd)


def parity_spot(y, mod):
    """forward from put-call parity on the last quotes <= mod: K + C - P at the 5 strikes with the smallest |C - P|."""
    import polars as pl
    q = (y.filter((pl.col("m") <= mod) & (pl.col("bid_close") > 0) & (pl.col("ask_close") > pl.col("bid_close")))
         .sort("m").group_by("option_chain_id")
         .agg(pl.col("strike").first(), pl.col("option_type").first(),
              ((pl.col("bid_close") + pl.col("ask_close")) / 2).last().alias("mid")))
    c = {k: v for k, t, v in q.select("strike", "option_type", "mid").iter_rows() if t == "call"}
    p = {k: v for k, t, v in q.select("strike", "option_type", "mid").iter_rows() if t == "put"}
    both = sorted((abs(c[k] - p[k]), k + c[k] - p[k]) for k in set(c) & set(p))[:5]
    if len(both) < 3:
        return None
    f = sorted(x[1] for x in both)
    return f[len(f) // 2]


def rows_for(d, syms, mod=SNAP_MOD):
    """{sym: (spot, rows[{e,k,cg,pg}])} for the same-day expiry, knowable at minute-of-day `mod` (ET)."""
    import polars as pl
    x = (pl.scan_parquet(os.path.join(ROOT, "lake", "silver", "option-contracts-1m", f"date={d}", "bars.parquet"))
         .filter(pl.col("underlying_symbol").is_in(list(syms)) & (pl.col("expiry").cast(pl.Utf8) == d))
         .with_columns(m=pl.col("minute_et").dt.hour().cast(pl.Int32) * 60 + pl.col("minute_et").dt.minute().cast(pl.Int32))
         .select("underlying_symbol", "option_chain_id", "option_type", pl.col("strike").cast(pl.Float64), "m",
                 "gamma_close", "iv_close", "open_interest", "underlying_close", "bid_close", "ask_close")
         .collect())
    out = {}
    for s in syms:
        y = x.filter(pl.col("underlying_symbol") == s)
        pre = y.filter((pl.col("m") <= mod) & (pl.col("m") >= 570))
        if pre.is_empty():
            continue
        spot = float(pre.sort("m").select("underlying_close").drop_nulls().tail(50).median().item() or 0)
        if spot <= 0:
            spot = parity_spot(y, mod) or 0.0
        if spot <= 0:
            continue
        oi = y.group_by("option_chain_id").agg(pl.col("open_interest").drop_nulls().first().alias("oi"),
                                               pl.col("strike").first(), pl.col("option_type").first())
        g = (y.filter(pl.col("m") <= mod).sort("m").group_by("option_chain_id")
             .agg(pl.col("gamma_close").drop_nulls().last().alias("g"), pl.col("iv_close").drop_nulls().last().alias("iv")))
        c = oi.join(g, on="option_chain_id", how="left").filter(pl.col("oi") > 0)
        T = max(960 - mod, 1) / 1440.0
        agg = {}
        for o, k, typ, gm, iv in c.select("oi", "strike", "option_type", "g", "iv").iter_rows():
            if gm is None:
                gm = bs_gamma(spot, k, T, iv)
            if gm is None:
                continue
            v = gm * o * spot * spot
            a = agg.setdefault(k, [0.0, 0.0])
            if typ == "call":
                a[0] += v
            else:
                a[1] -= v
        out[s] = (spot, [dict(e=d, k=k, cg=a[0], pg=a[1]) for k, a in agg.items()])
    return out


def zone_map(d, tk, mod=SNAP_MOD, blend=True):
    """the weekly page's 0DTE map (+ index): {spot, heat{k: [c, p]}, peak, peak_alt, lo, hi, sign}"""
    import index_gex as IX
    r = rows_for(d, (tk, IDX[tk]) if blend else (tk,), mod)
    if tk not in r:
        return None
    spot, rows = r[tk]
    lo_b, hi_b = spot * (1 - BAND), spot * (1 + BAND)
    if blend and IDX[tk] in r:
        rows, _prov = IX.blend_rows(rows, r[IDX[tk]][1], r[IDX[tk]][0] / spot)
    heat = IX.heat_of(rows, {d}, lo_b, hi_b)
    if not heat:
        return None
    w = IX.walls_with_ties({k: (v[0], v[1]) for k, v in heat.items()}) or {}
    P, A = w.get("peak"), w.get("peak_alt")
    if P is None:
        return None
    lo, hi = (min(P, A), max(P, A)) if A is not None and abs(A - P) <= ALT_JOIN else (P - 1.0, P + 1.0)
    net = sum(sum(v) for k, v in heat.items() if lo - 1e-9 <= k <= hi + 1e-9)
    return dict(spot=spot, heat=heat, peak=P, peak_alt=A, lo=lo, hi=hi, sign=1 if net >= 0 else -1, net=net)


def main():
    # Scoring code is written AFTER this pre-registration is committed (see the docstring); --machinery lives in the
    # scratchpad check (pz_map.py) that produced the numbers quoted above.
    print(__doc__)


if __name__ == "__main__":
    main()
