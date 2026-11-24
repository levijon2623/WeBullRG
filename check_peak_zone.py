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
  (SEE THE DEVIATION BELOW: the rule in force is a $1 edge gap and +-60 min.)

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
  P5  (ADDENDUM, user 2026-11-24, before the holdout) HEAVY vs LIGHT volume: + zones, the P1 estimand on LIGHT-volume
      arrivals minus on HEAVY-volume ones (predicted < 0: dampening strongest when hedging is a bigger share of the
      flow; the six-pack found stalls and the push signals stronger on low-volume, + gamma days). RVOL = SPY stock
      volume 09:30 .. the minute before t0 / the median of the same over the prior 20 sessions (check_sixpack_lake's
      'rvol', knowable at the arrival); LIGHT <= 0.976, HEAVY >= 1.236 = the tertile cuts of the 873 DISCOVERY zone
      arrivals (volume only, no outcomes), FROZEN; the middle third is not used. - zones by volume: secondary.
PASS (each)  holdout estimate with the predicted sign (two-sided: either) AND its day-block bootstrap 95% CI (2,000
  resamples of sessions) excluding 0 AND Benjamini-Hochberg q < 0.10 across the 8 (P1-P5), AND the discovery estimate
  has the same sign. Scored ONCE on the holdout.

DEVIATION (2026-11-24, after the first DISCOVERY run, before any holdout session was built or read). The control rule
  as written ("placebo windows not overlapping [lo - 2, hi + 2]", which in code meant a nearest edge >= $3 away, and
  arrivals within +-30 min) matched only 13% of SPY zone arrivals (40 of 276 sessions): when price sits at the zone it
  rarely reaches a level $3+ away within half an hour, so the few matches came from the fastest days and the null
  check's bands were wide. COVERAGE ONLY (no outcomes; scratchpad pz/coverage.py) on discovery, SPY+ zone arrivals
  matched: edge gap $1 30m 63% / 60m 77% / 120m 86%; $2 27 / 43 / 60%; $3 13 / 25 / 38%. RULE, chosen as the smallest
  gap that keeps controls off the zone ($1) with the shortest window reaching >= 75% coverage: a control window's
  nearest edge >= $1 from the zone's, arrivals within +-60 min (CTRL_BUF, MATCH_S). Discovery is re-scored under it as
  descriptive; the holdout is unaffected. The discovery P1-P4 estimates under the OLD rule were seen (all fail, P3a
  +0.031 against its prediction, n 42) -- recorded here so the change is not mistaken for a rescue.
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

RESULT (2026-11-24; discovery _peakzone_cache/discovery_v2.txt, holdout scored once _peakzone_cache/RESULT-holdout.txt)
  NO PEAK-GAMMA EFFECT. Holdout (276 sessions) vs discovery:
    P1  + zones THROUGH   +0.034 [+0.004, +0.060]  (disc +0.013)  FAIL -- the OPPOSITE of the prediction: + zones are
                          crossed slightly MORE than matched levels, not less. No dealer dampening.
    P2  - zones THROUGH   +0.025 [-0.003, +0.055]  (disc +0.019)  fail (right sign both halves, CI spans 0)
    P3a + zones IV15      -0.081 [-0.212, +0.042]  (disc +0.068)  fail;  P3b - zones -0.081 [-0.154, +0.002] fail
    P4a SURGE             -0.282 [-0.468, -0.087] q 0.040 (disc -0.159)  meets the letter of the rule, BUT the
                          discovery null check (global shuffle, a random frequently visited window as the "zone")
                          centred at -0.31 [-0.57, -0.05], NOT on 0 as the pre-registration required: any level price
                          keeps returning to is quieter on arrival than levels reached in passing. NOT a gamma effect.
    P4b PUSH_IMB          +0.007 fail
    P4c PUSH_IMB slope    -0.078 [-0.153, -0.006] q 0.096, but discovery +0.015 -- opposite sign: FAIL (the printed
                          'PASS' does not check the discovery-sign condition).
    P5  light - heavy     -0.069 [-0.099, +0.013] (disc -0.049)  fail (right sign both halves, CI spans 0)
  Secondary: zone arrivals STALL more than matched (+ zones +11 / +12.5 pts, - zones +6 / +1) and the next 15 min moves
    less -- consistent with the price-lives-here artefact above (no null run for stall). - zones in the FIRST 30 MIN:
    through vs matched +0.120 [+0.069, +0.193] holdout, +0.030 [-0.004, +0.094] discovery -- unregistered, a lead only.
    Futures (ES, ~40 sessions): zone = control; through 0% everywhere when futures push in. QQQ / IWM / ETF-only: same.

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
MATCH_S = 60 * 60
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


# ================================================================ scoring (written after the pre-registration, 3ab2aab)
CACHE = os.path.join(ROOT, "_peakzone_cache", "v2")   # v2: every non-overlapping window kept (control rule applied at scoring)
T0_MIN, T0_MAX = 9 * 3600 + 32 * 60, 15 * 3600 + 30 * 60
VARIANTS = (("SPY", True), ("QQQ", True), ("IWM", True), ("SPY", False))


def arrivals(b, cl, lo, hi, R):
    """[(t0, dir)] -- price enters [lo, hi] after being >= R beyond the edge it enters from (dir +1 from below)."""
    out, arm = [], 0
    for t, c in zip(b, cl):
        if c <= lo - R:
            arm = 1
        elif c >= hi + R:
            arm = -1
        elif arm == 1 and c >= lo or arm == -1 and c <= hi:
            if T0_MIN <= t <= T0_MAX:
                out.append((int(t), arm))
            arm = 0
    return out


def outcome(b, cl, i0, t0, d, lo, hi, R):
    """THROUGH / REVERSE / STALL within 15 min, and MOVE15 in bp (index i0 = the arrival bar)."""
    import numpy as np
    j1 = int(np.searchsorted(b, t0 + T_OUT_S, side="right"))
    res = "stall"
    for c in cl[i0 + 1:j1]:
        if (d > 0 and c >= hi + R) or (d < 0 and c <= lo - R):
            res = "through"
            break
        if (d > 0 and c <= lo - R) or (d < 0 and c >= hi + R):
            res = "reverse"
            break
    s0, s1 = cl[i0], cl[j1 - 1]
    return res, abs(s1 - s0) / s0 * 1e4


def contract_tables(d, tk):
    """{(strike, 'C'/'P'): {minute: (iv, vol, ask, bid)}} for the same-day expiry, whole-dollar strikes."""
    import polars as pl
    x = (pl.scan_parquet(os.path.join(ROOT, "lake", "silver", "option-contracts-1m", f"date={d}", "bars.parquet"))
         .filter((pl.col("underlying_symbol") == tk) & (pl.col("expiry").cast(pl.Utf8) == d))
         .with_columns(m=pl.col("minute_et").dt.hour().cast(pl.Int32) * 60 + pl.col("minute_et").dt.minute().cast(pl.Int32))
         .select("option_type", pl.col("strike").cast(pl.Float64), "m", "iv_close", "volume", "ask_volume", "bid_volume")
         .collect())
    out = {}
    for typ, k, m, iv, v, a, bv in x.iter_rows():
        if abs(k - round(k)) > 1e-6:
            continue
        out.setdefault((float(round(k)), typ[0].upper()), {})[m] = (iv, v or 0, a or 0, bv or 0)
    return out


def contract_feats(tab, ks, t0, d, Sx):
    """IV15 at the strike in ks nearest Sx; SURGE and PUSH_IMB over the contracts at every strike in ks."""
    m0 = t0 // 60
    kk = min(ks, key=lambda k: abs(k - Sx))
    ivs = []
    for cp in "CP":
        r = tab.get((kk, cp), {})
        a, z = r.get(m0), r.get(m0 + 15)
        if a and z and a[0] and z[0] and a[0] > 0:
            ivs.append((z[0] - a[0]) / a[0])
    iv15 = sum(ivs) / len(ivs) if ivs else None
    v2, base, nb, ask, bid = 0.0, 0.0, 0, 0.0, 0.0
    push = "C" if d > 0 else "P"
    for k in ks:
        for cp in "CP":
            r = tab.get((k, cp), {})
            for m in (m0 - 1, m0):
                x = r.get(m)
                if x:
                    v2 += x[1]
                    if cp == push:
                        ask += x[2]
                        bid += x[3]
    mins = [m for m in range(m0 - 31, m0 - 1) if m >= 570]
    if mins:
        tot = sum((tab.get((k, cp), {}).get(m) or (0, 0, 0, 0))[1] for k in ks for cp in "CP" for m in mins)
        base = tot / len(mins) * 2
        nb = len(mins)
    surge = v2 / base if nb and base > 0 else None
    pimb = (ask - bid) / (ask + bid) if ask + bid > 0 else None
    return iv15, surge, pimb


def fut_feats(fut, t0, d, b, cl, i0):
    """futures (descriptive): signed imbalance 60 s pre / post, prints burst, absorption pieces."""
    if fut is None:
        return {}
    import numpy as np
    sec, buy, sell, pr = fut

    def win(a, z):
        i, j = int(np.searchsorted(sec, a)), int(np.searchsorted(sec, z))
        return buy[i:j].sum(), sell[i:j].sum(), pr[i:j].sum()
    bp, sp, _ = win(t0 - 60, t0)
    ba, sa, pa = win(t0, t0 + 60)
    _, _, p10 = win(t0 - 600, t0)
    j = int(np.searchsorted(b, t0 + 60, side="right")) - 1
    mv = (cl[j] - cl[i0]) / cl[i0] * 1e4 * d
    return dict(f_pre=(bp - sp) / (bp + sp) * d if bp + sp else None,
                f_post=(ba - sa) / (ba + sa) * d if ba + sa else None,
                f_burst=pa / (p10 / 10.0) if p10 > 0 else None,
                f_signed=float(ba - sa) * d, mv60=mv)


def session(d):
    """every variant's zone, its arrivals and every placebo window's arrivals, with outcomes. Cached per day."""
    import json
    import numpy as np
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{d}.json")
    if os.path.exists(p):
        return json.load(open(p))
    import check_sixpack_lake as L
    import polars as pl
    out = {"d": d, "v": {}}
    try:
        path = L.price_path(d)
    except Exception as e:                               # noqa: BLE001
        out["err"] = type(e).__name__
        return out
    tabs, futs = {}, {}
    for tk, blend in VARIANTS:
        key = f"{tk}{'+' if blend else ''}"
        try:
            zm = zone_map(d, tk, blend=blend)
        except Exception as e:                           # noqa: BLE001
            out["v"][key] = {"err": type(e).__name__}
            continue
        if not zm or tk not in path:
            continue
        b, _hi, _lo, cl = path[tk]
        b = np.asarray(b, dtype=np.int64)
        R = zm["spot"] * R_BP / 1e4
        if tk not in tabs:
            tabs[tk] = contract_tables(d, tk)
        if tk not in futs:
            fp = os.path.join(ROOT, "lake", "silver", "futures-1s", f"date={d}", f"{FUT[tk]}.parquet")
            if os.path.exists(fp):
                f = pl.read_parquet(fp).sort("sec")
                futs[tk] = (f["sec"].to_numpy(), f["buy"].to_numpy(), f["sell"].to_numpy(), f["prints"].to_numpy())
            else:
                futs[tk] = None
        lo, hi = zm["lo"], zm["hi"]
        w = int(round(hi - lo))
        blo, bhi = zm["spot"] * (1 - BAND), zm["spot"] * (1 + BAND)
        wins = [(lo, hi, 1)]
        for k in range(int(math.ceil(blo)), int(math.floor(bhi)) - w + 1):
            if k + w < lo or k > hi:
                wins.append((float(k), float(k + w), 0))
        rows = []
        for wl, wh, isz in wins:
            for t0, dr in arrivals(b, cl, wl, wh, R):
                i0 = int(np.searchsorted(b, t0))
                res, mv = outcome(b, cl, i0, t0, dr, wl, wh, R)
                ks = [float(k) for k in range(int(wl), int(wh) + 1)]
                iv15, surge, pimb = contract_feats(tabs[tk], ks, t0, dr, cl[i0])
                r = dict(lo=wl, z=isz, t0=t0, dir=dr, res=res, mv15=round(mv, 3), iv15=iv15, surge=surge, pimb=pimb)
                if isz or tk == "SPY":
                    r.update(fut_feats(futs[tk], t0, dr, b, cl, i0))
                rows.append(r)
        out["v"][key] = dict(zone=dict(lo=lo, hi=hi, peak=zm["peak"], alt=zm["peak_alt"], sign=zm["sign"],
                                       net=zm["net"], spot=zm["spot"]), w=w, rows=rows)
    json.dump(out, open(p, "w"))
    return out


def sessions_of(half):
    import glob
    days = sorted(os.path.basename(f)[:10] for f in glob.glob(os.path.join(ROOT, "_breach_cache", "events", "*.json")))
    return [d for d in days if (int(d[5:7]) % 2 == 1) == (half == "discovery")]


def build(days, workers=6):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(session, days))


def pairs(S, key, metric, sign=None, zone_lo=None):
    """per session: [(zone value - mean matched control value)] for the zone (or a fake zone at zone_lo)."""
    out = {}
    for s in S:
        v = s.get("v", {}).get(key)
        if not v or "rows" not in v:
            continue
        if sign is not None and v["zone"]["sign"] != sign:
            continue
        rows = v["rows"]
        if zone_lo is None:
            Z = [r for r in rows if r["z"]]
            C = [r for r in rows if ctrl_ok(r, v["w"], v["zone"]["lo"], v["zone"]["hi"])]
        else:
            zl = zone_lo(s, v)
            if zl is None:
                continue
            Z = [r for r in rows if not r["z"] and r["lo"] == zl]
            C = [r for r in rows if r["lo"] != zl and ctrl_ok(r, v["w"], zl, zl + v["w"])]
        diffs = []
        for z in Z:
            c = [metric(r) for r in C if abs(r["t0"] - z["t0"]) <= MATCH_S]
            c = [x for x in c if x is not None]
            mz = metric(z)
            if c and mz is not None:
                diffs.append(mz - sum(c) / len(c))
        if diffs:
            out[s["d"]] = diffs
    return out


def pairs_by(S, key, metric, sign, cut, zone_lo=None):
    """P5: per session [(zone - matched control) , volume bucket of the zone arrival]; bucket 'L' rvol <= cut[0],
    'H' rvol >= cut[1], middle third dropped. Uses the same matching as pairs()."""
    out = {}
    for s in S:
        v = s.get("v", {}).get(key)
        if not v or "rows" not in v or v["zone"]["sign"] != sign:
            continue
        rows = v["rows"]
        if zone_lo is None:
            Z = [r for r in rows if r["z"]]
            C = [r for r in rows if ctrl_ok(r, v["w"], v["zone"]["lo"], v["zone"]["hi"])]
        else:
            zl = zone_lo(s, v)
            if zl is None:
                continue
            Z = [r for r in rows if not r["z"] and r["lo"] == zl]
            C = [r for r in rows if r["lo"] != zl and ctrl_ok(r, v["w"], zl, zl + v["w"])]
        got = []
        for z in Z:
            rv = z.get("rvol")
            if rv is None or cut[0] < rv < cut[1]:
                continue
            c = [metric(r) for r in C if abs(r["t0"] - z["t0"]) <= MATCH_S]
            c = [x for x in c if x is not None]
            if c and metric(z) is not None:
                got.append((metric(z) - sum(c) / len(c), "L" if rv <= cut[0] else "H"))
        if got:
            out[s["d"]] = got
    return out


def est_light_minus_heavy(P, days):
    L_ = [x for d in days for x, b in P.get(d, []) if b == "L"]
    H_ = [x for d in days for x, b in P.get(d, []) if b == "H"]
    if len(L_) < 5 or len(H_) < 5:
        return float("nan"), len(L_) + len(H_)
    return sum(L_) / len(L_) - sum(H_) / len(H_), len(L_) + len(H_)


VOLF = (os.path.join(ROOT, "_sixpack_cache", "volume.json"), os.path.join(ROOT, "_peakzone_cache", "volume_extra.json"))


def attach_rvol(S, tk="SPY", fetch=True):
    """rows of every SPY variant get rvol = SPY stock volume 09:30 .. the minute before t0 / the median of the same over
    the prior 20 sessions (>= 15 needed) -- check_sixpack_lake's 'rvol', knowable at the arrival."""
    import json
    import numpy as np
    vol = {}
    for f in VOLF:
        if os.path.exists(f):
            vol.update(json.load(open(f)))
    alld = sorted({k.split("|")[0] for k in vol if k.endswith("|" + tk)} | {s["d"] for s in S})
    need = [d for s in S for d in alld[max(0, alld.index(s["d"]) - 20):alld.index(s["d"]) + 1]
            if f"{d}|{tk}" not in vol]
    if need and fetch:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        import positioning_map as Pm
        NY = ZoneInfo("America/New_York")
        uw, extra = Pm.UW(), (json.load(open(VOLF[1])) if os.path.exists(VOLF[1]) else {})
        for d in sorted(set(need)):
            rows = uw.get(f"/stock/{tk}/ohlc/1m", date=d, limit=2500).get("data") or []
            mm = lambda ts: (lambda z: z.hour * 60 + z.minute)(datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(NY))  # noqa: E731
            extra[f"{d}|{tk}"] = {str(mm(r["start_time"])): Pm.num(r.get("volume")) or 0.0
                                  for r in rows if 570 <= mm(r["start_time"]) < 960}
        json.dump(extra, open(VOLF[1], "w"))
        vol.update(extra)
    cum = {}
    for k, v in vol.items():
        if k.endswith("|" + tk):
            a = np.zeros(390)
            for m, x in v.items():
                if 570 <= int(m) < 960:
                    a[int(m) - 570] = x
            cum[k.split("|")[0]] = np.cumsum(a)
    days_v = sorted(cum)
    for s in S:
        d = s["d"]
        if d not in cum:
            continue
        i = days_v.index(d)
        prev = days_v[max(0, i - 20):i]
        for key, v in s.get("v", {}).items():
            if not key.startswith(tk) or "rows" not in v:
                continue
            for r in v["rows"]:
                j = r["t0"] // 60 - 570 - 1
                if len(prev) >= 15 and 0 <= j < 390:
                    base = float(np.median([cum[p][j] for p in prev]))
                    r["rvol"] = float(cum[d][j]) / base if base > 0 else None
    return S


def slope_diff(S, key, zone_lo=None):
    """P4c: per session, zone and control (through, push_imb) pairs; slope difference computed after pooling."""
    out = {}
    for s in S:
        v = s.get("v", {}).get(key)
        if not v or "rows" not in v:
            continue
        rows = v["rows"]
        if zone_lo is None:
            Z = [r for r in rows if r["z"]]
            C0 = [r for r in rows if ctrl_ok(r, v["w"], v["zone"]["lo"], v["zone"]["hi"])]
        else:
            zl = zone_lo(s, v)
            if zl is None:
                continue
            Z = [r for r in rows if not r["z"] and r["lo"] == zl]
            C0 = [r for r in rows if r["lo"] != zl and ctrl_ok(r, v["w"], zl, zl + v["w"])]
        C = [r for r in C0 if any(abs(r["t0"] - z["t0"]) <= MATCH_S for z in Z)]
        zz =[(r["pimb"], r["res"] == "through", 1) for r in Z if r["pimb"] is not None]
        cc = [(r["pimb"], r["res"] == "through", 0) for r in C if r["pimb"] is not None]
        if zz and cc:
            out[s["d"]] = zz + cc
    return out


def est_mean(P, days):
    xs = [x for d in days for x in P.get(d, [])]
    return sum(xs) / len(xs) if xs else float("nan"), len(xs)


def est_slope(P, days):
    import numpy as np
    xs = [x for d in days for x in P.get(d, [])]
    if len(xs) < 20:
        return float("nan"), len(xs)
    a = np.array(xs, dtype=float)
    from scipy.stats import rankdata
    rk = (rankdata(a[:, 0]) - 1) / max(len(a) - 1, 1)

    def sl(m):
        x, y = rk[m], a[m, 1]
        return np.polyfit(x, y, 1)[0] if m.sum() >= 10 and x.std() > 0 else np.nan
    return sl(a[:, 2] == 1) - sl(a[:, 2] == 0), int((a[:, 2] == 1).sum())


def boot(P, est, n=2000, seed=7):
    import numpy as np
    days = sorted(P)
    pt, nobs = est(P, days)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        dd = [days[i] for i in rng.integers(0, len(days), len(days))]
        bs.append(est(P, dd)[0])
    bs = np.array([x for x in bs if x == x])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return pt, lo, hi, min(1.0, p), nobs, len(days)


# CONTROL RULE (deviation, see the docstring): a control window's nearest edge at least CTRL_BUF dollars from the
# zone's, arrivals within MATCH_S of the zone arrival. Set by coverage on discovery, before the holdout.
CTRL_BUF = 1.0


def ctrl_ok(r, w, lo, hi):
    return not r["z"] and (r["lo"] + w <= lo - CTRL_BUF or r["lo"] >= hi + CTRL_BUF)


THROUGH = lambda r: 1.0 if r["res"] == "through" else 0.0          # noqa: E731
QUESTIONS = (
    ("P1", "+ zones THROUGH", -1, lambda S, zl=None: pairs(S, "SPY+", THROUGH, +1, zl), est_mean),
    ("P2", "- zones THROUGH", +1, lambda S, zl=None: pairs(S, "SPY+", THROUGH, -1, zl), est_mean),
    ("P3a", "+ zones IV15", -1, lambda S, zl=None: pairs(S, "SPY+", lambda r: r["iv15"], +1, zl), est_mean),
    ("P3b", "- zones IV15", +1, lambda S, zl=None: pairs(S, "SPY+", lambda r: r["iv15"], -1, zl), est_mean),
    ("P4a", "SURGE at arrival", 0, lambda S, zl=None: pairs(S, "SPY+", lambda r: r["surge"], None, zl), est_mean),
    ("P4b", "PUSH_IMB at arrival", 0, lambda S, zl=None: pairs(S, "SPY+", lambda r: r["pimb"], None, zl), est_mean),
    ("P4c", "PUSH_IMB -> THROUGH slope, zone - ctrl", 0, lambda S, zl=None: slope_diff(S, "SPY+", zl), est_slope),
    # addendum (user, 2026-11-24, before the holdout): heavy vs light volume
    ("P5", "+ zones THROUGH, light - heavy RVOL", -1,
     lambda S, zl=None: pairs_by(S, "SPY+", THROUGH, +1, RVOL_CUT, zl), est_light_minus_heavy),
)
RVOL_CUT = (0.976, 1.236)        # SPY rvol tertile cut points, frozen from the 873 DISCOVERY zone arrivals (docstring)


def report(S, label):
    import numpy as np
    print(f"=== {label}: {len(S)} sessions")
    res = []
    for q, name, sgn, get, est in QUESTIONS:
        P = get(S)
        pt, lo, hi, p, n, nd = boot(P, est)
        res.append((q, name, sgn, pt, lo, hi, p, n, nd))
    ps = np.array([r[6] for r in res])
    order = np.argsort(ps)
    q_bh = np.empty(len(ps))
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, ps[i] * len(ps) / rank)
        q_bh[i] = prev
    for (q, name, sgn, pt, lo, hi, p, n, nd), qq in zip(res, q_bh):
        ok_sign = sgn == 0 or np.sign(pt) == sgn
        verdict = "PASS" if ok_sign and (lo > 0 or hi < 0) and qq < 0.10 else "fail"
        print(f"  {q:4s} {name:40s} {pt:+.4f} [{lo:+.4f}, {hi:+.4f}] p {p:.3f} q {qq:.3f}  n {n} arrivals / {nd} sessions"
              f"  {'(pred ' + ('-' if sgn < 0 else '+') + ')' if sgn else '(2-sided)'}  {verdict}")
    return res


def secondary(S):
    import numpy as np
    print("--- secondary (descriptive)")
    for sg in (+1, -1):
        P = pairs_by(S, "SPY+", THROUGH, sg, RVOL_CUT)
        for b in ("L", "H"):
            Q = {d: [x for x, bb in v if bb == b] for d, v in P.items()}
            Q = {d: v for d, v in Q.items() if v}
            if Q:
                pt, lo, hi, p, n, nd = boot(Q, est_mean, n=500)
                print(f"  SPY+ {'+' if sg > 0 else '-'} zones {'light' if b == 'L' else 'heavy'} volume: through vs matched "
                      f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}] n {n}")
    for key in ("SPY+", "SPY", "QQQ+", "IWM+"):
        for sg in (+1, -1):
            V = [s["v"][key] for s in S if key in s.get("v", {}) and "rows" in s["v"][key] and s["v"][key]["zone"]["sign"] == sg]
            Z = [r for v in V for r in v["rows"] if r["z"]]
            if not Z:
                continue
            rate = lambda R, x: np.mean([r["res"] == x for r in R]) if R else np.nan   # noqa: E731
            tP = pairs(S, key, THROUGH, sg)
            mP = pairs(S, key, lambda r: r["mv15"], sg)
            sP = pairs(S, key, lambda r: 1.0 if r["res"] == "stall" else 0.0, sg)
            print(f"  {key:5s} {'+' if sg > 0 else '-'} zones: {len(V)} sessions, {len(Z)} arrivals | zone through "
                  f"{rate(Z, 'through'):.0%} stall {rate(Z, 'stall'):.0%} reverse {rate(Z, 'reverse'):.0%} | vs matched: "
                  f"through {est_mean(tP, sorted(tP))[0]:+.3f}  stall {est_mean(sP, sorted(sP))[0]:+.3f}  "
                  f"move15 {est_mean(mP, sorted(mP))[0]:+.2f} bp")
    # first 30 minutes vs later (SPY+)
    for lab, cond in (("first 30 min", lambda r: r["t0"] < 10 * 3600), ("after 10:00", lambda r: r["t0"] >= 10 * 3600)):
        for sg in (+1, -1):
            P = {}
            for s in S:
                v = s.get("v", {}).get("SPY+")
                if not v or "rows" not in v or v["zone"]["sign"] != sg:
                    continue
                for z in [r for r in v["rows"] if r["z"] and cond(r)]:
                    c = [THROUGH(r) for r in v["rows"] if ctrl_ok(r, v["w"], v["zone"]["lo"], v["zone"]["hi"])
                         and abs(r["t0"] - z["t0"]) <= MATCH_S]
                    if c:
                        P.setdefault(s["d"], []).append(THROUGH(z) - sum(c) / len(c))
            pt, lo, hi, p, n, nd = boot(P, est_mean, n=500)
            print(f"  SPY+ {'+' if sg > 0 else '-'} zones {lab:13s} through vs matched {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] n {n}")


def futures_desc(S):
    import numpy as np
    print("--- futures (ES, descriptive; sessions with futures-1s only)")
    for sg in (+1, -1, None):
        Z, C = [], []
        for s in S:
            v = s.get("v", {}).get("SPY+")
            if not v or "rows" not in v or (sg is not None and v["zone"]["sign"] != sg):
                continue
            zs = [r for r in v["rows"] if r["z"] and r.get("f_post") is not None]
            Z += zs
            C += [r for r in v["rows"] if ctrl_ok(r, v["w"], v["zone"]["lo"], v["zone"]["hi"]) and r.get("f_post") is not None
                  and any(abs(r["t0"] - z["t0"]) <= MATCH_S for z in zs)]
        if not Z:
            continue
        f = lambda R, k: np.nanmedian([r[k] for r in R if r.get(k) is not None]) if R else np.nan   # noqa: E731
        absorb = lambda R: np.mean([(r["f_post"] > 0.1) and abs(r["mv60"]) < 2 for r in R]) if R else np.nan  # noqa: E731
        thr = lambda R: np.mean([r["res"] == "through" for r in R])                                        # noqa: E731
        lab = {1: "+ zones", -1: "- zones", None: "all"}[sg]
        print(f"  {lab:8s} zone n {len(Z):4d} | imbalance pre {f(Z, 'f_pre'):+.3f} post {f(Z, 'f_post'):+.3f} burst "
              f"{f(Z, 'f_burst'):.2f} pushing-not-moving {absorb(Z):.0%} through {thr(Z):.0%}")
        print(f"  {'':8s} ctrl n {len(C):4d} | imbalance pre {f(C, 'f_pre'):+.3f} post {f(C, 'f_post'):+.3f} burst "
              f"{f(C, 'f_burst'):.2f} pushing-not-moving {absorb(C):.0%} through {thr(C):.0%}")
        # does futures pushing INTO the zone change the outcome more at the zone than elsewhere?
        for nm, R in (("zone", Z), ("ctrl", C)):
            hi_ = [r for r in R if (r.get("f_pre") or 0) > 0.1]
            lo_ = [r for r in R if (r.get("f_pre") or 0) <= 0.1]
            if hi_ and lo_:
                print(f"  {'':8s} {nm}: futures pushing in (pre > 0.1) through {thr(hi_):.0%} (n {len(hi_)}) vs not "
                      f"{thr(lo_):.0%} (n {len(lo_)})")


def null_check(S, draws=200, seed=11):
    """global shuffle: a random placebo window (same width) relabelled as the zone, sign taken from a random session."""
    import numpy as np
    rng = np.random.default_rng(seed)
    signs = [s["v"]["SPY+"]["zone"]["sign"] for s in S if "rows" in s.get("v", {}).get("SPY+", {})]
    res = {q: [] for q, *_ in QUESTIONS}
    for _ in range(draws):
        fake = {}
        S2 = []
        for s in S:
            v = s.get("v", {}).get("SPY+")
            if not v or "rows" not in v:
                continue
            los = sorted({r["lo"] for r in v["rows"] if not r["z"]})
            if not los:
                continue
            zl = los[rng.integers(0, len(los))]
            fake[s["d"]] = zl
            v2 = dict(v, zone=dict(v["zone"], sign=int(signs[rng.integers(0, len(signs))])))
            S2.append(dict(s, v=dict(s["v"], **{"SPY+": v2})))
        for q, name, sgn, get, est in QUESTIONS:
            P = get(S2, lambda s, v: fake.get(s["d"]))
            res[q].append(est(P, sorted(P))[0])
    print(f"--- null check ({draws} global-shuffle draws): estimand mean / 95% band -- should centre on 0")
    for q, name, *_ in QUESTIONS:
        a = np.array([x for x in res[q] if x == x])
        if not len(a):
            print(f"  {q:4s} {name:40s} (no estimate)")
            continue
        print(f"  {q:4s} {name:40s} mean {a.mean():+.4f}  [{np.percentile(a, 2.5):+.4f}, {np.percentile(a, 97.5):+.4f}]")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--null", type=int, default=200)
    ap.add_argument("--days", type=int, default=0, help="first N sessions only (code check)")
    ap.add_argument("--build-only", action="store_true", help="build the per-session cache, score nothing")
    a = ap.parse_args()
    if a.holdout:
        mark = os.path.join(CACHE, "HOLDOUT-SCORED.txt")
        if os.path.exists(mark):
            sys.exit("holdout already scored -- see " + mark)
        days = sessions_of("holdout")
    elif a.discovery:
        days = sessions_of("discovery")
    else:
        print(__doc__)
        return
    if a.days:
        days = days[:a.days]
    S = build(days)
    attach_rvol(S)
    errs = [s for s in S if s.get("err")]
    print(f"{len(S)} sessions built, {len(errs)} errors")
    if a.build_only:
        return
    if a.discovery and a.null:
        null_check(S, a.null)
    report(S, "HOLDOUT (scored once)" if a.holdout else "DISCOVERY")
    secondary(S)
    futures_desc(S)
    if a.holdout:
        open(os.path.join(CACHE, "HOLDOUT-SCORED.txt"), "w").write("scored\n")


if __name__ == "__main__":
    main()
