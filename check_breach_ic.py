"""
check_breach_ic.py -- IC SCREEN + LASSO ON 0DTE STRIKE BREACHES (user, 2026-11-14). PRE-REGISTERED 2026-11-14,
committed before any input below was scored against any target.

CONTEXT. Breach scalps sized on the lake today (scratchpad study_breach / study_breach_tape): a strike breach on
SPY/QQQ/IWM is a coin flip mid-to-mid (continuation A -$0.11, turnback B1 -$0.32..-0.45 per contract, tape-timed),
and no single filter (surge, gamma sign, 30-min trend, VWAP, 0DTE flow, path30) made it pay. The question here is
whether a careful SET of inputs, all knowable at the confirmed breach, ranks breaches by continuation vs failure.
Prior: the 10-03 IC screen on the flow trigger found inputs that rank the SIZE of a move, never its direction, so
the target is signed by the breach (continuation vs failure), and the money test is separate.

EVENT (same as study_breach_tape, look-ahead fixed). 5-second bars from the underlying price on option prints
(trades-core, bronze full-tape after 2026-09-25), median of 3 closes. Armed when price is >= 5 bp + $0.10 below
strike K (above for a down-break); breach when the median reaches K + $0.10. The median at bar i needs bar i+1, so
the breach is KNOWN at t_in = end of bar i+1 + 1 s. One event per ticker per 5 minutes; t_in 09:45-15:30.
SPY / QQQ / IWM, $1 strikes. side = +1 up-break, -1 down-break.

TARGETS (from t_in, on the median path)
  Y   PRIMARY, ternary: +1 the next strike (K + side x 0.95) is reached before price falls back through K by $0.10
      (K - side x 0.10), -1 the reverse, 0 neither within 15 minutes.
  M2 / M5 / M10  side x (price at t_in + 2 / 5 / 10 min - price at t_in), $ (= strike units). Reported only.

INPUTS (28; everything stamped at or before t_in; option minute bars only when CLOSED, i.e. minute m with
(m + 1) x 60 <= t_in; "signed" = multiplied by side, so + always means "with the breach")
  APPROACH  a_speed    bp per minute from arming to the breach
            a_accel    signed second difference of price over the last 2 minutes (bp)
            a_tests    times the median crossed K earlier today
            a_since    minutes since the median was last within $0.05 of K before arming (390 if never)
            a_fresh    1 if price had not been beyond K on the breach side ($0.10) earlier today
  STRIKE    s_surge    six-pack surge: K-1..K+1 calls+puts, volume last 5 min / 15 min before (rate), 0 if < 50
            s_kcall    signed (ask - bid volume) at the K call, last 5 min
            s_kput     -side x (ask - bid volume) at the K put, last 5 min  (put buying against an up-break = -)
            s_oi_k     log(start-of-day OI at K, calls+puts / median OI of K-5..K+5), 0DTE
            s_oi_next  the same at the next strike K + side
            s_wall     1 if K is the call wall (up-break) / put wall (down-break), 0-7 day chain at 10:00
            s_flip     signed distance of price from the gamma flip (bp), 0-7 day chain, IVs at 10:00 (gamma_flip)
  QUOTE     q_spread   (ask - bid) / mid of the K contract in the breach direction, last closed minute
            q_imb      (ask - bid volume) / (ask + bid + 1) of that contract, last 3 minutes
            q_strad    % change of the K straddle mid over the last 15 minutes
  DAY       c_am_used  |price - 09:30 price| / the 09:35 ATM straddle
            c_path30   absolute path of the last 30 minutes / the 09:35 straddle (volatility-gauge proxy)
            c_tr30     signed 30-minute trend (bp)
            c_vwap     signed distance from VWAP (bp)
            c_range    signed position in the day's range so far (1 = at the breach-side extreme)
            c_tod      minutes since 09:30
            c_gex      dealer gamma (spot-exposures, gamma per 1% move), last minute
            c_rvol     cumulative volume / mean cumulative volume at the same minute, prior 20 sessions
            c_tide     signed 15-minute change of UW Market Tide (net call premium - net put premium)  [user]
  FLOW      f_prem15   signed 15-minute change of 0DTE net premium (calls - puts, ask - bid side)
            f_delta5   signed 0DTE directional delta bought, 5 min: sum (ask - bid vol) x delta x 100  [user]
            f_delta15  the same over 15 minutes                                                   [user]
  CROSS     x_breach   other tickers' confirmed breaches in the same direction in [t_in - 60 s, t_in]
Six inputs were already looked at on the whole lake today as single filters (s_surge, c_gex sign, c_tr30, c_vwap,
f_prem15, c_path30), so the holdout is only PARTLY clean for them. Said here, before scoring.

SAMPLES
  DISCOVERY  2024-08-20 .. 2025-12-31 (halves split at 2025-04-30)
  HOLDOUT    2026-01-02 .. 2026-09-25, scored ONCE (--holdout writes _breach_cache/RESULT-holdout.txt and refuses
             to run again)
  FRESH      2026-09-26 .. 2026-10-30 (bronze tape, never touched): reported, not scored

PREPROCESSING  each input -> its rank within ticker against the DISCOVERY distribution (empirical CDF, 0..1);
               missing -> 0.5. The same discovery cut points are applied to the holdout.

PASS RULES
  P1 IC (discovery): Spearman(input, Y) pooled; p from a day-block bootstrap (2,000); BH q < 0.10 across the 28
     AND the same sign in both halves. Reported; not needed for the verdict.
  P2 LASSO: LassoCV (GroupKFold by ISO week, 5 folds) on all 28 ranks -> Y, fit on discovery. PASS = it keeps >= 1
     input AND on the holdout Spearman(score, Y) > 0 with the day-block bootstrap 95% CI above 0.
  P3 MONEY (holdout): score quintile cut points from discovery. TOP quintile traded as A (continuation: ATM K
     contract in the breach direction; TP next strike - $0.05, SL back through K by $0.10, 15-min stop); BOTTOM
     quintile traded as B1 sl0.35 (turnback: the opposite K contract; TP K - side x $0.20, stop when its mid is
     $0.35 below entry). Priced on the tape (NBBO on the contract's own prints) at the user's measured fills: buy
     mid + $0.06, sell mid - $0.08 per contract, $0.10 fees round trip, 1 s latency. PASS each = mean P&L per
     contract > 0 with the day-block 95% CI above 0; BH q < 0.10 across the two.
  OVERALL PASS = P2 AND at least one P3 -> a forward paper log from the live strike cue. Nothing is deployed on it.

Usage:  python check_breach_ic.py --tide (once) | --build | --discovery | --holdout | --fresh
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import check_sixpack_lake as L                     # noqa: E402
import gamma_flip as GF                            # noqa: E402

CACHE = os.path.join(ROOT, "_breach_cache")
EV = os.path.join(CACHE, "events")
TK = ("SPY", "QQQ", "IWM")
ZB, RBP, LAT = 0.10, 5, 1.0
BUY_C, SELL_C, FEE_RT = 0.0006, 0.0008, 0.0010       # $ per share
DISC = ("2024-08-20", "2025-12-31"); HALF = "2025-04-30"
HOLD = ("2026-01-02", "2026-09-25"); FRESH = ("2026-09-26", "2026-10-30")
INPUTS = ["a_speed", "a_accel", "a_tests", "a_since", "a_fresh",
          "s_surge", "s_kcall", "s_kput", "s_oi_k", "s_oi_next", "s_wall", "s_flip",
          "q_spread", "q_imb", "q_strad",
          "c_am_used", "c_path30", "c_tr30", "c_vwap", "c_range", "c_tod", "c_gex", "c_rvol", "c_tide",
          "f_prem15", "f_delta5", "f_delta15", "x_breach"]
N_BOOT, SEED = 2000, 20261114
_GEX = {}


# ------------------------------------------------------------------ data
def _secs(df, col="executed_at"):
    et = df[col].dt.convert_time_zone("America/New_York")
    return (et.dt.hour().cast(pl.Float64) * 3600 + et.dt.minute().cast(pl.Float64) * 60
            + et.dt.second().cast(pl.Float64) + et.dt.microsecond().cast(pl.Float64) / 1e6)


def tape(d):
    """{(tk, K, 'C'/'P'): (sec[], bid[], ask[])} same-day expiry, whole-dollar strikes, from the trade tape."""
    src = glob.glob(os.path.join(L.LAKE, "silver", "trades-core", f"date={d}", "*.parquet")) or \
        [p for p in [os.path.join(L.LAKE, "bronze", "full-tape", f"{d}.parquet")] if os.path.exists(p)]
    x = (pl.scan_parquet(src).filter(pl.col("underlying_symbol").is_in(list(TK)) & (pl.col("expiry").cast(pl.Utf8) == d)
                                     & (pl.col("canceled") != True))
         .select("underlying_symbol", "executed_at", "option_type", pl.col("strike").cast(pl.Float64),
                 pl.col("nbbo_bid").cast(pl.Float64), pl.col("nbbo_ask").cast(pl.Float64)).collect())
    x = x.with_columns(sec=_secs(x), cp=pl.col("option_type").str.to_uppercase().str.slice(0, 1))
    x = x.filter((pl.col("nbbo_ask") > pl.col("nbbo_bid")) & (pl.col("nbbo_bid") >= 0)).sort("executed_at")
    out = {}
    for (t, k, cp), g in x.group_by(["underlying_symbol", "strike", "cp"]):
        if abs(k - round(k)) < 1e-6:
            out[(t, int(round(k)), cp)] = (g["sec"].to_numpy(), g["nbbo_bid"].to_numpy(), g["nbbo_ask"].to_numpy())
    return out


def minute_book(d):
    """Option minute bars: 0DTE book {(tk,K,cp): {m: row}} and the 0-7 day chain at 10:00 for the flip."""
    f = os.path.join(L.LAKE, "silver", "option-contracts-1m", f"date={d}", "bars.parquet")
    dd = dt.date.fromisoformat(d)
    x = (pl.scan_parquet(f).filter(pl.col("underlying_symbol").is_in(list(TK)))
         .select("underlying_symbol", "option_type", pl.col("strike").cast(pl.Float64), pl.col("expiry").cast(pl.Utf8),
                 "minute_et", "bid_close", "ask_close", "ask_volume", "bid_volume", "volume", "vwap", "delta_close",
                 "iv_close", "open_interest").collect())
    x = x.with_columns(m=pl.col("minute_et").dt.hour().cast(pl.Int32) * 60 + pl.col("minute_et").dt.minute().cast(pl.Int32),
                       cp=pl.col("option_type").str.to_uppercase().str.slice(0, 1))
    z = x.filter(pl.col("expiry") == d)
    book = defaultdict(dict)
    for t, k, cp, m, b, a, av, bv, v, vw, de, oi in z.select("underlying_symbol", "strike", "cp", "m", "bid_close", "ask_close",
                                                             "ask_volume", "bid_volume", "volume", "vwap", "delta_close",
                                                             "open_interest").iter_rows():
        if k is None or abs(k - round(k)) > 1e-6:
            continue
        book[(t, int(round(k)), cp)][m] = (b, a, av or 0, bv or 0, v or 0, vw, de, oi)
    chains = {}
    w = x.filter((pl.col("m") <= 600) & (pl.col("iv_close").is_not_null()))
    w = w.sort("m").group_by(["underlying_symbol", "strike", "expiry", "cp"]).agg(pl.col("iv_close").last(),
                                                                                    pl.col("open_interest").first())
    for t in TK:
        rows = []
        for k, e, cp, iv, oi in w.filter(pl.col("underlying_symbol") == t).select("strike", "expiry", "cp", "iv_close",
                                                                                   "open_interest").iter_rows():
            days = (dt.date.fromisoformat(e) - dd).days
            if 0 <= days <= 7 and oi:
                rows.append((k, max(days, 0) / 365.0 + 6.0 / (24 * 365), iv, oi, cp == "C"))
        chains[t] = rows
    return book, chains


def pull_tide():
    """UW Market Tide, 1-minute, one call per session -> _breach_cache/tide/DATE.json (history from 2024-08-20)."""
    import time
    import positioning_map as P
    uw, out = P.UW(), os.path.join(CACHE, "tide")
    os.makedirs(out, exist_ok=True)
    for d in L.lake_days(DISC[0], FRESH[1]):
        f = os.path.join(out, d + ".json")
        if os.path.exists(f):
            continue
        r = None
        for k in range(3):
            try:
                r = uw.get("/market/market-tide", date=d, interval_5m="false"); break
            except Exception:
                time.sleep(3 * (k + 1))
        json.dump((r or {}).get("data") or [], open(f, "w"))
        time.sleep(0.25)


def tide_series(d):
    f = os.path.join(CACHE, "tide", d + ".json")
    if not os.path.exists(f):
        return {}
    out = {}
    for r in json.load(open(f)):
        ts = str(r.get("timestamp") or "")
        if len(ts) >= 16:
            m = int(ts[11:13]) * 60 + int(ts[14:16])
            try:
                out[m] = float(r["net_call_premium"]) - float(r["net_put_premium"])
            except (TypeError, ValueError, KeyError):
                pass
    return out


# ------------------------------------------------------------------ one day
def build_day(d):
    f = os.path.join(EV, d + ".json")
    if os.path.exists(f):
        return d, "have"
    try:
        path = L.price_path(d)
        book, chains = minute_book(d)
        tq = tape(d)
    except Exception as e:
        return d, f"skip {type(e).__name__}"
    gexapi = json.load(open(os.path.join(ROOT, "_sixpack_cache", "gex_api.json")))
    vol = json.load(open(os.path.join(ROOT, "_sixpack_cache", "volume.json")))
    tide = tide_series(d)
    days_all = L.lake_days("2024-06-01", d)
    prior = [x for x in days_all if x < d][-20:]
    raw, ev = [], []
    per_tk = {}
    for tk, (sc, hi, lo, cl) in path.items():
        if sc[-1] < 57300 or len(sc) < 100:
            continue
        med = np.median(np.vstack([np.r_[cl[:1], cl[:-1]], cl, np.r_[cl[1:], cl[-1:]]]), axis=0)
        per_tk[tk] = (sc, med)
        p0 = float(med[0]); Rr = p0 * RBP / 1e4
        for K in range(int(np.floor(med.min())), int(np.ceil(med.max())) + 1):
            armed, arm_i = 0, None
            for i in range(len(med)):
                if med[i] <= K - ZB - Rr:
                    armed, arm_i = 1, i
                elif med[i] >= K + ZB + Rr:
                    armed, arm_i = -1, i
                if armed == 1 and med[i] >= K + ZB:
                    side = 1
                elif armed == -1 and med[i] <= K - ZB:
                    side = -1
                else:
                    continue
                if i + 1 < len(sc):
                    raw.append((tk, float(sc[i + 1]) + 5 + LAT, i, K, side, arm_i))
                armed = 0
    raw.sort(key=lambda r: r[1])
    last = {}
    for tk, t_in, i, K, side, arm_i in raw:
        if t_in - last.get(tk, -1e9) < 300 or not 585 * 60 <= t_in <= 930 * 60:
            continue
        last[tk] = t_in
        ev.append((tk, t_in, i, K, side, arm_i))
    rows = []
    for tk, t_in, i, K, side, arm_i in ev:
        try:
            r = features(d, tk, t_in, i, K, side, arm_i, per_tk[tk], book, chains.get(tk) or [], gexapi, vol, prior, tide, raw)
        except Exception as e:                       # one bad event must not drop the day
            r = None
        if r is None:
            continue
        r.update(targets(per_tk[tk], t_in, K, side))
        r.update(money(tq, tk, K, side, t_in, per_tk[tk]))
        rows.append(r)
    os.makedirs(EV, exist_ok=True)
    json.dump(rows, open(f, "w"))
    return d, len(rows)


def _kt(sc):
    """The time each bar's MEDIAN becomes known: med[k] needs bar k+1, complete at sc[k+1] + 5 (bars can skip)."""
    return np.r_[sc[1:] + 5, np.inf].astype(float)


def _known_idx(sc, T):
    return max(int(np.searchsorted(_kt(sc), T, side="right")) - 1, 0)


def _known(sc, med, T):
    """Median value known at time T."""
    return float(med[_known_idx(sc, T)])


def features(d, tk, t_in, i, K, side, arm_i, pm, book, chain, gexapi, vol, prior, tide, raw):
    sc, med = pm
    p_in = _known(sc, med, t_in)
    p0 = float(med[0])
    bp = lambda x: x / p0 * 1e4
    m_last = int(t_in // 60) - 1                     # last CLOSED minute
    ki = _known_idx(sc, t_in)                         # last bar index whose median is known
    seen = med[:max(ki, 1) + 1]
    sc_seen = sc[:len(seen)]
    # approach
    t_arm = float(_kt(sc)[arm_i]) if arm_i is not None else t_in
    a_speed = bp(side * (p_in - float(med[arm_i]))) / max((t_in - t_arm) / 60.0, 0.1) if arm_i is not None else None
    a_accel = bp(side * (p_in - 2 * _known(sc, med, t_in - 60) + _known(sc, med, t_in - 120)))
    pre = seen[:arm_i] if arm_i is not None else seen
    sgn = np.sign(pre - K)
    sgn = sgn[sgn != 0]
    a_tests = int(np.sum(sgn[1:] != sgn[:-1])) if sgn.size > 1 else 0
    near = np.nonzero(np.abs(pre - K) <= 0.05)[0]
    a_since = (t_arm - float(sc_seen[near[-1]])) / 60.0 if near.size else 390.0
    a_fresh = 0 if np.any(side * (pre - K) >= ZB) else 1
    # strike-local
    def tot(k, cp, fld, m0, m1):
        c = book.get((tk, k, cp), {})
        return sum((c.get(m) or (0,) * 8)[fld] for m in range(m0, m1 + 1))
    def net(k, cp, m0, m1):
        return tot(k, cp, 2, m0, m1) - tot(k, cp, 3, m0, m1)
    six = [(K + dk, c) for dk in (-1, 0, 1) for c in ("C", "P")]
    v5 = sum(tot(k, c, 4, m_last - 4, m_last) for k, c in six)
    vb = sum(tot(k, c, 4, m_last - 19, m_last - 5) for k, c in six)
    s_surge = (v5 / 5) / (vb / 15 + 1) if v5 >= 50 else 0.0
    s_kcall = side * net(K, "C", m_last - 4, m_last)
    s_kput = -side * net(K, "P", m_last - 4, m_last)
    def oi(k):
        tot_ = 0
        for cp in ("C", "P"):
            c = book.get((tk, k, cp), {})
            v = next((c[m][7] for m in sorted(c) if c[m][7]), None)
            tot_ += v or 0
        return tot_
    ring = [oi(K + j) for j in range(-5, 6)]
    medoi = float(np.median([x for x in ring if x > 0])) if any(x > 0 for x in ring) else None
    s_oi_k = float(np.log((oi(K) + 1) / (medoi + 1))) if medoi else None
    s_oi_next = float(np.log((oi(K + side) + 1) / (medoi + 1))) if medoi else None
    fl = GF.flip_of(chain, p_in) if chain else None
    s_wall = None if not fl else float((side > 0 and fl["call_wall"] == K) or (side < 0 and fl["put_wall"] == K))
    s_flip = None if not fl or fl["flip"] is None else bp(side * (p_in - fl["flip"]))
    # quote state
    cw = "C" if side > 0 else "P"
    cq = book.get((tk, K, cw), {})
    lastq = cq.get(m_last) or cq.get(m_last - 1)
    q_spread = ((lastq[1] - lastq[0]) / ((lastq[0] + lastq[1]) / 2)) if lastq and lastq[0] and lastq[1] and lastq[1] > lastq[0] else None
    av, bv = tot(K, cw, 2, m_last - 2, m_last), tot(K, cw, 3, m_last - 2, m_last)
    q_imb = (av - bv) / (av + bv + 1)
    def strad(m):
        a, b = book.get((tk, K, "C"), {}).get(m), book.get((tk, K, "P"), {}).get(m)
        if a and b and a[0] and a[1] and b[0] and b[1]:
            return (a[0] + a[1] + b[0] + b[1]) / 2
        return None
    s1, s0 = strad(m_last), strad(m_last - 15)
    q_strad = (s1 / s0 - 1) if s1 and s0 else None
    # day context
    def pat(T):
        return _known(sc, med, T)
    k0 = int(round(pat(575 * 60)))
    c0, p0q = book.get((tk, k0, "C"), {}).get(574), book.get((tk, k0, "P"), {}).get(574)
    am = ((c0[0] + c0[1]) / 2 + (p0q[0] + p0q[1]) / 2) if c0 and p0q and c0[0] and p0q[0] else None
    c_am_used = abs(p_in - float(med[0])) / am if am else None
    mins = np.arange(570, m_last + 1)
    mclose = np.array([pat((m + 1) * 60) for m in mins]) if mins.size else np.array([p_in])
    c_path30 = float(np.sum(np.abs(np.diff(mclose[-31:])))) / am if am and mclose.size > 31 else None
    c_tr30 = bp(side * (mclose[-1] - mclose[-31])) if mclose.size > 31 else None
    vd = vol.get(f"{d}|{tk}") or {}
    mv = np.array([float(vd.get(str(m), 0)) for m in mins]) if mins.size else np.array([0.0])
    c_vwap = bp(side * (p_in - float(np.sum(mclose * mv) / np.sum(mv)))) if mv.sum() > 0 else None
    hi_, lo_ = float(seen.max()), float(seen.min())
    c_range = ((p_in - lo_) / (hi_ - lo_) if side > 0 else (hi_ - p_in) / (hi_ - lo_)) if hi_ > lo_ else None
    c_tod = t_in / 60.0 - 570
    g = _GEX.get((d, tk))
    if g is None:
        g = _GEX[(d, tk)] = L.gex_series(d, tk, gexapi)
    gm = m_last
    while gm >= 560 and gm not in g:
        gm -= 1
    c_gex = g.get(gm)
    cum = float(mv.sum())
    ref = []
    for x in prior:
        pv = vol.get(f"{x}|{tk}") or {}
        if pv:
            ref.append(sum(float(pv.get(str(m), 0)) for m in mins))
    c_rvol = cum / float(np.mean(ref)) if ref and np.mean(ref) > 0 else None
    tm = [m for m in tide if m <= m_last]
    c_tide = None
    if tm:
        m1 = max(tm); m0 = max([m for m in tm if m <= m1 - 15], default=None)
        if m0 is not None:
            c_tide = side * (tide[m1] - tide[m0])
    # flow (0DTE, this ticker)
    prem = dlt5 = dlt15 = 0.0
    for (t, k, cp), c in book.items():
        if t != tk:
            continue
        for m in range(m_last - 14, m_last + 1):
            r = c.get(m)
            if not r:
                continue
            b, a, av_, bv_, v, vw, de, _oi = r
            if vw:
                prem += (1 if cp == "C" else -1) * (av_ - bv_) * vw * 100
            if de is not None:
                dd = (av_ - bv_) * de * 100
                dlt15 += dd
                if m >= m_last - 4:
                    dlt5 += dd
    x_breach = sum(1 for t2, ti2, _i, _k, s2, _a in raw if t2 != tk and s2 == side and t_in - 60 <= ti2 <= t_in)
    return dict(d=d, t=tk, K=K, side=side, t_in=round(t_in, 1),
                a_speed=a_speed, a_accel=a_accel, a_tests=a_tests, a_since=a_since, a_fresh=a_fresh,
                s_surge=s_surge, s_kcall=s_kcall, s_kput=s_kput, s_oi_k=s_oi_k, s_oi_next=s_oi_next, s_wall=s_wall,
                s_flip=s_flip, q_spread=q_spread, q_imb=q_imb, q_strad=q_strad,
                c_am_used=c_am_used, c_path30=c_path30, c_tr30=c_tr30, c_vwap=c_vwap, c_range=c_range, c_tod=c_tod,
                c_gex=c_gex, c_rvol=c_rvol, c_tide=c_tide,
                f_prem15=side * prem, f_delta5=side * dlt5, f_delta15=side * dlt15, x_breach=x_breach)


def targets(pm, t_in, K, side):
    sc, med = pm
    kt = _kt(sc)
    j0 = int(np.searchsorted(kt, t_in, side="right"))           # first bar whose median becomes known after t_in
    y = 0
    for k in range(j0, len(sc)):
        if kt[k] > t_in + 900:
            break
        if side * (med[k] - K) >= 0.95:
            y = 1; break
        if side * (med[k] - K) <= -ZB:
            y = -1; break
    p = _known(sc, med, t_in)
    out = {"Y": y}
    for h in (2, 5, 10):
        out[f"M{h}"] = side * (_known(sc, med, t_in + 60 * h) - p) if t_in + 60 * h <= 57600 else None
    return out


def money(tq, tk, K, side, t_in, pm):
    """A and B1 sl0.35 at the user's fills, $ per contract (None if a quote is missing)."""
    sc, med = pm
    end = t_in + 900
    def q(c, T):
        s, b, a = c
        j = int(np.searchsorted(s, T, side="right")) - 1
        return None if j < 0 or T - s[j] > 30 else (b[j], a[j])
    kt = _kt(sc)
    def path_hit(cond):
        j0 = int(np.searchsorted(kt, t_in, side="right"))
        for k in range(j0, len(sc)):
            T = float(kt[k]) + LAT
            if T > end:
                return None
            if cond(med[k]):
                return T
        return None
    out = {"A": None, "B1": None}
    for name, cp, tp, slp, prem_sl in (
            ("A", "C" if side > 0 else "P", lambda p: side * (p - K) >= 0.95, lambda p: side * (p - K) <= -ZB, None),
            ("B1", "P" if side > 0 else "C", lambda p: side * (p - K) <= -0.20, None, 0.35)):
        c = tq.get((tk, K, cp))
        if c is None:
            continue
        e = q(c, t_in)
        if e is None or e[1] <= 0.05:
            continue
        emid = (e[0] + e[1]) / 2
        cands = []
        t = path_hit(tp)
        if t: cands.append(t)
        if slp:
            t = path_hit(slp)
            if t: cands.append(t)
        if prem_sl:
            s, b, a = c
            j0, j1 = int(np.searchsorted(s, t_in, side="right")), int(np.searchsorted(s, end, side="right"))
            hit = np.nonzero((b[j0:j1] + a[j0:j1]) / 2 <= emid - prem_sl)[0]
            if hit.size:
                cands.append(float(s[j0 + hit[0]]) + LAT)
        tx = min(cands) if cands else end
        x = q(c, tx)
        if x is None:
            continue
        out[name] = ((x[0] + x[1]) / 2 - SELL_C - (emid + BUY_C) - FEE_RT) * 100
    return out


# ------------------------------------------------------------------ scoring
def load(lo, hi):
    rows = []
    for f in sorted(glob.glob(os.path.join(EV, "*.json"))):
        d = os.path.basename(f)[:10]
        if lo <= d <= hi:
            rows += json.load(open(f))
    return rows


def ranker(disc):
    """Per ticker, per input: sorted discovery values -> empirical CDF."""
    cuts = {}
    for tk in TK:
        for k in INPUTS:
            v = np.array([r[k] for r in disc if r["t"] == tk and r[k] is not None], float)
            cuts[(tk, k)] = np.sort(v)
    def X(rows):
        out = np.full((len(rows), len(INPUTS)), 0.5)
        for i, r in enumerate(rows):
            for j, k in enumerate(INPUTS):
                v, c = r[k], cuts[(r["t"], k)]
                if v is not None and c.size:
                    out[i, j] = (np.searchsorted(c, v, side="left") + np.searchsorted(c, v, side="right")) / (2.0 * c.size)
        return out
    return X


def spearman(x, y):
    from scipy.stats import rankdata
    a, b = rankdata(x), rankdata(y)
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def day_boot(rows, stat, rng, n=N_BOOT):
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["d"]].append(i)
    ds = list(by)
    vals = []
    for _ in range(n):
        idx = [i for d in rng.choice(ds, len(ds)) for i in by[d]]
        vals.append(stat(np.array(idx)))
    return np.array(vals)


def bh(ps):
    ps = np.asarray(ps, float); n = len(ps); o = np.argsort(ps)
    q = np.empty(n); prev = 1.0
    for rank, i in reversed(list(enumerate(o, 1))):
        prev = min(prev, ps[i] * n / rank); q[i] = prev
    return q


def fit(disc):
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import GroupKFold
    X = ranker(disc)
    Xd = X(disc) - 0.5
    y = np.array([r["Y"] for r in disc], float)
    wk = np.array([dt.date.fromisoformat(r["d"]).isocalendar()[0] * 100 + dt.date.fromisoformat(r["d"]).isocalendar()[1]
                   for r in disc])
    cv = list(GroupKFold(n_splits=5).split(Xd, y, wk))
    m = LassoCV(cv=cv, random_state=0, n_alphas=60, max_iter=20000).fit(Xd, y)
    return X, m


def discovery():
    rng = np.random.default_rng(SEED)
    disc = load(*DISC)
    print(f"DISCOVERY {DISC[0]}..{DISC[1]}: {len(disc)} breaches, {len({r['d'] for r in disc})} sessions;"
          f" Y +1 {np.mean([r['Y'] == 1 for r in disc]):.0%} / -1 {np.mean([r['Y'] == -1 for r in disc]):.0%}"
          f" / 0 {np.mean([r['Y'] == 0 for r in disc]):.0%}")
    X = ranker(disc)
    Xd = X(disc)
    y = np.array([r["Y"] for r in disc], float)
    h1 = np.array([r["d"] <= HALF for r in disc])
    ps, rows = [], []
    for j, k in enumerate(INPUTS):
        ic = spearman(Xd[:, j], y)
        bs = day_boot(disc, lambda idx: spearman(Xd[idx, j], y[idx]), rng, 500)
        p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
        a, b = spearman(Xd[h1, j], y[h1]), spearman(Xd[~h1, j], y[~h1])
        ms = {h: spearman(Xd[:, j], np.array([r[f"M{h}"] if r[f"M{h}"] is not None else 0 for r in disc])) for h in (2, 5, 10)}
        cov = np.mean([r[k] is not None for r in disc])
        ps.append(p); rows.append((k, ic, p, a, b, ms, cov))
    qs = bh(ps)
    print(f"\n  P1 IC vs Y (Spearman, ranks within ticker)   [M2 / M5 / M10 ICs reported]")
    for (k, ic, p, a, b, ms, cov), q in zip(rows, qs):
        ok = q < 0.10 and np.sign(a) == np.sign(b) == np.sign(ic)
        print(f"    {k:10s} IC {ic:+.3f}  p {p:.3f}  q {q:.3f}  halves {a:+.3f} / {b:+.3f}  {'PASS' if ok else '    '}"
              f"  | M2 {ms[2]:+.3f} M5 {ms[5]:+.3f} M10 {ms[10]:+.3f}  coverage {cov:.0%}")
    X, m = fit(disc)
    coef = sorted([(abs(c), k, c) for k, c in zip(INPUTS, m.coef_) if abs(c) > 1e-9], reverse=True)
    print(f"\n  P2 LASSO (CV by week): alpha {m.alpha_:.5f}, keeps {len(coef)} of {len(INPUTS)}: "
          + ", ".join(f"{k} {c:+.3f}" for _a, k, c in coef))
    sd = m.predict(X(disc) - 0.5)
    print(f"    in-sample IC of the score {spearman(sd, y):+.3f}")
    return disc, X, m


def holdout():
    res = os.path.join(CACHE, "RESULT-holdout.txt")
    if os.path.exists(res):
        sys.exit(f"the holdout was scored already -- see {res}")
    rng = np.random.default_rng(SEED + 1)
    disc, X, m = discovery()
    hold = load(*HOLD)
    lines = [f"HOLDOUT {HOLD[0]}..{HOLD[1]} scored {dt.datetime.now():%Y-%m-%d %H:%M}: {len(hold)} breaches, "
             f"{len({r['d'] for r in hold})} sessions"]
    kept = [k for k, c in zip(INPUTS, m.coef_) if abs(c) > 1e-9]
    sh = m.predict(X(hold) - 0.5)
    y = np.array([r["Y"] for r in hold], float)
    ic = spearman(sh, y)
    bs = day_boot(hold, lambda idx: spearman(sh[idx], y[idx]), rng)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    p2 = bool(kept) and lo > 0
    lines.append(f"  P2 LASSO keeps {len(kept)} ({', '.join(kept) or '-'}): holdout IC {ic:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'PASS' if p2 else 'FAIL'}")
    sdisc = m.predict(X(disc) - 0.5)
    q20, q80 = np.quantile(sdisc, [0.2, 0.8])
    p3, ps = {}, []
    for name, sel in (("A on the TOP quintile", sh >= q80), ("B1 sl0.35 on the BOTTOM quintile", sh <= q20)):
        key = "A" if name.startswith("A") else "B1"
        rs = [r for r, s in zip(hold, sel) if s and r[key] is not None]
        allv = [r[key] for r in hold if r[key] is not None]
        if len(rs) < 30:
            lines.append(f"  P3 {name}: n={len(rs)} too few -> FAIL"); ps.append(1.0); p3[name] = (False, 0, 0, 0); continue
        v = np.array([r[key] for r in rs])
        bsm = day_boot(rs, lambda idx: v[idx].mean(), rng)
        l2, h2 = np.quantile(bsm, [0.025, 0.975])
        pv = min(1.0, 2 * min((bsm <= 0).mean(), (bsm >= 0).mean()))
        ps.append(pv); p3[name] = (v.mean(), l2, h2, len(rs), np.mean(allv))
    qs = bh(ps)
    any3 = False
    for (name, val), q in zip(p3.items(), qs):
        if len(val) < 5:
            continue
        mu, l2, h2, n, base = val
        ok = mu > 0 and l2 > 0 and q < 0.10
        any3 |= ok
        lines.append(f"  P3 {name}: n={n} mean ${mu:+.2f}/contract [{l2:+.2f}, {h2:+.2f}] q {q:.3f} (all holdout breaches ${base:+.2f})"
                     f" -> {'PASS' if ok else 'FAIL'}")
    lines.append(f"  OVERALL: {'PASS -> forward paper log' if p2 and any3 else 'FAIL'}")
    print("\n" + "\n".join(lines))
    open(res, "w").write("\n".join(lines) + "\n")


def fresh():
    disc = load(*DISC)
    X, m = fit(disc)
    fr = load(*FRESH)
    if not fr:
        sys.exit("no fresh events built")
    s = m.predict(X(fr) - 0.5)
    y = np.array([r["Y"] for r in fr], float)
    sd = m.predict(X(disc) - 0.5)
    q20, q80 = np.quantile(sd, [0.2, 0.8])
    print(f"FRESH {FRESH[0]}..{FRESH[1]} (descriptive): {len(fr)} breaches, IC {spearman(s, y):+.3f}")
    for key, sel in (("A", s >= q80), ("B1", s <= q20)):
        v = [r[key] for r, k in zip(fr, sel) if k and r[key] is not None]
        print(f"  {key}: n={len(v)} mean ${np.mean(v) if v else float('nan'):+.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("tide", "build", "discovery", "holdout", "fresh"):
        ap.add_argument(f"--{a}", action="store_true")
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    if a.tide:
        pull_tide()
    if a.build:
        days = [d for d in L.lake_days(DISC[0], FRESH[1])
                if glob.glob(os.path.join(L.LAKE, "silver", "trades-core", f"date={d}", "*.parquet"))
                or os.path.exists(os.path.join(L.LAKE, "bronze", "full-tape", f"{d}.parquet"))]
        if a.days:
            days = days[: a.days]
        with ProcessPoolExecutor(6) as ex:
            for k, (d, msg) in enumerate(ex.map(build_day, days)):
                if k % 25 == 0:
                    print(k, d, msg, flush=True)
        print("built", len(days))
    if a.discovery:
        discovery()
    if a.holdout:
        holdout()
    if a.fresh:
        fresh()
