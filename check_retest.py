"""
check_retest.py -- TRADE THE RETEST, NOT THE BREAK (user, 2026-11-17). PRE-REGISTERED 2026-11-17, committed before any
retest event was computed.

ORIGIN. 2026-11-16 (a selloff): THRU flagged SPY down through 748 at 09:48:55, it snapped back within 20 s, retested
~748.8 under 749 and then the day fell to 739. Across that day's breaks, most snapped back through the strike before
the trend resumed (down-breaks in the top two score quintiles: 7 of 31 reached the next strike, 23 fell back). The
user's read: take the retest in the trend's direction, not the first break. The breach screen (check_breach_ic) found
trend / Tide inputs ranked first breaks not at all (c_tr30 IC -0.008, c_tide -0.003) -- the open question is whether
they matter for the RETEST.

EVENT (5-second median-of-3 path, check_breach_ic's breach and look-ahead rules; SPY / QQQ / IWM, breaches 09:45-15:30)
  1 BREACH      check_breach_ic's event: price reaches K + side x $0.10 from >= 5 bp beyond (side -1 = down-break).
  2 SNAP-BACK   within 10 minutes of the breach, price comes back through K by >= $0.10 (side x (p - K) <= -0.10).
  3 VOID        if price reaches the PREVIOUS strike (side x (p - K) <= -1.00) before the entry, or no entry comes
                within 15 minutes of the snap-back: no trade.
  4 ENTRY  E1 (PRIMARY) the SECOND BREAK: price goes through K again by $0.10 in the breach direction
                (side x (p - K) >= +0.10). Entry 1 s after that median is known.
           E2 (reported) the RETEST TURN: the retest extreme X (the furthest point back through K since the
                snap-back) is within $0.75 of K, and price comes $0.15 off X toward K. Entry 1 s after.
  5 TRADE       the ATM K contract in the breach direction (put on a down-break). TP: the next strike in the breach
                direction - $0.05 (side x (p - K) >= 0.95). STOP: back beyond the retest extreme by $0.05
                (side x (p - K) <= side x (X - K) - 0.05). TIME: 30 minutes. Priced on the tape (NBBO on the contract's
                own prints) at the user's measured fills: buy mid + $0.06, sell mid - $0.08, $0.10 fees per contract.
                One retest trade per breach; overlapping trades on a ticker are all counted (reported).
CONDITIONS (stamped at the ENTRY, closed minutes only)
  TIDE   aligned = UW Market Tide (net call - net put premium) 15-minute change has the breach's sign.
  TREND  aligned = the 30-minute price change has the breach's sign AND price is on the breach side of VWAP.
  K-FLOW the with-direction K contract (put on a down-break, call on an up-break), net SOLD (bid - ask volume) from
         the breach minute to the last closed minute before entry -- the user's 748-puts-sold question. Hedge
         logic: put selling (to open or to close) makes dealers BUY stock -> fuels the snap-back; once spent, the
         re-break should run. Predicted: more selling -> better retest trades.
HYPOTHESES
  H1 (money)  E1 mean P&L per contract > 0, day-block bootstrap 95% CI above 0.
  H2 (Tide)   E1 P&L aligned minus not-aligned > 0, CI above 0.
  H3 (trend)  E1 P&L aligned minus not-aligned > 0, CI above 0.
  H4 (K-flow) Spearman(K-flow sold, E1 P&L) > 0, CI above 0.
SAMPLES
  SIZING     2024-08-20 .. 2026-09-25 (the lake, already mined for strike behaviour): reported, labelled
             exploratory, decides nothing.
  CONFIRM    2026-09-26 .. 2026-11-13 (34 sessions no breach or retest study has looked at): H1-H4 scored ONCE
             (--confirm writes _retest_cache/RESULT-confirm.txt and refuses to run again). Power is low and said
             now: an inconclusive H1 is a FAIL; H2-H4 are read for direction.
  FORWARD    from 2026-11-16 with each lake refresh; a confirmed H1 goes to a paper log, nothing deployed.

Usage:  python check_retest.py --build | --sizing | --confirm
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import check_breach_ic as B                        # noqa: E402
import check_sixpack_lake as L                     # noqa: E402

CACHE = os.path.join(ROOT, "_retest_cache")
EV = os.path.join(CACHE, "events")
SIZING = ("2024-08-20", "2026-09-25")
CONFIRM = ("2026-09-26", "2026-11-13")
SNAP_S, ENTRY_S, HOLD_S = 600, 900, 1800
E2_ZONE, E2_TURN = 0.75, 0.15
SEED = 20261117


def minute_closes(d, tk):
    """{minute: (close, volume)} regular-session underlying bars from historical/{T}.parquet."""
    x = (pl.scan_parquet(os.path.join(ROOT, "historical", f"{tk}.parquet")).select("date", "minute_et", "close", "volume")
         .filter(pl.col("date").cast(pl.Utf8) == d).collect())
    out = {}
    for me, c, v in x.select("minute_et", "close", "volume").iter_rows():
        m = me.hour * 60 + me.minute
        if 570 <= m < 960:
            out[m] = (c, v or 0)
    return out


def k_flow(d, tk):
    f = os.path.join(L.LAKE, "silver", "option-contracts-1m", f"date={d}", "bars.parquet")
    x = (pl.scan_parquet(f).filter((pl.col("underlying_symbol") == tk) & (pl.col("expiry").cast(pl.Utf8) == d))
         .select(pl.col("strike").cast(pl.Float64), "option_type", "minute_et", "ask_volume", "bid_volume").collect())
    out = {}
    for k, ot, me, a, b in x.iter_rows():
        if k is None or abs(k - round(k)) > 1e-6:
            continue
        out.setdefault((int(round(k)), ot[0].upper()), {})[me.hour * 60 + me.minute] = (a or 0) - (b or 0)
    return out


def build_day(d):
    f = os.path.join(EV, d + ".json")
    if os.path.exists(f):
        return d, "have"
    try:
        path, tq, tide = L.price_path(d), B.tape(d), B.tide_series(d)
    except Exception as e:
        return d, f"skip {type(e).__name__}"
    rows = []
    for tk, (sc, hi, lo, cl) in path.items():
        if sc[-1] < 57300 or len(sc) < 100:
            continue
        med = np.median(np.vstack([np.r_[cl[:1], cl[:-1]], cl, np.r_[cl[1:], cl[-1:]]]), axis=0)
        kt = B._kt(sc)
        p0 = float(med[0]); Rr = p0 * B.RBP / 1e4
        raw = []
        for K in range(int(np.floor(med.min())), int(np.ceil(med.max())) + 1):
            armed = 0
            for i in range(len(med)):
                if med[i] <= K - B.ZB - Rr: armed = 1
                elif med[i] >= K + B.ZB + Rr: armed = -1
                if armed == 1 and med[i] >= K + B.ZB: side = 1
                elif armed == -1 and med[i] <= K - B.ZB: side = -1
                else: continue
                armed = 0
                if i + 1 < len(sc):
                    raw.append((float(kt[i]) + B.LAT, i, K, side))
        raw.sort()
        last, mins, kf = -1e9, None, None
        for t_br, i, K, side in raw:
            if t_br - last < 300 or not 585 * 60 <= t_br <= 930 * 60:
                continue
            last = t_br
            if mins is None:
                mins, kf = minute_closes(d, tk), k_flow(d, tk)
            rows += retest(d, tk, K, side, t_br, sc, med, kt, tq, tide, mins, kf)
    os.makedirs(EV, exist_ok=True)
    json.dump(rows, open(f, "w"))
    return d, len(rows)


def retest(d, tk, K, side, t_br, sc, med, kt, tq, tide, mins, kf):
    j = int(np.searchsorted(kt, t_br, side="right"))
    x = lambda k: side * (med[k] - K)
    t_snap = None
    while j < len(sc) and kt[j] <= t_br + SNAP_S:
        if x(j) <= -0.10:
            t_snap = kt[j]; break
        j += 1
    if t_snap is None:
        return []
    out = []
    ext = x(j)                                   # the retest extreme so far (most negative side x (p - K))
    e2 = None
    k = j + 1
    while k < len(sc) and kt[k] <= t_snap + ENTRY_S:
        v = x(k)
        if v <= -1.0:
            return out                           # void: back to the previous strike
        if v < ext:
            ext = v
        if e2 is None and ext >= -E2_ZONE and v - ext >= E2_TURN and v < 0.10:
            e2 = (float(kt[k]) + B.LAT, ext)
        if v >= 0.10:
            pairs = [("E1", (float(kt[k]) + B.LAT, ext))]
            if e2:
                pairs.append(("E2", e2))
            for name, (t_in, X) in pairs:
                out.append(trade(d, tk, K, side, name, t_in, X, t_br, sc, med, kt, tq, tide, mins, kf))
            return [o for o in out if o]
        k += 1
    if e2:                                       # E2 fired but no second break came: still an E2 trade
        o = trade(d, tk, K, side, "E2", e2[0], e2[1], t_br, sc, med, kt, tq, tide, mins, kf)
        return [o] if o else []
    return []


def trade(d, tk, K, side, name, t_in, X, t_br, sc, med, kt, tq, tide, mins, kf):
    cp = "C" if side > 0 else "P"
    c = tq.get((tk, K, cp))
    if c is None:
        return None
    def q(T):
        s, b, a = c
        i = int(np.searchsorted(s, T, side="right")) - 1
        return None if i < 0 or T - s[i] > 30 else (b[i], a[i])
    e = q(t_in)
    if e is None or e[1] <= 0.05:
        return None
    stop = X - 0.05
    tx, why = t_in + HOLD_S, "time"
    for k in range(int(np.searchsorted(kt, t_in, side="right")), len(sc)):
        T = float(kt[k]) + B.LAT
        if T > t_in + HOLD_S:
            break
        v = side * (med[k] - K)
        if v >= 0.95: tx, why = T, "tp"; break
        if v <= stop: tx, why = T, "stop"; break
    xq = q(tx)
    if xq is None:
        return None
    pnl = ((xq[0] + xq[1]) / 2 - B.SELL_C - ((e[0] + e[1]) / 2 + B.BUY_C) - B.FEE_RT) * 100
    m_last = int(t_in // 60) - 1
    tm = [m for m in tide if m <= m_last]
    tide_al = None
    if tm:
        m1 = max(tm); m0 = max([m for m in tm if m <= m1 - 15], default=None)
        if m0 is not None:
            tide_al = side * (tide[m1] - tide[m0]) > 0
    cl = [mins[m][0] for m in range(m_last - 30, m_last + 1) if m in mins]
    vw = [(mins[m][0], mins[m][1]) for m in range(570, m_last + 1) if m in mins]
    trend_al = None
    if len(cl) >= 25 and vw and sum(v for _p, v in vw) > 0:
        vwap = sum(p * v for p, v in vw) / sum(v for _p, v in vw)
        p_now = B._known(sc, med, t_in)
        trend_al = side * (cl[-1] - cl[0]) > 0 and side * (p_now - vwap) > 0
    fl = kf.get((K, cp), {})
    m_br = int(t_br // 60)
    sold = -sum(fl.get(m, 0) for m in range(m_br, m_last + 1))      # net SOLD (bid - ask) of the with-direction K contract
    return dict(d=d, t=tk, K=K, side=side, entry=name, t_in=round(t_in, 1), X=round(float(X), 3), why=why,
                pnl=round(pnl, 2), tide_al=tide_al, trend_al=trend_al, k_sold=sold)


def load(lo, hi):
    return [r for f in sorted(glob.glob(os.path.join(EV, "*.json"))) if lo <= os.path.basename(f)[:10] <= hi
            for r in json.load(open(f))]


def report(rows, title, scored=False):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(SEED)
    lines = [f"{title}: {len({r['d'] for r in rows})} sessions"]
    def boot(rs, fn, n=2000):
        by = {}
        for i, r in enumerate(rs):
            by.setdefault(r["d"], []).append(i)
        ds = list(by); v = []
        for _ in range(n):
            idx = [i for d in rng.choice(ds, len(ds)) for i in by[d]]
            v.append(fn(np.array(idx)))
        v = np.array([x for x in v if np.isfinite(x)])
        return np.quantile(v, [0.025, 0.975]) if len(v) else (np.nan, np.nan)
    res = {}
    for e in ("E1", "E2"):
        rs = [r for r in rows if r["entry"] == e]
        if len(rs) < 10:
            lines.append(f"  {e}: n={len(rs)} too few"); continue
        p = np.array([r["pnl"] for r in rs])
        lo, hi = boot(rs, lambda i: p[i].mean())
        why = {w: np.mean([r["why"] == w for r in rs]) for w in ("tp", "stop", "time")}
        lines.append(f"  {e}: n={len(rs)} mean ${p.mean():+.2f}/contract [{lo:+.2f}, {hi:+.2f}] median ${np.median(p):+.2f} win {np.mean(p > 0):.0%}"
                     f" | TP {why['tp']:.0%} stop {why['stop']:.0%} time {why['time']:.0%}")
        res[(e, "H1")] = (p.mean(), lo, hi)
        for key, lab in (("tide_al", "Tide"), ("trend_al", "trend")):
            a = np.array([r[key] is True for r in rs]); b = np.array([r[key] is False for r in rs])
            if a.sum() >= 5 and b.sum() >= 5:
                lo2, hi2 = boot(rs, lambda i: p[i][a[i]].mean() - p[i][b[i]].mean() if a[i].any() and b[i].any() else np.nan)
                lines.append(f"     {lab:5s} aligned n={a.sum()} ${p[a].mean():+.2f} vs not n={b.sum()} ${p[b].mean():+.2f}: "
                             f"diff {p[a].mean() - p[b].mean():+.2f} [{lo2:+.2f}, {hi2:+.2f}]")
                res[(e, lab)] = (p[a].mean() - p[b].mean(), lo2, hi2)
        ks = np.array([r["k_sold"] for r in rs], float)
        rho = spearmanr(ks, p)[0]
        lo3, hi3 = boot(rs, lambda i: spearmanr(ks[i], p[i])[0])
        lines.append(f"     K-flow sold vs P&L: Spearman {rho:+.3f} [{lo3:+.3f}, {hi3:+.3f}]")
        res[(e, "kflow")] = (rho, lo3, hi3)
        for tk in ("SPY", "QQQ", "IWM"):
            x = [r["pnl"] for r in rs if r["t"] == tk]
            if x: lines.append(f"     {tk}: n={len(x)} ${np.mean(x):+.2f}")
    if scored:
        v = lambda k: res.get(k, (np.nan, np.nan, np.nan))
        h1 = v(("E1", "H1")); h2 = v(("E1", "Tide")); h3 = v(("E1", "trend")); h4 = v(("E1", "kflow"))
        lines.append(f"  H1 money  E1 mean {h1[0]:+.2f} [{h1[1]:+.2f}, {h1[2]:+.2f}] -> {'PASS' if h1[1] > 0 else 'FAIL'}")
        for lab, h in (("H2 Tide", h2), ("H3 trend", h3), ("H4 K-flow", h4)):
            lines.append(f"  {lab:9s} {h[0]:+.3f} [{h[1]:+.3f}, {h[2]:+.3f}] -> {'PASS' if h[1] > 0 else 'FAIL'}")
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("build", "sizing", "confirm"):
        ap.add_argument(f"--{a}", action="store_true")
    a = ap.parse_args()
    if a.build:
        B.pull_tide_range(SIZING[0], CONFIRM[1])
        days = [d for d in L.lake_days(SIZING[0], CONFIRM[1])
                if glob.glob(os.path.join(L.LAKE, "silver", "trades-core", f"date={d}", "*.parquet"))]
        with ProcessPoolExecutor(6) as ex:
            for k, (d, msg) in enumerate(ex.map(build_day, days)):
                if k % 50 == 0:
                    print(k, d, msg, flush=True)
        print("built", len(days))
    if a.sizing:
        print("\n".join(report(load(*SIZING), f"SIZING (exploratory) {SIZING[0]}..{SIZING[1]}")))
    if a.confirm:
        res = os.path.join(CACHE, "RESULT-confirm.txt")
        if os.path.exists(res):
            sys.exit(f"the confirmation window was scored already -- see {res}")
        lines = report(load(*CONFIRM), f"CONFIRM {CONFIRM[0]}..{CONFIRM[1]} scored {dt.datetime.now():%Y-%m-%d %H:%M}", scored=True)
        print("\n".join(lines))
        open(res, "w").write("\n".join(lines) + "\n")
