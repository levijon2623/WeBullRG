"""
check_structure.py -- FAILED BREAKOUT -> STRUCTURE SHIFT (user, 2026-11-17). PRE-REGISTERED 2026-11-17, committed before
any setup was computed.

ORIGIN. The user's read of SPY 2026-11-16 09:30-09:50: higher low, higher high (an attempted breakout), rejection, a
LOWER LOW under the last higher low, then a bounce that stalled as a LOWER HIGH (~748.8) -> short; the day fell to 739.
check_retest (strike retests, no structure condition) was break-even; this asks whether the STRUCTURE adds anything.

SWINGS   zigzag on 1-minute closes (the 5-second median-of-3 path's last value in each minute, known 10 s after the
         minute ends), reversal threshold R = 0.10% of the day's first price (SPY ~$0.75, QQQ ~$0.67, IWM ~$0.28).
         A pivot is confirmed once price has moved R back from it.
SETUP (bearish; bullish is the same on the negated price)
         pivots ... La, Hb, Lc, Hd, Le with Lc > La (higher low) and Hd > Hb (higher high) = the attempted breakout;
         Le < Lc (a LOWER LOW under the last higher low) = the failure, confirmed when price has bounced R off Le.
         From that confirmation the bounce's running high M is tracked on the 5-second path; VOID if M reaches Hd
         (not a lower high) or no entry within 30 minutes.
ENTRY    price comes 0.02% of price (SPY ~$0.15) off M, while M < Hd. 1 s after the median is known. SHORT for a
         bearish setup, LONG for a bullish one. STOP = M + 0.0067% of price (SPY ~$0.05). risk = stop - entry.
CONTROLS PB "plain pullback": the identical lower-low -> lower-high entry when the pivots before it did NOT form a higher
         low and higher high (no failed breakout). RAND: for each setup, 3 random entry times the same day and ticker
         (09:35-15:30, >= 10 min from the setup), same direction, same risk in % of price.
PRIMARY (target-free) P(price reaches entry - 2 x risk before the stop, within 30 minutes) -- "2R before 1R".
         PASS (holdout) = setup minus PB > 0 AND setup minus RAND > 0, both with day-block bootstrap 95% CIs above 0.
         Reported: the median favourable excursion before the stop, in R; first hour (entries before 10:30) apart.
MONEY    the ATM option (strike nearest the entry price) in the trade's direction, tape-priced at the user's measured
         fills (buy mid + $0.06, sell mid - $0.08, $0.10 fees per contract), exit at the stop, the target, or 30 min.
         TARGET MENU, chosen on DISCOVERY only by mean P&L: T1 the next strike (the first whole dollar >= $0.25 beyond
         the entry), a nickel before it; T2 2R; T3 3R; T4 the user's adaptive rule -- the strike BEYOND the next if the
         drop into the lower low was fast (its speed in % of price per minute above the discovery median), the next
         strike if slow. The chosen target is scored ONCE on the holdout: PASS = mean P&L > 0 with the CI above 0.
CONTEXT  reported, not scored: Tide (15-min change agrees with the trade) and the 30-minute trend agreeing.
SAMPLES  DISCOVERY 2024-08-20 .. 2025-12-31 (target choice, everything reported). HOLDOUT 2026-01-02 .. 2026-11-13,
         scored once (--holdout writes _structure_cache/RESULT-holdout.txt and refuses to run again). 2026-11-16 (the
         origin) and later are forward only. Entries 09:35-15:30, SPY / QQQ / IWM.

RUN LOG
  2026-11-17 (552 sessions built). 2R before 1R within 30 min:
    DISCOVERY  failed breakout 42.5% (n 2,439) | plain pullback 41.4% (n 7,701) | random same-day 37.2%
               setup - pullback +1.1 pts [-1.4, +3.4]; setup - random +5.3 [+2.8, +7.8]; target chosen T2 (2R)
               money $/contract: T1 +0.36, T2 +1.32, T3 +0.85, T4 (fast/slow) +0.32
    HOLDOUT    failed breakout 43.0% (n 1,424) | plain pullback 41.1% | random 35.7%
               setup - pullback +1.9 [-1.0, +4.6]; setup - random +7.3 [+4.2, +10.6]
               T2 mean +$0.60 [-0.76, +2.11]; (T1 +0.75, T3 +1.26, T4 +0.59)
    PRIMARY FAIL (beats random, NOT the plain pullback); MONEY FAIL.
    READ: a lower-low -> lower-high pullback entry with a tight stop does beat a random entry by ~5-7 points of
    "2R first" -- with or without the failed breakout before it; the failed-breakout part adds ~1-2 points (noise).
    In option money it is worth about +$0.5-1.3 a contract, inside zero. Tide / 30-min trend: no difference.
    First hour: 39.9% (discovery) vs 47.8% (holdout, n 230) -- inconsistent.

Usage:  python check_structure.py --build | --discovery | --holdout
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

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import check_breach_ic as B                        # noqa: E402
import check_sixpack_lake as L                     # noqa: E402

CACHE = os.path.join(ROOT, "_structure_cache")
EV = os.path.join(CACHE, "events")
DISC = ("2024-08-20", "2025-12-31")
HOLD = ("2026-01-02", "2026-11-13")
R_FRAC, TURN_FRAC, BUF_FRAC = 0.0010, 0.0002, 0.0000667
WAIT_S, HOLD_S = 1800, 1800
T_LO, T_HI = 575 * 60, 930 * 60
SEED = 20261117


def minute_closes(sc, med):
    """{minute: (value, known_at)} from the 5-s median path."""
    out = {}
    for s, v in zip(sc, med):
        m = int(s // 60)
        out[m] = v
    return {m: (v, (m + 1) * 60 + 10) for m, v in out.items() if 570 <= m < 960}


def zigzag(mc, R):
    """confirmed pivots [(type 'H'/'L', value, minute of extreme, confirmed_at)] in time order, alternating."""
    ms = sorted(mc)
    if not ms:
        return []
    piv, d = [], 0
    hi_v = lo_v = mc[ms[0]][0]
    hi_m = lo_m = ms[0]
    for m in ms[1:]:
        v, known = mc[m]
        if d == 0:
            if v > hi_v: hi_v, hi_m = v, m
            if v < lo_v: lo_v, lo_m = v, m
            if hi_v - v >= R:
                piv.append(("H", hi_v, hi_m, known)); d, lo_v, lo_m = -1, v, m
            elif v - lo_v >= R:
                piv.append(("L", lo_v, lo_m, known)); d, hi_v, hi_m = 1, v, m
        elif d == 1:
            if v > hi_v:
                hi_v, hi_m = v, m
            elif hi_v - v >= R:
                piv.append(("H", hi_v, hi_m, known)); d, lo_v, lo_m = -1, v, m
        else:
            if v < lo_v:
                lo_v, lo_m = v, m
            elif v - lo_v >= R:
                piv.append(("L", lo_v, lo_m, known)); d, hi_v, hi_m = 1, v, m
    return piv


def setups(sc, med, kt, R, p0, orient):
    """bearish setups on z = orient x price. -> list of dicts (structure 'fb' / 'pb', entry time, z values)."""
    z = orient * med
    mc = minute_closes(sc, z)
    piv = zigzag(mc, R)
    out = []
    for i in range(len(piv)):
        if piv[i][0] != "L" or i < 2:
            continue
        Le, Hd, Lc = piv[i], piv[i - 1], piv[i - 2]
        if Hd[0] != "H" or Lc[0] != "L" or not Le[1] < Lc[1]:
            continue
        fb = i >= 4 and piv[i - 3][0] == "H" and piv[i - 4][0] == "L" and Lc[1] > piv[i - 4][1] and Hd[1] > piv[i - 3][1]
        Tc = Le[3]
        if not T_LO - 900 <= Tc <= T_HI:
            continue
        k0 = int(np.searchsorted(sc, Le[2] * 60))           # from the low's minute
        j = int(np.searchsorted(kt, Tc, side="left"))
        if j >= len(sc):
            continue
        M = float(np.max(z[k0:j + 1])) if j >= k0 else float(z[j])
        ent = None
        for k in range(j, len(sc)):
            if kt[k] > Tc + WAIT_S:
                break
            M = max(M, float(z[k]))
            if M >= Hd[1]:
                break
            if M - z[k] >= TURN_FRAC * p0:
                ent = (float(kt[k]) + B.LAT, float(z[k]), M)
                break
        if ent is None or not T_LO <= ent[0] <= T_HI:
            continue
        # the drop into the lower low: from Hd to Le, % of price per minute
        speed = (Hd[1] - Le[1]) / p0 / max(Le[2] - Hd[2], 1)
        out.append(dict(struct="fb" if fb else "pb", t_in=ent[0], z_e=ent[1], M=ent[2], z_s=ent[2] + BUF_FRAC * p0,
                        Hd=Hd[1], Le=Le[1], speed=speed))
    return out


def outcome(sc, med, kt, orient, t_in, z_e, z_s):
    z = orient * med
    risk = z_s - z_e
    if risk <= 0:
        return None
    best, hit2, stopped = 0.0, False, False
    for k in range(int(np.searchsorted(kt, t_in, side="right")), len(sc)):
        if kt[k] + B.LAT > t_in + HOLD_S:
            break
        if z[k] >= z_s:
            stopped = True
            break
        best = max(best, z_e - z[k])
        if z_e - z[k] >= 2 * risk:
            hit2 = True
    return dict(hit2=hit2, mfe_r=best / risk, risk=risk, stopped=stopped)


def money(tq, tk, orient, t_in, z_e, z_s, sc, med, kt, target):
    """option P&L ($/contract) at the user's fills for one target rule; z = orient x price."""
    p_e = orient * z_e
    K = int(round(p_e))
    cp = "P" if orient > 0 else "C"
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
    z = orient * med
    risk = z_s - z_e
    nxt = (np.floor(p_e - 0.25) if orient > 0 else np.ceil(p_e + 0.25))
    if target == "T1": zt = orient * nxt + 0.05
    elif target == "T2": zt = z_e - 2 * risk
    elif target == "T3": zt = z_e - 3 * risk
    else: zt = orient * (nxt - orient) + 0.05          # the strike beyond (T4 fast)
    tx = t_in + HOLD_S
    for k in range(int(np.searchsorted(kt, t_in, side="right")), len(sc)):
        T = float(kt[k]) + B.LAT
        if T > t_in + HOLD_S:
            break
        if z[k] >= z_s or z[k] <= zt:
            tx = T
            break
    x = q(tx)
    if x is None:
        return None
    return ((x[0] + x[1]) / 2 - B.SELL_C - ((e[0] + e[1]) / 2 + B.BUY_C) - B.FEE_RT) * 100


def build_day(d):
    f = os.path.join(EV, d + ".json")
    if os.path.exists(f):
        return d, "have"
    try:
        path, tq, tide = L.price_path(d), B.tape(d), B.tide_series(d)
    except Exception as e:
        return d, f"skip {type(e).__name__}"
    rng = np.random.default_rng(abs(hash(d)) % (2 ** 32))
    rows = []
    for tk, (sc, hi, lo, cl) in path.items():
        if sc[-1] < 57300 or len(sc) < 200:
            continue
        med = np.median(np.vstack([np.r_[cl[:1], cl[:-1]], cl, np.r_[cl[1:], cl[-1:]]]), axis=0)
        kt = B._kt(sc)
        p0 = float(med[0])
        for orient in (1, -1):
            for s in setups(sc, med, kt, R_FRAC * p0, p0, orient):
                o = outcome(sc, med, kt, orient, s["t_in"], s["z_e"], s["z_s"])
                if o is None:
                    continue
                r = dict(d=d, t=tk, side=-orient, struct=s["struct"], t_in=round(s["t_in"], 1), speed=s["speed"],
                         risk_frac=o["risk"] / p0, **{k: v for k, v in o.items() if k != "risk"})
                for tg in ("T1", "T2", "T3", "T4"):
                    r[tg] = money(tq, tk, orient, s["t_in"], s["z_e"], s["z_s"], sc, med, kt, tg)
                # random controls: same day / ticker / direction / risk in % of price
                ctl = []
                for _ in range(3):
                    for _try in range(20):
                        t = float(rng.uniform(T_LO, T_HI))
                        if abs(t - s["t_in"]) >= 600:
                            break
                    j = int(np.searchsorted(kt, t, side="right")) - 1
                    if j < 0:
                        continue
                    ze = float(orient * med[j])
                    oc = outcome(sc, med, kt, orient, t, ze, ze + o["risk"])
                    if oc:
                        ctl.append(oc["hit2"])
                r["rand_hit2"] = float(np.mean(ctl)) if ctl else None
                m_last = int(s["t_in"] // 60) - 1
                tm = [m for m in tide if m <= m_last]
                r["tide_al"] = None
                if tm:
                    m1 = max(tm); m0 = max([m for m in tm if m <= m1 - 15], default=None)
                    if m0 is not None:
                        r["tide_al"] = (-orient) * (tide[m1] - tide[m0]) > 0
                j30 = int(np.searchsorted(kt, s["t_in"] - 1800, side="right")) - 1
                jn = int(np.searchsorted(kt, s["t_in"], side="right")) - 1
                r["trend_al"] = None if j30 < 0 else bool((-orient) * (med[jn] - med[j30]) > 0)
                rows.append(r)
    os.makedirs(EV, exist_ok=True)
    json.dump(rows, open(f, "w"))
    return d, len(rows)


def load(lo, hi):
    return [r for f in sorted(glob.glob(os.path.join(EV, "*.json"))) if lo <= os.path.basename(f)[:10] <= hi
            for r in json.load(open(f))]


def boot(rows, fn, rng, n=2000):
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["d"], []).append(i)
    ds = list(by)
    v = []
    for _ in range(n):
        idx = np.array([i for d in rng.choice(ds, len(ds)) for i in by[d]])
        x = fn(idx)
        if np.isfinite(x):
            v.append(x)
    return np.quantile(v, [0.025, 0.975]) if v else (np.nan, np.nan)


def report(rows, title, med_speed, target=None, scored=False):
    rng = np.random.default_rng(SEED)
    fb = [r for r in rows if r["struct"] == "fb"]
    pb = [r for r in rows if r["struct"] == "pb"]
    lines = [f"{title}: {len({r['d'] for r in rows})} sessions; failed-breakout setups {len(fb)}, plain pullbacks {len(pb)}"]
    h = lambda rs: np.mean([r["hit2"] for r in rs]) if rs else np.nan
    allr = fb + pb
    isfb = np.array([r["struct"] == "fb" for r in allr]); hit = np.array([r["hit2"] for r in allr], float)
    rnd = np.array([r["rand_hit2"] if r["rand_hit2"] is not None else np.nan for r in allr], float)
    d_pb = hit[isfb].mean() - hit[~isfb].mean()
    lo1, hi1 = boot(allr, lambda i: hit[i][isfb[i]].mean() - hit[i][~isfb[i]].mean() if isfb[i].any() and (~isfb[i]).any() else np.nan, rng)
    fbi = np.nonzero(isfb)[0]
    d_rand = np.nanmean(hit[fbi] - rnd[fbi])
    lo2, hi2 = boot(fb, lambda i: np.nanmean(np.array([r["hit2"] for r in fb], float)[i] - np.array([r["rand_hit2"] if r["rand_hit2"] is not None else np.nan for r in fb], float)[i]), rng)
    lines.append(f"  2R before 1R: failed breakout {h(fb):.1%} | plain pullback {h(pb):.1%} | random same-day {np.nanmean(rnd[fbi]):.1%}")
    lines.append(f"    setup - pullback {d_pb * 100:+.1f} pts [{lo1 * 100:+.1f}, {hi1 * 100:+.1f}]   setup - random {d_rand * 100:+.1f} pts [{lo2 * 100:+.1f}, {hi2 * 100:+.1f}]")
    lines.append(f"    median favourable excursion before the stop: setup {np.median([r['mfe_r'] for r in fb]):.2f} R, pullback {np.median([r['mfe_r'] for r in pb]):.2f} R")
    first = [r for r in fb if r["t_in"] < 630 * 60]
    lines.append(f"    first hour (entry < 10:30): n={len(first)} 2R {h(first):.1%}")
    for key, lab in (("tide_al", "Tide agrees"), ("trend_al", "30-min trend agrees")):
        a = [r for r in fb if r[key] is True]; b = [r for r in fb if r[key] is False]
        lines.append(f"    {lab}: {h(a):.1%} (n={len(a)}) vs not {h(b):.1%} (n={len(b)})")
    for tg in ("T1", "T2", "T3", "T4"):
        if tg == "T4":
            v = [(r["T4"] if r["speed"] > med_speed else r["T1"]) for r in fb]
        else:
            v = [r[tg] for r in fb]
        v = [x for x in v if x is not None]
        lines.append(f"    money {tg}: n={len(v)} mean ${np.mean(v):+.2f}/contract median ${np.median(v):+.2f} win {np.mean(np.array(v) > 0):.0%}")
    res = dict(d_pb=(d_pb, lo1, hi1), d_rand=(d_rand, lo2, hi2))
    if target:
        sel = [r for r in fb if (r["T4"] if (target == "T4" and r["speed"] > med_speed) else r["T1" if target == "T4" else target]) is not None]
        pv = np.array([(r["T4"] if r["speed"] > med_speed else r["T1"]) if target == "T4" else r[target] for r in sel], float)
        lo3, hi3 = boot(sel, lambda i: pv[i].mean(), rng)
        lines.append(f"  TARGET {target} (chosen on discovery): mean ${pv.mean():+.2f} [{lo3:+.2f}, {hi3:+.2f}]")
        res["money"] = (pv.mean(), lo3, hi3)
    if scored:
        p1 = lo1 > 0 and lo2 > 0
        p2 = res["money"][1] > 0
        lines.append(f"  PRIMARY (structure beats pullback AND random): {'PASS' if p1 else 'FAIL'}   MONEY: {'PASS' if p2 else 'FAIL'}")
    return lines, res


def discovery_choice(disc):
    fb = [r for r in disc if r["struct"] == "fb"]
    med_speed = float(np.median([r["speed"] for r in fb]))
    means = {}
    for tg in ("T1", "T2", "T3", "T4"):
        v = [(r["T4"] if r["speed"] > med_speed else r["T1"]) if tg == "T4" else r[tg] for r in fb]
        v = [x for x in v if x is not None]
        means[tg] = np.mean(v)
    return max(means, key=means.get), med_speed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("build", "discovery", "holdout"):
        ap.add_argument(f"--{a}", action="store_true")
    a = ap.parse_args()
    if a.build:
        B.pull_tide_range(DISC[0], HOLD[1])
        days = [d for d in L.lake_days(DISC[0], HOLD[1])
                if glob.glob(os.path.join(L.LAKE, "silver", "trades-core", f"date={d}", "*.parquet"))]
        with ProcessPoolExecutor(6) as ex:
            for k, (d, msg) in enumerate(ex.map(build_day, days)):
                if k % 50 == 0:
                    print(k, d, msg, flush=True)
        print("built", len(days))
    if a.discovery:
        disc = load(*DISC)
        tg, ms = discovery_choice(disc)
        lines, _ = report(disc, f"DISCOVERY {DISC[0]}..{DISC[1]}", ms)
        print("\n".join(lines + [f"  -> target chosen for the holdout: {tg} (fast/slow split at speed {ms:.2e})"]))
    if a.holdout:
        res = os.path.join(CACHE, "RESULT-holdout.txt")
        if os.path.exists(res):
            sys.exit(f"the holdout was scored already -- see {res}")
        tg, ms = discovery_choice(load(*DISC))
        lines, _ = report(load(*HOLD), f"HOLDOUT {HOLD[0]}..{HOLD[1]} scored {dt.datetime.now():%Y-%m-%d %H:%M} (target {tg})",
                          ms, target=tg, scored=True)
        print("\n".join(lines))
        open(res, "w").write("\n".join(lines) + "\n")
